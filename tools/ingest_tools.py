"""Unified file ingestion tool — accept ANY file Jarvis is given.

Dispatches by extension:
- text/code      -> plain read
- pdf            -> pypdf text extraction (page-range aware)
- images         -> NVIDIA vision OCR (photographed textbook/handwritten pages)
- audio          -> faster-whisper transcription (mp3/wav/m4a/... via ffmpeg)
- office         -> python-docx / openpyxl / python-pptx (docx/xlsx/pptx)
- unknown        -> safe binary summary (no crash)

Large files are capped (JARVIS_INGEST_MAX_CHARS) and summarized via the
internal text-only LLM so homework documents never blow the model context.
"""

import os

INGEST_MAX_CHARS = int(os.getenv("JARVIS_INGEST_MAX_CHARS", "6000"))
INGEST_MAX_BYTES = int(os.getenv("JARVIS_INGEST_MAX_BYTES", str(50 * 1024 * 1024)))

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift",
    ".sh", ".bash", ".zsh", ".fish", ".json", ".jsonl", ".csv", ".tsv",
    ".html", ".htm", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".log", ".sql", ".css", ".scss", ".r", ".jl", ".lua", ".pl", ".kt",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aiff", ".aif"}
OFFICE_EXTS = {".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx",
               ".xls": "xls", ".doc": "doc", ".ppt": "ppt"}


def _extract_pdf(path: str, page=None, page_count=None) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    total = len(reader.pages)
    start = max(0, int(page or 0))
    count = int(page_count or 0) or (total - start)
    parts = []
    for i in range(start, min(start + count, total)):
        t = reader.pages[i].extract_text()
        if t and t.strip():
            parts.append(f"--- page {i + 1} ---\n{t.strip()}")
    text = "\n".join(parts)
    if not text.strip():
        text = ("(no extractable text — this PDF may be scanned images. "
                "Convert a page to PNG/JPG and ingest that for OCR, or ask me to OCR it.)")
    if start > 0 or count < total - start:
        text += f"\n[showing pages {start + 1}–{min(start + count, total)} of {total}]"
    return text


def _extract_image(path: str, question: str = "") -> str:
    from tools.vision_tools import ocr_document

    q = question or "Extract all visible text from this image. Return only the text."
    result = ocr_document(path)
    if result and "not available" not in result and "not configured" not in result:
        return result
    from tools.vision_tools import analyze_image

    return analyze_image(path=path, question=q)


def _extract_audio(path: str) -> str:
    from stt import transcribe_file

    text = transcribe_file(path)
    if not text.strip():
        return "(transcription returned no text — unsupported audio codec? install ffmpeg.)"
    return text


def _extract_office(path: str, kind: str) -> str:
    if kind == "docx":
        import docx

        doc = docx.Document(path)
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    if kind == "xlsx":
        import openpyxl

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        out = []
        for ws in wb.worksheets:
            out.append(f"--- sheet: {ws.title} ---")
            for row in ws.iter_rows(values_only=True):
                vals = [str(v) if v is not None else "" for v in row]
                if any(vals):
                    out.append(" | ".join(vals))
        return "\n".join(out)
    if kind == "pptx":
        import pptx

        prs = pptx.Presentation(path)
        out = []
        for i, slide in enumerate(prs.slides, 1):
            out.append(f"--- slide {i} ---")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    out.append(shape.text)
        return "\n".join(out)
    return ("(legacy Office format — open it in the app and save as docx/xlsx/pptx "
            "so I can read it.)")


def _extract_text(path: str, ext: str, page=None, page_count=None, question: str = "") -> str | None:
    if ext in TEXT_EXTS:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    if ext == ".pdf":
        return _extract_pdf(path, page=page, page_count=page_count)
    if ext in IMAGE_EXTS:
        return _extract_image(path, question)
    if ext in AUDIO_EXTS:
        return _extract_audio(path)
    if ext in OFFICE_EXTS:
        return _extract_office(path, OFFICE_EXTS[ext])
    return None


def _read_binary_head(path: str) -> str:
    try:
        with open(path, "rb") as f:
            head = f.read(256)
        printable = "".join(chr(b) if 32 <= b < 127 else "." for b in head)
        return f"Size {os.path.getsize(path)} bytes. First bytes: {printable!r}"
    except Exception as e:
        return f"Size {os.path.getsize(path)} bytes. (could not preview: {e})"


def _summarize(text: str) -> str:
    import brain

    prompt = (
        "You are Jarvis's document summarizer. Give a concise bulleted outline "
        "(max 250 words) of the document below. Preserve key facts, numbers, and "
        "any explicit questions/tasks the document asks.\n\nDOCUMENT:\n"
        f"{text[:30000]}"
    )
    try:
        return brain.ask_llm_internal(prompt).strip()
    except Exception as e:
        return f"(summary unavailable: {e})"


def ingest_file(path: str = "", max_chars: int | None = None, summarize: bool | None = None,
                page: int | None = None, page_count: int | None = None,
                offset: int | None = None, question: str = "") -> str:
    """Read / OCR / transcribe any local file and return its text (capped + summarized)."""
    if not path:
        return "Provide a file path."
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"File not found: {path}"
    if os.path.isdir(path):
        return f"{path} is a directory — give me a file path."
    size = os.path.getsize(path)
    if size > INGEST_MAX_BYTES:
        return f"File too large ({size} bytes, limit {INGEST_MAX_BYTES})."
    ext = os.path.splitext(path)[1].lower()
    cap = int(max_chars or INGEST_MAX_CHARS)

    try:
        text = _extract_text(path, ext, page=page, page_count=page_count, question=question)
    except Exception as e:
        return f"Could not ingest {os.path.basename(path)} ({ext}): {e}"

    if text is None:
        return (f"Unsupported file type: {ext}. {_read_binary_head(path)}\n"
                f"Known types: text/code, pdf, images (png/jpg/...), audio (mp3/wav/m4a/...), "
                f"docx/xlsx/pptx.")
    if not text.strip():
        return f"{os.path.basename(path)} produced no readable text (binary or empty?)."

    header = f"[ingested {os.path.basename(path)} ({ext or '?'}, {size} bytes)]\n"
    if offset:
        lines = text.splitlines()
        start = max(0, int(offset))
        text = "\n".join(lines[start:start + 500])
        return f"{header}{text}"

    if len(text) > cap:
        preview = text[:cap]
        summary = _summarize(text) if (summarize is None or summarize) else ""
        note = (f"\n\n[file has {len(text)} chars; showing the first {cap}. "
                f"{('Remainder summary:\n' + summary) if summary else 'Ask me to read more.'}]")
        return f"{header}{preview}{note}"
    return f"{header}{text}"


INGEST_TOOLS = {
    "ingest_file": ingest_file,
}

INGEST_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "ingest_file",
            "description": (
                "Read, OCR, or transcribe any local file and return its text content: "
                "text/code, PDF (page-range aware), images (photos of pages / handwritten notes via OCR), "
                "audio (mp3/wav/m4a lectures via speech-to-text), and Word/Excel/PowerPoint. "
                "Use this for homework files, scanned documents, and recordings the user provides. "
                "Large files are capped and summarized automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "max_chars": {"type": "integer", "description": "Preview character cap (default 6000)"},
                    "summarize": {"type": "boolean", "description": "Whether to summarize the portion beyond the cap"},
                    "page": {"type": "integer", "description": "For PDFs: 0-based start page"},
                    "page_count": {"type": "integer", "description": "For PDFs: how many pages to read"},
                    "offset": {"type": "integer", "description": "For text: line offset to continue reading"},
                    "question": {"type": "string", "description": "For images: a specific question about the image"},
                },
                "required": ["path"],
            },
        },
    },
]
