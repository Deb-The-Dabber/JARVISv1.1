import base64
import datetime
import os
import subprocess
import tempfile

import requests
from dotenv import load_dotenv

HOME = os.path.expanduser("~")
load_dotenv()

_DEBUG = os.getenv("JARVIS_DEBUG", "1").lower() in ("1", "true", "yes", "on")


def _vision_debug(message: str):
    if _DEBUG:
        print(f"[vision] {message}")


def get_frontmost_app() -> str:
    script = 'tell application "System Events" to get name of first process whose frontmost is true'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return result.stdout.strip()


def _maybe_focus_app(app_name: str = ""):
    name = (app_name or "").strip()
    if not name:
        return None
    from tools.system_tools import focus_app

    return focus_app(name)


def capture_screenshot() -> str:
    path = os.path.join(tempfile.gettempdir(), f"jarvis_screen_{int(datetime.datetime.now().timestamp())}.png")
    subprocess.run(["screencapture", "-x", path], check=True)
    return path


def describe_screen(question: str = "What is on this screen? Be concise.") -> str:
    path = None
    try:
        path = capture_screenshot()
        with open(path, "rb") as f:
            image_bytes = f.read()

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{b64}"

        api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
        if not api_key:
            return "Screen analysis unavailable."

        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        payload = {
            "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        _vision_debug(f"NVIDIA status={resp.status_code}")
        if resp.status_code != 200:
            return "Screen analysis unavailable."

        data = resp.json()
        message = data.get("choices", [])[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        return "Screen analysis unavailable."
    except Exception:
        return "Screen analysis unavailable."
    finally:
        try:
            if path:
                os.remove(path)
        except Exception:
            pass


def read_screen(app_name: str = ""):
    focus_msg = _maybe_focus_app(app_name)
    app = get_frontmost_app()
    description = describe_screen("What is on this screen? Be concise.")
    prefix = f"{focus_msg} " if focus_msg else ""
    return f"{prefix}Frontmost app: {app}. {description}"


def find_on_screen(what: str, app_name: str = ""):
    focus_msg = _maybe_focus_app(app_name)
    description = describe_screen(f"Find '{what}' on this screen. If it is a sidebar list item, say roughly how many items from the top and whether a click is needed. Do not invent shell commands.")
    prefix = f"{focus_msg} " if focus_msg else ""
    return f"{prefix}{description}"


def summarize_screen(app_name: str = ""):
    focus_msg = _maybe_focus_app(app_name)
    app = get_frontmost_app()
    description = describe_screen("Briefly describe what's on this screen in 1-2 sentences.")
    prefix = f"{focus_msg} " if focus_msg else ""
    return f"{prefix}{app}: {description}"


def check_screen_for_alerts(app_name: str = ""):
    _maybe_focus_app(app_name)
    return describe_screen("Is there anything urgent on screen? Look for errors, alerts, notifications. If nothing urgent say 'Nothing urgent.' Be brief.")


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _nvidia_vision_query(image_b64: str, question: str) -> str:
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        return "Vision API key not configured."
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        return f"Vision API error: {resp.status_code}"
    data = resp.json()
    msg = data.get("choices", [{}])[0].get("message", {})
    return msg.get("content", "No response.") or ""


def analyze_image(path: str = "", url: str = "", question: str = "Describe this image.") -> str:
    if path:
        if not os.path.exists(path):
            return f"File not found: {path}"
        b64 = _encode_image(path)
    elif url:
        import requests as req

        resp = req.get(url, timeout=30)
        if resp.status_code != 200:
            return f"Could not download image from {url}"
        b64 = base64.b64encode(resp.content).decode("utf-8")
    else:
        return "Provide a file path or URL."
    return _nvidia_vision_query(b64, question)


def ocr_document(path: str) -> str:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pdf":
        try:
            import pypdf

            text = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        text.append(t)
            if text:
                return "\n".join(text)[:2000]
        except ImportError:
            pass
        return "OCR: PDF text extraction library not available (try: pip install pypdf)"
    if ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"):
        b64 = _encode_image(path)
        return _nvidia_vision_query(b64, "Extract all visible text from this image. Return only the text.")

    from config import SUPPORTED_EXTENSIONS

    if ext in SUPPORTED_EXTENSIONS:
        try:
            with open(path) as f:
                return f.read()[:2000]
        except Exception as e:
            return f"Could not read file: {e}"

    return f"Unsupported file type: {ext}"


def analyze_video(path: str, timestamps: str = "0") -> str:
    import shutil

    if not shutil.which("ffmpeg"):
        return "ffmpeg not available — install with: brew install ffmpeg"
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        ts_list = [t.strip() for t in timestamps.split(",") if t.strip()]
        frames = []
        for ts in ts_list:
            out = tempfile.mktemp(suffix=".png")
            subprocess.run(
                ["ffmpeg", "-y", "-ss", ts, "-i", path, "-vframes", "1", "-q:v", "2", out],
                capture_output=True,
                timeout=30,
            )
            if os.path.exists(out):
                b64 = _encode_image(out)
                result = _nvidia_vision_query(b64, f"Describe the video frame at timestamp {ts}.")
                frames.append(f"[{ts}] {result}")
                try:
                    os.remove(out)
                except Exception:
                    pass
        if frames:
            return "\n".join(frames)
        return "Could not extract any frames from the video."
    except Exception as e:
        return f"Video analysis error: {e}"


VISION_TOOLS = {
    "read_screen": read_screen,
    "find_on_screen": find_on_screen,
    "summarize_screen": summarize_screen,
    "check_screen_for_alerts": check_screen_for_alerts,
    "analyze_image": analyze_image,
    "ocr_document": ocr_document,
    "analyze_video": analyze_video,
}

VISION_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_screen",
            "description": "Screenshot and describe the screen. Pass app_name (e.g. Safari) to focus that app first.",
            "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_on_screen",
            "description": "Describe where something is on screen (does not click). Pass app_name to focus that app first.",
            "parameters": {
                "type": "object",
                "properties": {"what": {"type": "string"}, "app_name": {"type": "string"}},
                "required": ["what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_screen",
            "description": "Brief summary of what's on screen. Pass app_name to focus that app first.",
            "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_screen_for_alerts",
            "description": "Screenshot the screen and check for errors, alerts, or notifications. Pass app_name to focus that app first.",
            "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze an image file (path or URL) — describe contents, extract info, answer questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local file path to the image"},
                    "url": {"type": "string", "description": "URL of the image"},
                    "question": {"type": "string", "description": "Question about the image (optional)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ocr_document",
            "description": "Extract text from a document (PDF, image, or text file). Returns up to 2000 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the document file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_video",
            "description": "Analyze one or more frames from a video file using ffmpeg. Provide timestamps in seconds (comma-separated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the video file"},
                    "timestamps": {"type": "string", "description": "Comma-separated timestamps in seconds (e.g. 0,30,60)"},
                },
                "required": ["path"],
            },
        },
    },
]
