import os

from dotenv import load_dotenv

load_dotenv()

HOME = os.path.expanduser("~")
JARVIS_DIR = os.path.join(HOME, "Jarvis")
JARVIS_TTS_SILENT = os.getenv("JARVIS_TTS_SILENT", "0") == "1"
JARVIS_FORCE_PROVIDER = os.getenv("JARVIS_FORCE_PROVIDER", "").lower()
JARVIS_MOCK_PROVIDERS = os.getenv("JARVIS_MOCK_PROVIDERS", "0") == "1"

JARVIS_EMBEDDING = os.getenv("JARVIS_EMBEDDING", "nemo").lower().strip()
JARVIS_EMBED_BATCH = int(os.getenv("JARVIS_EMBED_BATCH", "64"))
JARVIS_EMBED_MAX_CHARS = int(os.getenv("JARVIS_EMBED_MAX_CHARS", "6000"))
JARVIS_INGEST_MAX_CHARS = int(os.getenv("JARVIS_INGEST_MAX_CHARS", "6000"))
JARVIS_INGEST_MAX_BYTES = int(os.getenv("JARVIS_INGEST_MAX_BYTES", str(50 * 1024 * 1024)))

USER_NAME = "Debasish"
USER_CITY = "Aurora"
USER_LAT = 41.7606
USER_LON = -88.3201
USER_TIMEZONE = "America/Chicago"

WAKE_WORD = "hey jarvis"
DEFAULT_BROWSER = os.getenv("JARVIS_DEFAULT_BROWSER", "Dia")

SAMPLE_RATE = 16000
RECORD_SAMPLE_RATE = 44100
RECORD_SECONDS = 6
MIC_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))
WHISPER_MODEL = "base"

RAG_FOLDER = os.path.expanduser(os.getenv("RAG_FOLDER", "~/Documents"))
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".py", ".js", ".png", ".jpg", ".jpeg",
                        ".csv", ".json", ".docx", ".xlsx", ".pptx", ".mp3", ".wav", ".m4a"}

MEMORY_DB_PATH = os.path.join(HOME, "jarvis_memory.db")
VECTOR_DB_PATH = os.path.join(HOME, "jarvis_vector_db")
AUDIT_DB_PATH = os.path.join(HOME, "jarvis_audit.db")
WATCHLOG_DB_PATH = os.path.join(HOME, "jarvis_watchlog.db")
PRIORITY_DB_PATH = os.path.join(HOME, "jarvis_priority.db")
EVAL_DB_PATH = os.path.join(HOME, "jarvis_eval.db")
GRAPH_PATH = os.path.join(HOME, "jarvis_graph.json")
PROCEDURES_PATH = os.path.join(HOME, "jarvis_procedures.json")
SELF_LOG_PATH = os.path.join(HOME, "jarvis_self_log.db")
GEMINI_USAGE_FILE = os.path.join(HOME, ".jarvis_gemini_usage.json")
LEARNED_TOOLS_DIR = os.path.join(HOME, "jarvis_learned_tools")
RAG_DB_PATH = os.path.join(HOME, "jarvis_rag_db")
SCREENSHOT_DIRS = [
    os.path.join(HOME, "Desktop"),
    os.path.join(HOME, "Pictures", "Screenshots"),
]

PROACTIVE_CHECK_INTERVAL = 30
CPU_THRESHOLD = 80
RAM_THRESHOLD = 85
DISK_FREE_MIN_GB = 10
DOWNLOADS_MAX_GB = 5
DESKTOP_MAX_FILES = 20
APP_OPEN_HOURS = 3
WEATHER_CHECK_MINS = 30
SCREEN_CHECK_MINS = 30
BATTERY_MIN_PCT = 20
MIN_SECONDS_BETWEEN_ALERTS = 30

GEMINI_DAILY_LIMIT = 20
GEMINI_MAX_TOOL_ROUNDS = 8

CACHE_WEATHER_TTL = 600
CACHE_SYSTEM_INFO_TTL = 300
CACHE_SEARCH_TTL = 300

# ── VAD (Voice Activity Detection) ──
VAD_AGGRESSIVENESS = 1
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 100
VAD_FRAME_MS = 30

# ── TTS ──
TTS_RETRY_MAX_ATTEMPTS = 3
TTS_RETRY_BASE_DELAY = 2.0
TTS_EDGE_FALLBACK = True
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-US-JennyNeural")
EDGE_TTS_VOICES = [
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-LibbyNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
]

# ── API Keys (Phase 3) ──
WEATHERAPI_API_KEY = os.getenv("WEATHERAPI_API_KEY", "")
VISUAL_CROSSING_API_KEY = os.getenv("VISUAL_CROSSING_API_KEY", "")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
NASA_API_KEY = os.getenv("NASA_API_KEY", "")

# Alpha Vantage rate limit (25 req/day)
ALPHA_VANTAGE_DAILY_LIMIT = 25

# ── Safety ──
SAFETY_PENDING_TTL = int(os.getenv("SAFETY_PENDING_TTL", "86400"))  # 24h default (per-session)

# ── Context Window Management ──
MODEL_CONTEXT_LIMITS = {
    "nvidia/nemotron-3-ultra-550b-a55b": 8192,
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning": 8192,
    "nvidia/nemotron-3-super-120b-a12b": 32768,
    # EOL-removal (2026-09-12): llama-3.3-nemotron-super-49b-v1.5,
    # nemotron-3-nano-30b-a3b, step-3.7-flash, and minimax-m3 all returned
    # 410 Gone on the live NIM endpoint — never route to them again.
    "openai/gpt-oss-20b": 32768,
    "deepseek-ai/deepseek-v4-flash-0731": 32768,
    "gemini-3.5-flash": 32768,
    "gemini-2.5-flash": 32768,
    "deepseek-ai/deepseek-v4-flash": 32768,
    "groq/compound-mini": 131072,
    "meta-llama/llama-3.3-70b-instruct:free": 32768,
    "openai": 32768,
}
MAX_EXPLICIT_MEMORIES = 10
MAX_SEMANTIC_MEMORIES = 5

# ── Tool Parser Self-Learning ──
PARSERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parsers")
FORMAT_LOG_PATH = os.path.join(PARSERS_DIR, "format_log.jsonl")

# ── Remote Access ──
JARVIS_API_KEY = os.getenv("JARVIS_API_KEY", "")

# ── Screen Capture ──
SCREEN_CAPTURE_INTERVAL = 10  # seconds between captures
SCREEN_BUFFER_SIZE = 5  # rolling buffer count

# ── Face Unlock ──
FACE_SIMILARITY_THRESHOLD = 0.85  # cosine similarity threshold for match

# ── OAuth Integrations ──
# Gmail OAuth (Google Cloud Console → Credentials → OAuth 2.0 Client ID)
GMAIL_CLIENT_ID = os.getenv("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET = os.getenv("GMAIL_CLIENT_SECRET", "")
GMAIL_REDIRECT_URI = os.getenv("GMAIL_REDIRECT_URI", "http://localhost:8002/oauth/callback/gmail")
GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/userinfo.email",
]

# GitHub OAuth (GitHub Settings → Developer Settings → OAuth Apps)
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8002/oauth/callback/github")
GITHUB_SCOPES = ["repo", "read:user", "read:org", "user:email"]

OAUTH_REDIRECT_BASE = os.getenv("OAUTH_REDIRECT_BASE", "http://localhost:8002")

# Google Drive OAuth (.env: DRIVE_CLIENT_ID / DRIVE_CLIENT_SECRET)
GOOGLE_DRIVE_CLIENT_ID = os.getenv("DRIVE_CLIENT_ID", os.getenv("GOOGLE_DRIVE_CLIENT_ID", ""))
GOOGLE_DRIVE_CLIENT_SECRET = os.getenv("DRIVE_CLIENT_SECRET", os.getenv("GOOGLE_DRIVE_CLIENT_SECRET", ""))
GOOGLE_DRIVE_REDIRECT_URI = os.getenv("GOOGLE_DRIVE_REDIRECT_URI", "http://localhost:8002/oauth/callback/google-drive")
GOOGLE_DRIVE_SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
]

# Google Sheets OAuth (.env: SHEETS_CLIENT_ID / SHEETS_CLIENT_SECRET)
GOOGLE_SHEETS_CLIENT_ID = os.getenv("SHEETS_CLIENT_ID", os.getenv("GOOGLE_SHEETS_CLIENT_ID", ""))
GOOGLE_SHEETS_CLIENT_SECRET = os.getenv("SHEETS_CLIENT_SECRET", os.getenv("GOOGLE_SHEETS_CLIENT_SECRET", ""))
GOOGLE_SHEETS_REDIRECT_URI = os.getenv("GOOGLE_SHEETS_REDIRECT_URI", "http://localhost:8002/oauth/callback/google-sheets")
GOOGLE_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.email",
]

# Google Docs OAuth (.env: DOCS_CLIENT_ID / DOCS_CLIENT_SECRET)
GOOGLE_DOCS_CLIENT_ID = os.getenv("DOCS_CLIENT_ID", os.getenv("GOOGLE_DOCS_CLIENT_ID", ""))
GOOGLE_DOCS_CLIENT_SECRET = os.getenv("DOCS_CLIENT_SECRET", os.getenv("GOOGLE_DOCS_CLIENT_SECRET", ""))
GOOGLE_DOCS_REDIRECT_URI = os.getenv("GOOGLE_DOCS_REDIRECT_URI", "http://localhost:8002/oauth/callback/google-docs")
GOOGLE_DOCS_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.email",
]

# Google Slides OAuth (.env: SLIDES_CLIENT_ID / SLIDES_CLIENT_SECRET)
GOOGLE_SLIDES_CLIENT_ID = os.getenv("SLIDES_CLIENT_ID", os.getenv("GOOGLE_SLIDES_CLIENT_ID", ""))
GOOGLE_SLIDES_CLIENT_SECRET = os.getenv("SLIDES_CLIENT_SECRET", os.getenv("GOOGLE_SLIDES_CLIENT_SECRET", ""))
GOOGLE_SLIDES_REDIRECT_URI = os.getenv("GOOGLE_SLIDES_REDIRECT_URI", "http://localhost:8002/oauth/callback/google-slides")
GOOGLE_SLIDES_SCOPES = [
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/presentations.readonly",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/userinfo.email",
]

# Google Forms OAuth (.env: FORMS_CLIENT_ID / FORMS_CLIENT_SECRET)
GOOGLE_FORMS_CLIENT_ID = os.getenv("FORMS_CLIENT_ID", os.getenv("GOOGLE_FORMS_CLIENT_ID", ""))
GOOGLE_FORMS_CLIENT_SECRET = os.getenv("FORMS_CLIENT_SECRET", os.getenv("GOOGLE_FORMS_CLIENT_SECRET", ""))
GOOGLE_FORMS_REDIRECT_URI = os.getenv("GOOGLE_FORMS_REDIRECT_URI", "http://localhost:8002/oauth/callback/google-forms")
GOOGLE_FORMS_SCOPES = [
    "https://www.googleapis.com/auth/forms.body",
    "https://www.googleapis.com/auth/forms.body.readonly",
    "https://www.googleapis.com/auth/forms.responses.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

# Email IMAP settings (.env: EMAIL_IMAP_HOST / EMAIL_IMAP_PORT / EMAIL_IMAP_SSL / EMAIL_USERNAME / EMAIL_PASSWORD)
EMAIL_IMAP_HOST = os.getenv("EMAIL_IMAP_HOST", "imap.gmail.com")
EMAIL_IMAP_PORT = int(os.getenv("EMAIL_IMAP_PORT", "993"))
EMAIL_IMAP_SSL = os.getenv("EMAIL_IMAP_SSL", "true").lower() == "true"
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")  # App password for Gmail/Outlook
EMAIL_DEFAULT_FOLDER = os.getenv("EMAIL_DEFAULT_FOLDER", "INBOX")
EMAIL_MAX_RESULTS = int(os.getenv("EMAIL_MAX_RESULTS", "20"))
EMAIL_FETCH_BODY = os.getenv("EMAIL_FETCH_BODY", "true").lower() == "true"
