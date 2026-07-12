import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Optional SOCKS5/HTTP Proxy URL for Telegram Bot (e.g. socks5://user:pass@host:port)
TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY")

# LLM Providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Allowed Telegram User IDs (comma-separated, e.g. 123456,7891011). If empty, allows everyone.
allowed_ids_raw = os.getenv("ALLOWED_TELEGRAM_IDS", "")
ALLOWED_TELEGRAM_IDS = [
    int(x.strip()) for p in allowed_ids_raw.split(",") if (x := p.strip()).isdigit()
]

# Storage configuration (yandex / google)
STORAGE_PROVIDER = os.getenv("STORAGE_PROVIDER", "yandex").lower()

# Yandex WebDAV Configuration
YANDEX_USER = os.getenv("YANDEX_USER")
YANDEX_PASSWORD = os.getenv("YANDEX_PASSWORD")
YANDEX_OBSIDIAN_DIR = os.getenv("YANDEX_OBSIDIAN_DIR", "/ObsidianVault/Day-plans").strip()

# Google Drive Configuration
# Path to service account key JSON file, or inline JSON string
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
# Google Drive Folder ID to upload the daily plans into
GOOGLE_OBSIDIAN_DIR_ID = os.getenv("GOOGLE_OBSIDIAN_DIR_ID")

# Google Calendar Configuration
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Moscow")

