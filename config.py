import os
from dotenv import load_dotenv

load_dotenv()

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

if not SUPABASE_URL or SUPABASE_URL == "YOUR_URL":
    raise ValueError("SUPABASE_URL is not set. Add it to your .env file.")
if not SUPABASE_KEY or SUPABASE_KEY == "YOUR_KEY":
    raise ValueError("SUPABASE_KEY is not set. Add it to your .env file.")

# ── Recognition tuning ────────────────────────────────────────────────────────
THRESHOLD    = float(os.getenv("THRESHOLD",    0.58))  # cosine similarity cutoff
CONFIRM_TIME = float(os.getenv("CONFIRM_TIME", 2.0))   # seconds face must match before confirming

# ── Audio alerts ──────────────────────────────────────────────────────────────
BEEP_DELAY    = float(os.getenv("BEEP_DELAY",    2.0))  # seconds after confirm before beep
BEEP_COOLDOWN = float(os.getenv("BEEP_COOLDOWN", 5.0))  # minimum seconds between beeps