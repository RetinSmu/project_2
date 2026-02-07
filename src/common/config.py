import os
from dotenv import load_dotenv

load_dotenv()

def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
    return key
