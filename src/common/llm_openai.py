from openai import OpenAI
from src.common.config import get_openai_key

def chat_answer(system: str, user: str) -> str:
    client = OpenAI(api_key=get_openai_key())
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
