# openai_client.py
from functools import lru_cache
from pathlib import Path
from openai import OpenAI
import json, os, re


def _ensure_openai_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    repo_root = Path(__file__).resolve().parents[2]
    search_paths = [
        repo_root / ".env",
        Path.cwd() / ".env",
    ]

    for env_path in search_paths:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if not line or line.strip().startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    os.environ.setdefault("OPENAI_API_KEY", value.strip().strip('"').strip("'"))
                    return
        except OSError:
            continue

    raise RuntimeError(
        "OPENAI_API_KEY not set. Please export it or add it to your .env file."
    )


_ensure_openai_api_key()
client = OpenAI()

QUESTIONIFY_SYS = "You convert a quoted answer into one concise question. Return ONLY JSON: {\"question\":\"...\"}."

@lru_cache(maxsize=2048)
def _questionify_cached(answer_snippet: str, context: str) -> str | None:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": QUESTIONIFY_SYS},
            {"role": "user", "content":
             f'Answer:\\n\"\"\"{answer_snippet}\"\"\"\\n\\nContext:\\n\"\"\"{context}\"\"\"\\nReturn only {{\"question\":\"...\"}}'}
        ],
        temperature=0.2, max_tokens=120
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        return (json.loads(txt).get("question") or "").strip()
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m: return None
        try:
            return (json.loads(m.group(0)).get("question") or "").strip()
        except Exception:
            return None

def questionify(answer_snippet: str, context: str) -> dict | None:
    q = _questionify_cached(answer_snippet, context)
    return {"question": q} if q else None
