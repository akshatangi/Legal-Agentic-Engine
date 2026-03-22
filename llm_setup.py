"""
llm_setup.py — Multi-key LLM setup with round-robin load balancing.

Keys used:
  - GROQ_API_KEY_1    : Your Groq account
  - GROQ_API_KEY_2    : Friend's Groq account
  - OPENROUTER_API_KEY: OpenRouter (fallback + overflow)

Round-robin assigns each agent call to a different key,
distributing TPM load across all three accounts.
"""

import os
import time
import threading
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

# ─────────────────────────────────────────────
# .env should contain:
#   GROQ_API_KEY_1=gsk_...
#   GROQ_API_KEY_2=gsk_...
#   OPENROUTER_API_KEY=sk-or-...
# ─────────────────────────────────────────────

GROQ_MODEL       = "llama-3.3-70b-versatile"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct"


def _make_groq(api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        api_key=api_key,
        max_retries=3,
        request_timeout=60,
    )


def _make_openrouter(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        temperature=temperature,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
        request_timeout=60,
    )


# ─────────────────────────────────────────────
# Round-Robin Counter — thread-safe
# 0 → Groq Key 1 | 1 → Groq Key 2 | 2 → OpenRouter
# ─────────────────────────────────────────────

_counter_lock = threading.Lock()
_counter = 0


def _next_index() -> int:
    global _counter
    with _counter_lock:
        idx = _counter % 3
        _counter += 1
    return idx


def get_llm(temperature: float = 0.0):
    """
    Returns the next LLM in the round-robin rotation.
    Each call to get_llm() gets a different key than the last.

    Rotation:
      Agent 01 (Metadata)   → Groq Key 1
      Agent 02 (Facts)      → Groq Key 2
      Agent 03 (Issues)     → OpenRouter
      Agent 04 (Statutes)   → Groq Key 1
      Agent 05 (Reasoning)  → Groq Key 2
      Agent 06 (Judgment)   → OpenRouter
      Agent 07 (Headnote)   → Groq Key 1
    """
    idx = _next_index()

    groq_key_1     = os.getenv("GROQ_API_KEY_1")
    groq_key_2     = os.getenv("GROQ_API_KEY_2")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if idx == 0 and groq_key_1:
        print(f"    [LLM Router] Groq Key 1")
        return _make_groq(groq_key_1, temperature)
    elif idx == 1 and groq_key_2:
        print(f"    [LLM Router] Groq Key 2")
        return _make_groq(groq_key_2, temperature)
    elif openrouter_key:
        print(f"    [LLM Router] OpenRouter")
        return _make_openrouter(temperature)
    elif groq_key_1:
        print(f"    [LLM Router] Groq Key 1 (fallback)")
        return _make_groq(groq_key_1, temperature)
    else:
        raise ValueError("No API keys found. Check your .env file.")


def call_with_retry(chain, inputs: dict, max_retries: int = 5, base_wait: float = 30.0):
    """
    Wraps any LangChain chain call with retry + automatic key rotation on 429.
    On rate limit: waits the exact time the provider specifies, then retries
    with the next key in rotation automatically.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait = base_wait
                try:
                    import re
                    match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                    if match:
                        wait = float(match.group(1)) + 2
                except Exception:
                    pass
                print(f"    [Rate Limit] Waiting {wait:.0f}s then retrying with next key... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"All {max_retries} retries failed.")