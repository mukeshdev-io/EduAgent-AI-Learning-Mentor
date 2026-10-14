# llm.py
# -------
# Single entry-point for all LLM calls.
# Change the provider here; nothing else in the codebase needs to change.
#
# Supported providers (set LLM_PROVIDER in .env):
#   "groq"   → Groq API  (free tier, llama-3.3-70b-versatile)
#   "gemini" → Google Gemini API (free tier, gemini-1.5-flash)
#
# The function call_llm(prompt, json_mode=False) is the ONLY public symbol
# used by the rest of the app.

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ── read config ────────────────────────────────────────────────────────────────
PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_RETRIES = 3          # retry on transient API errors
RETRY_DELAY = 2          # seconds between retries


# ── Groq ───────────────────────────────────────────────────────────────────────
def _call_groq(prompt: str, json_mode: bool) -> str:
    """Call Groq chat completions API using the groq SDK."""
    from groq import Groq  # imported lazily so unused providers don't break

    client = Groq(api_key=GROQ_API_KEY)
    kwargs = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ── Gemini ─────────────────────────────────────────────────────────────────────
def _call_gemini(prompt: str, json_mode: bool) -> str:
    """Call Google Gemini API using google-generativeai SDK."""
    import google.generativeai as genai  # imported lazily

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=2048,
            # Ask Gemini for JSON when needed
            response_mime_type="application/json" if json_mode else "text/plain",
        ),
    )
    response = model.generate_content(prompt)
    return response.text


# ── Public function ────────────────────────────────────────────────────────────
def call_llm(prompt: str, json_mode: bool = False) -> str:
    """
    Send a prompt to the configured LLM provider and return the text response.

    Args:
        prompt:    The instruction / question to send.
        json_mode: When True, tell the model to respond with strict JSON.
                   The caller is still responsible for parsing and validating.

    Returns:
        The model's text response as a string.

    Raises:
        RuntimeError: if all retries fail.
    """
    _call_fn = {"groq": _call_groq, "gemini": _call_gemini}.get(PROVIDER)
    if _call_fn is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{PROVIDER}'. "
            "Set LLM_PROVIDER=groq or LLM_PROVIDER=gemini in your .env file."
        )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _call_fn(prompt, json_mode)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # back-off: 2s, 4s
            # on the final attempt fall through and raise

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}\n"
        "Check your API key and internet connection."
    ) from last_error
