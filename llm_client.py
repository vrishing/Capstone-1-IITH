"""
Single entry point for Gemini calls that works on any machine.

Tries the new SDK (google-genai -> `from google import genai`) first and
falls back to the old one (google-generativeai -> `import google.generativeai`).
Scripts should only call `generate(...)`; they never import either SDK.
"""
from __future__ import annotations

BACKEND: str | None = None

try:
    from google import genai as _new_genai
    from google.genai import types as _new_types
    BACKEND = "google-genai"
except ImportError:
    _new_genai = None
    _new_types = None

if BACKEND is None:
    try:
        import google.generativeai as _old_genai
        BACKEND = "google-generativeai"
    except ImportError:
        _old_genai = None

if BACKEND is None:
    raise ImportError(
        "No Gemini SDK found. Install one of: "
        "`pip install google-genai` (preferred) or `pip install google-generativeai`."
    )

_client_cache: dict = {}


def generate(
    prompt: str,
    *,
    model: str,
    api_key: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1000,
    json_output: bool = False,
) -> str:
    """Return the model's text response using whichever SDK is installed.

    json_output=True asks the API for a JSON response (parse it with json.loads).
    """
    if BACKEND == "google-genai":
        client = _client_cache.get(api_key)
        if client is None:
            client = _new_genai.Client(api_key=api_key)
            _client_cache[api_key] = client
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_new_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **({"response_mime_type": "application/json"} if json_output else {}),
            ),
        )
        return response.text

    # Old SDK path (needs google-generativeai >= 0.5 for system_instruction)
    _old_genai.configure(api_key=api_key)
    gm = _old_genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
    )
    response = gm.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            **({"response_mime_type": "application/json"} if json_output else {}),
        },
    )
    return response.text


if __name__ == "__main__":
    print(f"Gemini backend in use: {BACKEND}")
