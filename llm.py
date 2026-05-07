import os
import time
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

# ==========================================
# 1. Load Environment
# ==========================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

# ==========================================
# 2. Token/Quota Error Detection
# ==========================================
TOKEN_ERROR_KEYWORDS = [
    "quota",
    "rate_limit",
    "rate limit",
    "resource_exhausted",
    "resource exhausted",
    "tokens",
    "token limit",
    "context length",
    "too many requests",
    "429",
    "overloaded",
]

def is_token_or_quota_error(error_message: str) -> bool:
    """Detect if the error is related to token limits or quota exhaustion."""
    msg = error_message.lower()
    return any(keyword in msg for keyword in TOKEN_ERROR_KEYWORDS)

# ==========================================
# 3. Multi-Model Client (Gemini + Groq Fallback)
# ==========================================
class FallbackLLMClient:
    def __init__(self):
        # Google Gemini client
        self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

        # Groq client (free tier available at console.groq.com)
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # 🌟 Primary Gemini models (tried in order)
        self.gemini_models = [
            "gemini-2.5-flash",   # Primary (very fast)
            "gemini-2.0-flash",   # First fallback (smarter, slightly slower)
            "gemini-2.5-pro",     # Second fallback
        ]

        # ⚡ Groq fallback models — all free tier, extremely fast inference
        # Note: Groq does NOT support image/vision inputs
        self.groq_models = [
            "llama-3.3-70b-versatile",   # Best quality on free tier
            "llama-3.1-8b-instant",      # Fastest, lightest
            "mixtral-8x7b-32768",        # Large context window (32k)
        ]

    # ------------------------------------------
    # Internal: Call a single Gemini model
    # ------------------------------------------
    def _call_gemini(self, model_name: str, contents) -> str:
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=contents,
            config={"temperature": 0.2}
        )
        if response and response.text:
            return response.text.strip()
        raise ValueError("Empty response from Gemini")

    # ------------------------------------------
    # Internal: Call a single Groq model
    # ------------------------------------------
    def _call_groq(self, model_name: str, prompt: str) -> str:
        """Text-only — Groq does not support vision/image inputs."""
        response = self.groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=8096,
        )
        result = response.choices[0].message.content
        if result:
            return result.strip()
        raise ValueError("Empty response from Groq")

    # ------------------------------------------
    # Text-only completion with full fallback chain
    # ------------------------------------------
    def get_response(self, prompt: str) -> str:
        """Text-only completion: tries all Gemini models, then falls back to Groq."""
        last_error = ""
        gemini_exhausted = False

        # --- Try Gemini models first ---
        for model_name in self.gemini_models:
            try:
                print(f"🔄 LLM Routing: Trying text model [{model_name}]...")
                result = self._call_gemini(model_name, prompt)
                print(f"✅ Success with [{model_name}]")
                return result

            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Model [{model_name}] failed: {error_msg}")
                last_error = error_msg

                if is_token_or_quota_error(error_msg):
                    print(f"🔁 Token/Quota error detected on [{model_name}], switching model...")
                    gemini_exhausted = True

                time.sleep(2)

        # --- All Gemini models failed; try Groq ---
        if gemini_exhausted or last_error:
            print("⚡ All Gemini models failed. Switching to Groq fallback...")
            for groq_model in self.groq_models:
                try:
                    print(f"🔄 LLM Routing: Trying Groq model [{groq_model}]...")
                    result = self._call_groq(groq_model, prompt)
                    print(f"✅ Success with [{groq_model}]")
                    return result

                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ Groq model [{groq_model}] failed: {error_msg}")
                    last_error = error_msg
                    time.sleep(2)

        raise RuntimeError(f"All fallback models (Gemini + Groq) failed. Last error: {last_error}")

    # ------------------------------------------
    # Multimodal (Text + Image) with fallback chain
    # ------------------------------------------
    def get_vision_response(self, prompt: str, base64_data: str, mime_type: str = "image/jpeg") -> str:
        """
        Multimodal completion (Text + Image).
        Tries all Gemini vision models first.
        ⚠️  Groq does NOT support vision — if all Gemini models fail, raises an error.
        """
        last_error = ""

        # --- Try Gemini vision models ---
        for model_name in self.gemini_models:
            try:
                print(f"🔄 LLM Routing: Trying vision model [{model_name}]...")
                image_bytes = base64.b64decode(base64_data)
                contents = [
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt
                ]
                result = self._call_gemini(model_name, contents)
                print(f"✅ Success with [{model_name}]")
                return result

            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ Vision Model [{model_name}] failed: {error_msg}")
                last_error = error_msg

                if is_token_or_quota_error(error_msg):
                    print(f"🔁 Token/Quota error detected on [{model_name}], switching model...")

                time.sleep(2)

        # Groq has no vision support — raise a clear error
        raise RuntimeError(
            f"All Gemini vision models failed and Groq does not support vision inputs. "
            f"Last error: {last_error}"
        )


# ==========================================
# 4. Singleton Instance & Global Helpers
# ==========================================
llm_client = FallbackLLMClient()

def get_response(prompt: str) -> str:
    return llm_client.get_response(prompt)

def get_vision_response(prompt: str, base64_data: str, mime_type: str = "image/jpeg") -> str:
    return llm_client.get_vision_response(prompt, base64_data, mime_type)