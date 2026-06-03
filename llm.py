import os
import time
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq

# ==========================================
# 1. Load Environment & Collect Keys
# ==========================================
load_dotenv()

# تجميع كل مفاتيح جوجل اللي في ملف .env في لستة واحدة
GEMINI_KEYS = []
for key, value in os.environ.items():
    if key.startswith("GOOGLE_API_KEY") and value.strip():
        # لو المستخدم حطهم بينهم فاصلة في متغير واحد
        if "," in value:
            GEMINI_KEYS.extend([k.strip() for k in value.split(",") if k.strip()])
        else:
            GEMINI_KEYS.append(value.strip())

if not GEMINI_KEYS:
    raise ValueError("❌ No GOOGLE_API_KEY found in .env file")

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
# 3. Multi-Model & Multi-Key Client
# ==========================================
class FallbackLLMClient:
    def __init__(self):
        self.gemini_keys = GEMINI_KEYS
        
        # Groq client (free tier available at console.groq.com)
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # 🌟 Primary Gemini models (tried in order)
        self.gemini_models = [
            "gemini-2.5-flash",   # Primary (very fast)
            "gemini-2.0-flash",   # First fallback
            "gemini-2.5-pro",     # Second fallback
        ]

        # ⚡ Groq fallback models
        self.groq_models = [
            "llama-3.3-70b-versatile",   # Best quality on free tier
            "llama-3.1-8b-instant",      # Fastest, lightest
            "mixtral-8x7b-32768",        # Large context window (32k)
        ]

    # ------------------------------------------
    # Internal: Call a single Gemini model
    # ------------------------------------------
    def _call_gemini(self, client, model_name: str, contents) -> str:
        response = client.models.generate_content(
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
    # Text-only completion with Full Fallback Chain
    # ------------------------------------------
    def get_response(self, prompt: str) -> str:
        last_error = ""

        # --- Try Gemini Keys and Models First ---
        for key_idx, current_key in enumerate(self.gemini_keys):
            # تهيئة العميل بالمفتاح الحالي
            gemini_client = genai.Client(api_key=current_key)
            print(f"🔑 LLM Routing: Active Gemini Key [{key_idx + 1}/{len(self.gemini_keys)}]")

            for model_name in self.gemini_models:
                try:
                    print(f"  🔄 Trying text model [{model_name}]...")
                    result = self._call_gemini(gemini_client, model_name, prompt)
                    print(f"  ✅ Success with [{model_name}] using Key {key_idx + 1}")
                    return result

                except Exception as e:
                    error_msg = str(e)
                    print(f"  ⚠️ Model [{model_name}] failed: {error_msg}")
                    last_error = error_msg

                    if is_token_or_quota_error(error_msg):
                        print(f"  🔁 Quota/Rate Limit hit on Key {key_idx + 1}. Switching to next key...")
                        break  # بيكسر لوب الموديلات وبيروح للـ Key اللي بعده فوراً
                    
                    time.sleep(2)

        # --- All Gemini Keys/Models failed; try Groq ---
        print("⚡ All Gemini keys/models failed. Switching to Groq fallback...")
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
    # Multimodal (Text + Image) with Full Fallback Chain
    # ------------------------------------------
    def get_vision_response(self, prompt: str, base64_data: str, mime_type: str = "image/jpeg") -> str:
        last_error = ""

        # --- Try Gemini Keys and Vision Models First ---
        for key_idx, current_key in enumerate(self.gemini_keys):
            gemini_client = genai.Client(api_key=current_key)
            print(f"🔑 Vision Routing: Active Gemini Key [{key_idx + 1}/{len(self.gemini_keys)}]")

            for model_name in self.gemini_models:
                try:
                    print(f"  🔄 Trying vision model [{model_name}]...")
                    image_bytes = base64.b64decode(base64_data)
                    contents = [
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt
                    ]
                    result = self._call_gemini(gemini_client, model_name, contents)
                    print(f"  ✅ Success with [{model_name}] using Key {key_idx + 1}")
                    return result

                except Exception as e:
                    error_msg = str(e)
                    print(f"  ⚠️ Vision Model [{model_name}] failed: {error_msg}")
                    last_error = error_msg

                    if is_token_or_quota_error(error_msg):
                        print(f"  🔁 Quota/Rate Limit hit on Key {key_idx + 1}. Switching to next key...")
                        break  # كسر اللوب وتجربة المفتاح التالي
                    
                    time.sleep(2)

        raise RuntimeError(
            f"All Gemini vision models and keys failed, and Groq does not support vision inputs. "
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