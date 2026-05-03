import os
import base64
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ==========================================
# 1. Load Environment
# ==========================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file")

# ==========================================
# 2. Gemini Client Wrapper
# ==========================================
class GeminiClient:
    def __init__(self, model_name: str = "gemma-3-27b-it"):
        self.model_name = model_name
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def get_response(self, prompt: str) -> str:
        """Text-only completion"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"temperature": 0.2}
            )
            if not response or not response.text:
                return "Empty response from model"
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def get_vision_response(self, prompt: str, base64_data: str, mime_type: str = "image/jpeg") -> str:
        """Multimodal completion (Text + Image)"""
        try:
            image_bytes = base64.b64decode(base64_data)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt
                ],
                config={"temperature": 0.2}
            )
            if not response or not response.text:
                return "Empty response from model"
            return response.text.strip()
        except Exception as e:
            return f"Error: {str(e)}"

# ==========================================
# 3. Singleton Instance & Helpers
# ==========================================
gemini_client = GeminiClient()

def get_response(prompt: str) -> str:
    return gemini_client.get_response(prompt)

def get_vision_response(prompt: str, base64_data: str, mime_type: str = "image/jpeg") -> str:
    return gemini_client.get_vision_response(prompt, base64_data, mime_type)
