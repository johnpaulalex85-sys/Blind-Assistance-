import json
import logging
from PIL import Image
from google import genai
from config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not set in .env. Gemini Service will not work.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini Service initialized successfully.")

    def answer_question(self, question: str, frame, scene_json: dict) -> str:
        if not self.client:
            return "My cloud brain is not configured with an API key yet."

        logger.info("Generating answer using Gemini...")
        
        # Convert OpenCV frame (numpy array) to PIL Image
        # Assuming frame is BGR (OpenCV default), convert to RGB
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Remove large binary/base64 data from the context to avoid token limits
        clean_scene = {k: v for k, v in scene_json.items() if k != "depth_map_base64"}

        prompt = (
            "You are FRIDAY, a highly intelligent and helpful AI assistant for a visually impaired user.\n"
            f"The user is asking: '{question}'\n\n"
            "Answer the user's question directly, conversationally, and concisely (under 50 words).\n"
            "Use the provided image and scene data to inform your answer if the question relates to the environment.\n\n"
            f"Scene Data Context:\n{json.dumps(clean_scene, indent=2)}"
        )

        try:
            response = self.client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[pil_image, prompt]
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during Gemini generation: {e}")
            return "I'm having trouble connecting to my cloud brain right now."
