from pathlib import Path
from dotenv import load_dotenv
import os
import time
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from deepeval.models import DeepEvalBaseLLM

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

print("GEMINI KEY:", os.getenv("GEMINI_API_KEY")[:10])

class GeminiModel(DeepEvalBaseLLM):

    def __init__(self):
        genai.configure(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model_name = os.getenv(
            "GEMINI_MODEL_NAME",
            "models/gemini-flash-lite-latest"
        )
        print("GEMINI MODEL:", self.model_name)

        self.model = genai.GenerativeModel(
            self.model_name
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str):
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"❌ Max retries ({max_retries}) exceeded. Please try again later.")
                    raise
                
                # Extract retry delay from error message if available
                error_str = str(e)
                wait_time = 50  # default wait time
                
                if "retry in" in error_str:
                    try:
                        import re
                        match = re.search(r'retry in (\d+\.?\d*)', error_str)
                        if match:
                            wait_time = int(float(match.group(1))) + 5
                    except:
                        pass
                
                print(f"⏳ Rate limit hit. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(wait_time)

    async def a_generate(self, prompt: str):
        return self.generate(prompt)

    def get_model_name(self):
        return "gemini-1.5-flash"