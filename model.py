from dotenv import load_dotenv
import os
import google.generativeai as genai

class GeminiModel:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file.")
        genai.configure(api_key=self.api_key)

        generation_config = {
            "temperature": 1,
            "top_p": 1,
            "top_k": 1,
        }

        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
            safety_settings=safety_settings
        )
    
    def configureModel(self):
        return self.model

class ModelGeneration:
    def __init__(self):
        pass

    def generate_explanation(self, model, query, code_text=None):
        if not code_text:
            prompt = f"""You are a code assistant. A user asked: "{query}"
            Convert the code as per the user's request. Only provide the code.

            Note: Don't provide any explanation or formatting like triple backticks. Just return the raw code without mentioning the language."""
        else:
            prompt = f"""You are a code assistant. A user asked: "{query}"
            Here is the relevent code which was found {code_text}
            Convert the code as per the user's request. Only provide the code.

            Note: Don't provide any explanation or formatting like triple backticks. Just return the raw code without mentioning the language."""
        response = model.generate_content(prompt)
        return response.text
