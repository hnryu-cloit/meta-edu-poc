import os
import time
import base64

from google import genai
from dotenv import load_dotenv

import vertexai

def encode_image_to_base64(file_path: str) -> str:
    """로컬 이미지 파일을 Base64로 인코딩하는 함수"""
    try:
        with open(file_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None

def load_image_bytes(file_path: str) -> bytes:
    """로컬 이미지 파일을 읽어 원본 바이트를 반환하는 함수"""
    try:
        with open(file_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        print(f"오류: 파일 '{file_path}'을(를) 찾을 수 없습니다.")
        return None


class Gemini:
    
    def __init__(self):
        load_dotenv()
        vertexai.init(project=os.getenv('PROJECT_ID'), location=os.getenv('LOCATION'))
        self.api_key = os.getenv('API_KEY')

        self.client = genai.Client(api_key=self.api_key)

        self.model = "gemini-2.5-flash" #'gemini-3-pro-preview'
        self.max_retries = 3
        self.initial_delay = 1

        # logger 초기화 추가
        from common import logger
        self.logger = logger.init_logger()

    def retry_with_delay(func):
        """재시도 데코레이터"""
        def wrapper(self, *args, **kwargs):
            delay = self.initial_delay
            for attempt in range(self.max_retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise e
                    self.logger.error(f"gemini 호출 {attempt + 1}번째 실패: {e}")
                    time.sleep(delay)
                    delay *= 2
        return wrapper

    @retry_with_delay
    def call_gemini_image_text(self, prompt, image, text=None, response_type="application/json", model=None):
        """이미지와 텍스트를 함께 처리하는 함수"""

        target_image = self.client.files.upload(file=image)
        response = self.client.models.generate_content(
            model=model if model else self.model,
            contents=[
                prompt,
                target_image,
                text,
            ],
            config={
                "response_mime_type": response_type,
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
        )
        return response.candidates[0].content.parts[0].text

    @retry_with_delay
    def call_gemini_text(self, prompt, response_type="application/json", model=None):
        """텍스트만 처리하는 함수"""
        response = self.client.models.generate_content(
            model=model if model else self.model,
            contents=[prompt],
            config={
                "response_mime_type": response_type,
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
        )
        return response.candidates[0].content.parts[0].text

    @retry_with_delay
    def call_extract_metadata(self, content, response_type="application/json", model=None):
        """이미지와 텍스트를 함께 처리하는 함수"""

        response = self.client.models.generate_content(
            model=model if model else self.model,
            contents=content,
            config={
                "response_mime_type": response_type,
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
        )
        return response.candidates[0].content.parts[0].text