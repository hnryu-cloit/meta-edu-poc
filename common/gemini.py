import os
import time
import base64
import csv
from datetime import datetime
from pathlib import Path
import threading

from google import genai
from dotenv import load_dotenv

import vertexai
from common.logger import timefn

# CSV 로깅을 위한 전역 변수
BILLING_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/billing.csv")
csv_lock = threading.Lock()

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

def log_gemini_call(function_name: str, model_name: str, prompt: str, response: str, status: str, api_call_count: int = 1):
    """
    Gemini API 호출 내역을 CSV 파일에 로깅하는 함수

    Args:
        function_name: 호출한 함수명
        model_name: 사용한 Gemini 모델명
        prompt: 입력 프롬프트 (긴 경우 요약)
        response: 응답 결과 (긴 경우 요약)
        status: 성공/실패 상태
        api_call_count: API 호출 횟수
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # CSV 파일 경로 생성
    csv_path = Path(BILLING_CSV_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_lock:
        try:
            with open(csv_path, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)

                if not file_exists:
                    writer.writerow(['function', 'model_name', 'timestamp', 'prompt', 'response', 'status', 'api_call_count'])

                # 로그 데이터 작성
                writer.writerow([
                    function_name,
                    model_name,
                    timestamp,
                    prompt,
                    response,
                    status,
                    api_call_count
                ])
        except Exception as e:
            print(f"CSV 로깅 중 오류 발생: {e}")


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

    def retry_with_delay(max_retries=None):
        """재시도 데코레이터 - 함수별로 다른 retry 횟수 지원"""
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                retries = max_retries if max_retries is not None else self.max_retries
                delay = self.initial_delay
                api_call_count = 0

                for attempt in range(retries):
                    try:
                        api_call_count = attempt + 1
                        result = func(self, *args, api_call_count=api_call_count, **kwargs)
                        return result
                    except Exception as e:
                        if attempt == retries - 1:
                            raise e
                        self.logger.error(f"gemini 호출 {attempt + 1}번째 실패: {e}")
                        time.sleep(delay)
                        delay *= 2
            return wrapper
        return decorator

    @retry_with_delay(max_retries=3)
    @timefn
    def call_gemini_image_text(self, prompt, image, text=None, response_type="application/json", model=None, api_call_count=1):
        """이미지와 텍스트를 함께 처리하는 함수"""
        used_model = model if model else self.model
        response_text = None
        status = "실패"

        try:
            target_image = self.client.files.upload(file=image)
            response = self.client.models.generate_content(
                model=used_model,
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
            response_text = response.candidates[0].content.parts[0].text
            status = "성공"
            return response_text
        except Exception as e:
            response_text = f"오류: {str(e)}"
            raise
        finally:
            # 로깅
            prompt_str = f"{str(prompt)[:200]}... [이미지 포함]"
            log_gemini_call(
                function_name="call_gemini_image_text",
                model_name=used_model,
                prompt=prompt_str,
                response=response_text if response_text else "응답 없음",
                status=status,
                api_call_count=api_call_count
            )

    @retry_with_delay(max_retries=5)
    @timefn
    def call_gemini_text(self, prompt, response_type="application/json", model=None, api_call_count=1):
        """텍스트만 처리하는 함수"""
        used_model = model if model else self.model
        response_text = None
        status = "실패"

        try:
            response = self.client.models.generate_content(
                model=used_model,
                contents=[prompt],
                config={
                    "response_mime_type": response_type,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                }
            )
            response_text = response.candidates[0].content.parts[0].text
            status = "성공"
            return response_text
        except Exception as e:
            response_text = f"오류: {str(e)}"
            raise
        finally:
            # 로깅
            log_gemini_call(
                function_name="call_gemini_text",
                model_name=used_model,
                prompt=str(prompt),
                response=response_text if response_text else "응답 없음",
                status=status,
                api_call_count=api_call_count
            )

    @retry_with_delay(max_retries=4)
    @timefn
    def call_extract_metadata(self, content, response_type="application/json", model=None, api_call_count=1):
        """이미지와 텍스트를 함께 처리하는 함수"""
        used_model = model if model else self.model
        response_text = None
        status = "실패"

        try:
            response = self.client.models.generate_content(
                model=used_model,
                contents=content,
                config={
                    "response_mime_type": response_type,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                }
            )
            response_text = response.candidates[0].content.parts[0].text
            status = "성공"
            return response_text
        except Exception as e:
            response_text = f"오류: {str(e)}"
            raise
        finally:
            # 로깅
            content_str = f"{str(content)[:200]}... [content 포함]"
            log_gemini_call(
                function_name="call_extract_metadata",
                model_name=used_model,
                prompt=content_str,
                response=response_text if response_text else "응답 없음",
                status=status,
                api_call_count=api_call_count
            )