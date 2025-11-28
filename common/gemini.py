import json
import requests
import os
import time
import base64
import mimetypes
import traceback

from io import BytesIO

from google import genai
from google.genai import types
from dotenv import load_dotenv

import vertexai
from vertexai.preview.vision_models import (
  Image,
  ImageGenerationModel,
)

from PIL import Image as PIL_Image


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

        self.model = "gemini-2.0-flash" #'gemini-2.5-flash-preview-05-20' | 'gemini-2.5-pro-preview-06-05'
        self.model_nb = "gemini-2.5-flash-image-preview"

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
    def call_gemini_image_stream(self, prompt, image):
        target_image = self.client.files.upload(file=image, mime_type="image/jpeg")
        model = "gemini-2.5-flash-image-preview"
        contents = [
            types.Content(
                role="user",
                parts=[
                    target_image,
                    prompt
                ]
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=0,
            top_k= 1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT", "IMAGE"],
            safety_settings=[types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_HATE",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_HARASSMENT",
                threshold="OFF"
            ), types.SafetySetting(
                category="HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT",
                threshold="OFF"
            )],
        )
        for chunk in self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
        ):
            print(chunk.text, end="")

    @retry_with_delay
    def call_gemini_text_stream(self, prompt, model=None):
        """스트리밍 텍스트 생성 함수"""
        response = self.client.models.generate_content_stream(
            model=model if model else self.model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=prompt)
                    ]
                ),
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=1,
                top_k=1,
                max_output_tokens=8192,
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="OFF"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="OFF"
                    )
                ],
            )
        )
        return response

    @retry_with_delay
    def call_gemini_multimodal(self, contents, model=None):
        """멀티모달 콘텐츠 처리 함수"""
        response = self.client.models.generate_content(
            model=model if model else self.model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
        )
        return response.text

    @retry_with_delay
    def extract_image_component(self, img_bytes, model):
        """이미지의 요소들을 추출하는 함수"""
        prompt = """
            입력받은 이미지를 보고 이미지의 특징, 구성요소들을 상세하게 설명해주세요.
        """
        contents = [prompt, types.Part.from_bytes(data=img_bytes, mime_type="image/png")]
        response = self.call_gemini_multimodal(contents, model)
        return response

    @retry_with_delay
    def call_vertexai_imagen3_text_image(self, prompt, negative_prompt=None, reference_images_path=None):

        model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")

        reference_images = []
        if reference_images_path:
            for i, path in enumerate(reference_images_path):
                image_bytes = load_image_bytes(path)
                if image_bytes:
                    image_component = self.extract_image_component(image_bytes, model=self.model)
                    reference_images.append(
                        SubjectReferenceImage(
                            reference_id=i + 1,
                            image=Image(image_bytes=image_bytes),
                            subject_description=image_component,
                            subject_type="SUBJECT_TYPE_DEFAULT",
                        )
                    )

        response = model._generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            negative_prompt=negative_prompt,
            person_generation="allow_all",
            safety_filter_level="block_few",
            add_watermark=False,
            reference_images=reference_images,
            seed=1004
        )
        
        image = PIL_Image.open(BytesIO(response[0]._image_bytes))
        return image

    @retry_with_delay
    def call_vertexai_imagen4_text(self, prompt, negative_prompt=""):
        """VertexAI Imagen4를 사용한 이미지 생성"""

        model = ImageGenerationModel.from_pretrained("imagen-4.0-generate-preview-06-06")
        
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            negative_prompt=negative_prompt,
            person_generation="allow_all",
            safety_filter_level="block_few",
            add_watermark=False,
            seed=1004
        )
        
        image = PIL_Image.open(BytesIO(response[0]._image_bytes))
        return image

    @retry_with_delay
    def call_veo_text_video(self, prompt, model="veo-3.0-generate-preview", output_path=None, image=None):
        """VEO를 사용한 비디오 생성"""
        video_config = types.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            negative_prompt="text, title, logo",
            seed=1004
        )

        try:
            self.logger.info("동영상 생성을 시작합니다...")
            operation = self.client.models.generate_videos(
                model=model,
                prompt=prompt,
                config=video_config,
                image=image,
            )

            # 영상 생성 완료까지 대기
            while not operation.done:
                time.sleep(5)
                operation = self.client.operations.get(operation)

            # 생성된 영상 로컬에 저장
            if output_path and operation.response.generated_videos:
                for n, generated_video in enumerate(operation.response.generated_videos):
                    download_url = generated_video.video.uri
                    headers = {'x-goog-api-key': self.api_key}

                    response = requests.get(download_url, headers=headers)
                    response.raise_for_status()

                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    self.logger.info(f"영상 저장 성공: {output_path}")

        except Exception as e:
            self.logger.error(f"비디오 생성 오류: {e}")
            traceback.print_exc()
            return None

        return operation

    def call_image_generator(self, prompt, image_files):

        # 입력 파일이 존재 하는지 확인 합니다.
        if not image_files:
            print(f"오류: 입력 이미지 파일을 찾을 수 없습니다.")
            return [], ""

        parts = [types.Part.from_text(text=prompt)]

        for image_file in image_files:
            if not os.path.exists(image_file):
                print(f"오류: 입력 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요: {image_file}")
                continue

            # 입력 이미지 로드
            try:
                with open(image_file, "rb") as f:
                    image_data = f.read()
                mime_type, _ = mimetypes.guess_type(image_file)
                if not mime_type:
                    mime_type = 'application/octet-stream'  # 기본값 설정
                parts.append(types.Part.from_bytes(data=image_data, mime_type=mime_type))
            except Exception as e:
                self.logger.error(f"이미지 파일 로딩 중 오류 발생: {e}")
                return [], ""

        # 모델에게 전달할 콘텐츠 프롬프트 구성
        contents = [
            types.Content(
                role="user",
                parts=parts,
            ),
        ]

        generate_config = types.GenerateContentConfig(
            response_modalities=["IMAGE","TEXT"],
            temperature=0,
            top_p=1,
            top_k=1,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ]
        )

        image_parts = []
        full_text_response = ""

        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_nb,
                contents=contents,
                config=generate_config,
            )
            for chunk in response_stream:
                if not (chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts):
                    continue

                for part in chunk.candidates[0].content.parts:
                    # 이미지 데이터가 있다면 리스트에 추가합니다.
                    if part.inline_data:
                        image_parts.append(part.inline_data)
                    # 텍스트 데이터가 있다면 full_text_response에 추가합니다.
                    elif part.text:
                        full_text_response += part.text

        except Exception as e:
            self.logger.error(f"Gemini 스트리밍 중 오류 발생: {e}")
            return [], ""

        return image_parts, full_text_response

class GeminiTranslation(Gemini):

    def __init__(self):
        super().__init__()
        self.prompt = TRANSLATION()

    def generate_image_response(self, query: str, image_data, model:str):
        """이미지가 포함된 요청을 처리하는 함수"""
        response = self.client.models.generate_content(
            model=model if model else self.model,
            contents=[
                query,
                {
                    "inline_data": image_data
                }
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                top_p=1,
                top_k=1,
            )
        )

        result = json.loads(response.text)
        return result

    def process_translation(self, korean_texts: list, src_lang: str, target_lang: str):
        results = []
        for text in korean_texts:
            print(f"  Processing {text} ...")
            try:

                prompt_text = self.prompt.to_language(text, src_lang, target_lang)
                response = self.call_gemini_text(prompt_text)

                print(f"    response: {response}")

                if isinstance(response, dict):
                    for key in ["translation", "request", "response", "jp"]:
                        if key in response and response[key]:
                            result_text = response[key]
                            break
                    else:
                        result_text = str(response)
                elif isinstance(response, list):
                    result_text = response[0] if response else "번역 실패"
                else:
                    result_text = str(response)

                results.append(result_text)
                print(f"    result: {result_text}")

            except Exception as e:
                error_message = f"    처리 실패: {text} -> {e}"
                print(error_message)
                print(f"    요청 내용: {prompt_text}")
                results.append(f"번역 실패: {str(e)}")
        return results

    # 나노바나나
    def call_gemini_image_nb(self, prompt, image_file):
        """ 나노바나나: 텍스트 프롬프트를 통해 이미지를 생성하고, 추출된 텍스트를 JSON으로 저장 """

        # 나노바나나 인페인팅을 위한 모델 지정 (flash-2.5)
        model = self.model_nb

        # 입력 파일이 존재하는지 확인합니다.
        if not os.path.exists(image_file):
            print(f"오류: 입력 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요: {image_file}")
            return

        # 입력 이미지 로드
        try:
            with open(image_file, "rb") as f:
                image_data = f.read()
            mime_type, _ = mimetypes.guess_type(image_file)
            if not mime_type:
                mime_type = 'application/octet-stream'  # 기본값 설정
        except Exception as e:
            self.logger.error(f"이미지 파일 로딩 중 오류 발생: {e}")
            return [], ""

        # 모델에게 전달할 콘텐츠 (프롬프트)를 구성
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]

        # 모델의 응답 형식 지정. 이미지와 텍스트 모두 받을 수 있도록 설정
        generate_config = types.GenerateContentConfig(
            response_modalities=["IMAGE","TEXT"],
            temperature=0,
            top_p=1,
            top_k=1,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ]
        )

        image_parts = []  # 스트리밍으로 수신한 이미지 데이터를 저장할 리스트
        full_text_response = ""  # 스트리밍으로 들어오는 텍스트 데이터를 합칠 변수

        self.logger.info("Gemini 모델로부터 스트리밍 응답을 시작합니다...")
        try:
            response_stream = self.client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_config,
            )
            for chunk in response_stream:
                if not (chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts):
                    continue

                for part in chunk.candidates[0].content.parts:
                    # 이미지 데이터가 있다면 리스트에 추가합니다.
                    if part.inline_data:
                        image_parts.append(part.inline_data)
                    # 텍스트 데이터가 있다면 full_text_response에 추가합니다.
                    elif part.text:
                        full_text_response += part.text

            # 진단을 위해 수신된 데이터의 수를 로그로 남깁니다.
            self.logger.info("스트리밍 응답 완료.")
            self.logger.info(f"수신된 이미지 파트 개수: {len(image_parts)}")
            self.logger.info(f"수신된 텍스트 길이: {len(full_text_response)}")

            # API가 이미지를 반환하지 않은 경우, 수신된 텍스트 응답을 경고로 출력합니다.
            if not image_parts and full_text_response:
                self.logger.warning("API가 이미지를 반환하지 않았습니다. 다음은 API의 텍스트 응답입니다:")
                self.logger.warning(f"API 응답: {full_text_response.strip()}")

            self.logger.info("스트리밍 응답 완료.")

        except Exception as e:
            self.logger.error(f"Gemini 스트리밍 중 오류 발생: {e}")
            # traceback.print_exc() # 디버깅 시 상세 오류를 보려면 주석 해제
            return [], ""

        return image_parts, full_text_response
