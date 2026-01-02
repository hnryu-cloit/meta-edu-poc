"""
OCR 서비스 모듈

다양한 OCR 엔진을 선택하여 사용할 수 있는 인터페이스를 제공합니다.
현재 지원하는 엔진:
- LaTeX-OCR (pix2tex)
- Nougat-LaTeX
- Google Cloud Vision (기존)
"""
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from pix2tex.cli import LatexOCR as Pix2TexOCR

from common.logger import init_logger

# Google Vision 관련 import는 필요할 때만 로드되도록 클래스 내부로 이동
# from google.cloud import vision

logger = init_logger()


class OCREngine(ABC):
    """OCR 엔진을 위한 추상 기반 클래스"""

    @abstractmethod
    def extract_latex(self, image_path: str) -> List[Dict[str, Any]]:
        """
        이미지에서 LaTeX 수식과 바운딩 박스를 추출합니다.

        Args:
            image_path (str): 분석할 이미지 파일 경로

        Returns:
            List[Dict[str, Any]]: 추출된 LaTeX 정보 리스트.
                각 항목은 다음 형식:
                {
                    "text": "추출된 LaTeX 수식",
                    "bbox": [x_min, y_min, x_max, y_max]  # 좌상단, 우하단 좌표
                }
        """
        pass


class LatexOCREngine(OCREngine):
    """pix2tex (LaTeX-OCR) 라이브러리를 사용하는 OCR 엔진"""

    def __init__(self):
        try:
            # pix2tex 모델 초기화
            self.model = Pix2TexOCR()
            logger.info("LaTeX-OCR (pix2tex) 엔진 초기화 완료")
        except Exception as e:
            logger.error(f"LaTeX-OCR 엔진 초기화 실패: {e}")
            raise

    def extract_latex(self, image_path: str) -> List[Dict[str, Any]]:
        """
        이미지 전체에서 단일 LaTeX 수식을 추출합니다.
        pix2tex는 이미지 내 개별 수식의 바운딩 박스를 제공하지 않으므로,
        이미지 전체를 하나의 바운딩 박스로 간주합니다.
        """
        try:
            img = Image.open(image_path)
            # 모델을 사용하여 LaTeX 추출
            latex_text = self.model(img)
            logger.info(f"LaTeX-OCR 분석 완료: {image_path}")

            if latex_text:
                # 이미지 전체 크기를 바운딩 박스로 사용
                width, height = img.size
                return [{
                    "text": latex_text,
                    "bbox": [0, 0, width, height]
                }]
            else:
                return []
        except Exception as e:
            logger.error(f"LaTeX-OCR 처리 중 오류 발생: {e}", exc_info=True)
            return []


class NougatLatexOCREngine(OCREngine):
    """nougat-latex (Nougat) 라이브러리를 사용하는 OCR 엔진"""

    def __init__(self):
        try:
            from transformers import VisionEncoderDecoderModel, AutoTokenizer

            # Nougat 모델과 토크나이저 초기화
            model_name = "facebook/nougat-small"

            # 토크나이저를 먼저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Tokenizer 로드 완료: {model_name}")

            # VisionEncoderDecoderModel로 직접 로드
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

            # 디바이스 설정
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Nougat-LaTeX 엔진 초기화 완료 (device: {self.device})")
        except Exception as e:
            logger.error(f"Nougat-LaTeX 엔진 초기화 실패: {e}")
            raise

    def extract_latex(self, image_path: str) -> List[Dict[str, Any]]:
        """
        이미지에서 LaTeX 수식과 위치 정보를 추출합니다.
        Nougat는 텍스트 라인별로 결과를 반환하며, LaTeX 수식 부분을 파싱해야 합니다.
        """
        try:
            from transformers import NougatImageProcessor

            # 이미지 로드
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # 이미지 전처리 (Nougat용 processor 사용)
            processor = NougatImageProcessor.from_pretrained("facebook/nougat-small")
            pixel_values = processor(img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # 모델 추론
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    max_new_tokens=896,
                    num_beams=1,
                    do_sample=False,
                )

            # 디코딩
            sequence = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # 기본 후처리: 특수 토큰 제거
            sequence = sequence.strip()

            logger.info(f"Nougat-LaTeX 분석 완료: {image_path}")

            if sequence:
                width, height = img.size
                return [{
                    "text": sequence,
                    "bbox": [0, 0, width, height]
                }]
            else:
                return []

        except Exception as e:
            logger.error(f"Nougat-LaTeX 처리 중 오류 발생: {e}", exc_info=True)
            return []


def get_ocr_engine(engine_name: str) -> Optional[OCREngine]:
    """
    지정된 이름의 OCR 엔진 인스턴스를 생성하여 반환하는 팩토리 함수

    Args:
        engine_name (str): 사용할 엔진의 이름 ('latex_ocr' 또는 'nougat_latex')

    Returns:
        Optional[OCREngine]: 생성된 OCR 엔진 인스턴스. 지원하지 않는 이름일 경우 None.
    """
    if engine_name == 'latex_ocr':
        logger.info("LaTeX-OCR 엔진을 생성합니다.")
        return LatexOCREngine()
    elif engine_name == 'nougat_latex':
        logger.info("Nougat-LaTeX 엔진을 생성합니다.")
        return NougatLatexOCREngine()
    # elif engine_name == 'google':
    #     logger.info("Google Vision OCR 엔진을 생성합니다.")
    #     return GoogleVisionOCREngine()
    else:
        logger.error(f"지원하지 않는 OCR 엔진입니다: {engine_name}")
        return None

# --- 기존 Google Vision OCR 코드 (참고 및 향후 사용을 위해 유지) ---

class GoogleVisionOCREngine(OCREngine):
    """Google Cloud Vision API를 사용하는 OCR 서비스 (리팩토링된 버전)"""

    def __init__(self):
        try:
            # google-cloud-vision이 설치되어 있는지 확인
            from google.cloud import vision
            self.project_id = os.getenv('PROJECT_ID')
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision API 초기화 완료")
        except ImportError:
            logger.error("Google Cloud Vision 라이브러리가 설치되지 않았습니다. `pip install google-cloud-vision`")
            raise
        except Exception as e:
            logger.error(f"Vision API 초기화 실패: {e}")
            raise

    def extract_latex(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Google Vision API는 일반 텍스트를 추출하므로, 이 클래스는 LaTeX 변환을 수행하지 않습니다.
        대신 텍스트 블록과 바운딩 박스를 반환합니다.
        """
        logger.warning("GoogleVisionOCREngine은 LaTeX가 아닌 일반 텍스트를 추출합니다.")
        # `extract_text_with_bboxes` 메소드를 호출하고 반환 형식을 맞춤
        results = self.extract_text_with_bboxes(image_path, levels=['block'])
        
        # OCREngine의 반환 형식에 맞게 변환
        formatted_results = []
        for res in results:
            bbox = res.get('bbox', {})
            formatted_results.append({
                "text": res.get('text', ''),
                "bbox": [bbox.get('x', 0), bbox.get('y', 0), bbox.get('x', 0) + bbox.get('width', 0), bbox.get('y', 0) + bbox.get('height', 0)]
            })
        return formatted_results


    def extract_text_with_bboxes(self, image_path: str, levels: List[str] = ['word', 'paragraph', 'block']) -> List[Dict[str, Any]]:
        """이미지에서 지정된 레벨의 텍스트와 바운딩 박스를 추출"""
        from google.cloud import vision
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            image = vision.Image(content=content)
            response = self.client.document_text_detection(image=image)

            if response.error.message:
                raise Exception(f"Vision API 오류: {response.error.message}")

            annotations = []
            if response.full_text_annotation:
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        if 'block' in levels:
                            annotations.append(self._parse_annotation(block, 'block'))
                        for paragraph in block.paragraphs:
                            if 'paragraph' in levels:
                                annotations.append(self._parse_annotation(paragraph, 'paragraph'))
                            if 'word' in levels:
                                for word in paragraph.words:
                                    annotations.append(self._parse_annotation(word, 'word'))
            
            logger.info(f"Google Vision OCR 완료: {len(annotations)}개 텍스트 영역 추출 ({image_path})")
            return annotations
        except FileNotFoundError:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            return []
        except Exception as e:
            logger.error(f"Google Vision OCR 처리 중 오류 발생: {e}", exc_info=True)
            return []

    def _parse_annotation(self, entity, level: str) -> Dict[str, Any]:
        """Vision API의 응답 엔티티를 공통 형식으로 파싱"""
        text = ''.join([symbol.text for symbol in entity.symbols]) if level == 'word' else entity.text
        
        vertices = entity.bounding_box.vertices
        x_coords = [v.x for v in vertices]
        y_coords = [v.y for v in vertices]
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)

        return {
            "text": text,
            "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
            "vertices": [[v.x, v.y] for v in vertices],
            "confidence": entity.confidence,
            "level": level
        }