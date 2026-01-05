
import os
import random
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from common.logger import init_logger

logger = init_logger()

def get_random_color() -> Tuple[int, int, int]:
    """랜덤 색상 생성 (RGB)"""
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

def visualize_bboxes_with_ids(
    image_path: str, 
    ocr_results: List[Dict[str, Any]], 
    output_image_path: str
) -> List[Dict[str, Any]]:
    """
    OCR 결과(바운딩 박스)를 이미지에 시각화하고 고유 ID를 할당합니다.
    
    Args:
        image_path (str): 원본 이미지 경로
        ocr_results (List[Dict]): OCR 엔진에서 추출한 결과 리스트.
        output_image_path (str): 시각화된 이미지를 저장할 경로

    Returns:
        List[Dict]: ID 매핑 정보 리스트 [{"id": 0, "text": "...", "box": [x,y,w,h]}, ...]
    """
    if not os.path.exists(image_path):
        logger.error(f"이미지를 찾을 수 없습니다: {image_path}")
        return []

    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        logger.error(f"이미지 로드 실패 {image_path}: {e}")
        return []

    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 폰트 설정
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    id_map = []
    
    for idx, item in enumerate(ocr_results):
        bbox_data = item.get("bbox")
        vertices = item.get("vertices")
        text = item.get("text", "")
        
        # 색상 생성
        color = get_random_color()
        fill_color = color + (60,)   
        outline_color = color + (255,) 
        
        # 박스 그리기 (Vertices가 있으면 Polygon으로, 없으면 Rectangle로)
        if vertices and len(vertices) >= 3:
            # 다각형 그리기 [{"x":x1,"y":y1}, ...] -> [(x1,y1), (x2,y2), ...]
            poly_points = [(v.get('x', 0), v.get('y', 0)) for v in vertices]
            draw.polygon(poly_points, fill=fill_color, outline=outline_color, width=3)
            # 바운딩 박스 계산 (다각형의 최소/최대 좌표)
            x_coords = [v.get('x', 0) for v in vertices]
            y_coords = [v.get('y', 0) for v in vertices]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
        else:
            # Bbox 정규화
            if isinstance(bbox_data, dict):
                x = bbox_data.get("x", 0)
                y = bbox_data.get("y", 0)
                w = bbox_data.get("width", 0)
                h = bbox_data.get("height", 0)
                box = [x, y, x + w, y + h]
            elif isinstance(bbox_data, list):
                if len(bbox_data) == 4:
                    box = bbox_data
                else:
                    continue
            else:
                continue
                
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color, outline=outline_color, width=3)
        
        # ID 라벨 그리기
        label_text = str(idx)
        
        if hasattr(draw, "textbbox"):
            text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)
            
        draw.rectangle([x_min, y_min - text_h - 5, x_min + text_w + 10, y_min], fill=outline_color)
        draw.text((x_min + 5, y_min - text_h - 5), label_text, fill=(255, 255, 255, 255), font=font)
        
        id_map.append({
            "id": idx,
            "text": text,
            "box": [x_min, y_min, x_max, y_max]
        })

    result = Image.alpha_composite(image, overlay)
    result = result.convert("RGB")
    result.save(output_image_path)
    
    logger.info(f"시각화 이미지 저장 완료: {output_image_path}")
    return id_map
