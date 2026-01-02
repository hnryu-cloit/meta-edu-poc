"""
바운딩 박스 유틸리티
- 학생 풀이의 단계별 바운딩 박스 그룹화 및 색상 할당
- 이미지 위에 바운딩 박스 오버레이
"""

from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from common.logger import init_logger

logger = init_logger()

# 단계별 형광펜 색상 팔레트 (RGBA 형식, 투명도 80%)
STEP_COLORS = [
    {"name": "yellow", "rgb": "255, 255, 0", "rgba": "rgba(255, 255, 0, 0.2)", "border": "#FFD700"},     # 노란색
    {"name": "green", "rgb": "0, 255, 0", "rgba": "rgba(0, 255, 128, 0.2)", "border": "#00C853"},        # 초록색
    {"name": "orange", "rgb": "255, 165, 0", "rgba": "rgba(255, 165, 0, 0.2)", "border": "#FF6F00"},     # 주황색
    {"name": "purple", "rgb": "138, 43, 226", "rgba": "rgba(138, 43, 226, 0.2)", "border": "#7B1FA2"},   # 보라색
    {"name": "cyan", "rgb": "0, 255, 255", "rgba": "rgba(0, 255, 255, 0.2)", "border": "#00B8D4"},       # 하늘색
    {"name": "pink", "rgb": "255, 20, 147", "rgba": "rgba(255, 20, 147, 0.2)", "border": "#C51162"},     # 핑크색
    {"name": "lime", "rgb": "50, 205, 50", "rgba": "rgba(50, 205, 50, 0.2)", "border": "#64DD17"},       # 라임색
    {"name": "red", "rgb": "255, 0, 0", "rgba": "rgba(255, 0, 0, 0.2)", "border": "#D32F2F"},            # 빨간색
]


def get_step_color(step_number: int) -> Dict[str, str]:
    """
    단계 번호에 따른 색상 정보를 반환
    Args:
        step_number (int): 단계 번호 (1부터 시작)
    Returns:
        Dict[str, str]: 색상 정보 {"name": 색상명, "rgb": RGB 값, "rgba": RGBA 값, "border": 테두리 색}
    """
    # 단계 번호가 색상 팔레트 범위를 벗어나면 순환
    index = (step_number - 1) % len(STEP_COLORS)
    return STEP_COLORS[index]


def group_bboxes_by_step(
    ocr_bboxes: List[Dict[str, Any]],
    step_evaluations: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    OCR로 추출한 바운딩 박스를 단계별로 그룹화

    전략:
    1. 각 단계의 Y좌표 범위를 추정 (상단부터 순차적으로 분할)
    2. 바운딩 박스의 Y좌표를 기준으로 어느 단계에 속하는지 판단

    Args:
        ocr_bboxes (List[Dict]): OCR로 추출한 모든 바운딩 박스
            [{"text": "...", "bbox": {"x": ..., "y": ..., "width": ..., "height": ...}, ...}, ...]
        step_evaluations (List[Dict]): 단계별 평가 결과
            [{"step_number": 1, "step_name": "...", "status": "Correct/Incorrect", ...}, ...]

    Returns:
        Dict[str, List[Dict]]: 단계별로 그룹화된 바운딩 박스
            {"step_1": [...], "step_2": [...], ...}
    """
    if not ocr_bboxes or not step_evaluations:
        logger.warning("OCR 바운딩 박스 또는 단계 평가 결과가 없습니다.")
        return {}

    # 블록 레벨 바운딩 박스만 필터링
    block_bboxes = [bbox for bbox in ocr_bboxes if bbox.get("level") == "block"]

    if not block_bboxes:
        logger.warning("블록 레벨 바운딩 박스가 없습니다.")
        # 블록이 없으면 모든 바운딩 박스 사용
        block_bboxes = ocr_bboxes

    # Y좌표를 기준으로 바운딩 박스 정렬
    sorted_bboxes = sorted(block_bboxes, key=lambda b: b["bbox"]["y"])

    # 전체 Y좌표 범위 계산
    if sorted_bboxes:
        y_min = sorted_bboxes[0]["bbox"]["y"]
        y_max = max([b["bbox"]["y"] + b["bbox"]["height"] for b in sorted_bboxes])
    else:
        y_min, y_max = 0, 1000  # 기본값

    # 단계 수
    num_steps = len(step_evaluations)

    # Y좌표를 균등 분할하여 각 단계의 범위 할당
    step_height = (y_max - y_min) / num_steps
    grouped_bboxes = {}

    for step_idx, step_eval in enumerate(step_evaluations):
        step_number = step_eval.get("step_number", step_idx + 1)
        step_key = f"step_{step_number}"

        # 이 단계의 Y좌표 범위
        step_y_start = y_min + (step_idx * step_height)
        step_y_end = y_min + ((step_idx + 1) * step_height)

        # 이 범위에 속하는 바운딩 박스 필터링
        step_bboxes = []
        for bbox in sorted_bboxes:
            bbox_y_center = bbox["bbox"]["y"] + (bbox["bbox"]["height"] / 2)

            # Y좌표 중심이 이 단계 범위에 속하는지 확인
            if step_y_start <= bbox_y_center < step_y_end:
                # 색상 정보 추가
                bbox_with_color = bbox.copy()
                bbox_with_color["color"] = get_step_color(step_number)
                bbox_with_color["step_number"] = step_number
                bbox_with_color["step_name"] = step_eval.get("step_name", f"단계 {step_number}")
                bbox_with_color["step_status"] = step_eval.get("status", "Unknown")
                bbox_with_color["feedback"] = step_eval.get("feedback", "")
                step_bboxes.append(bbox_with_color)

        grouped_bboxes[step_key] = step_bboxes
        logger.info(f"{step_key}: {len(step_bboxes)}개 바운딩 박스 할당")

    return grouped_bboxes


def merge_adjacent_bboxes(
    bboxes: List[Dict[str, Any]],
    threshold_x: int = 50,
    threshold_y: int = 20
) -> List[Dict[str, Any]]:
    """
    인접한 바운딩 박스들을 병합합니다.

    Args:
        bboxes (List[Dict]): 바운딩 박스 리스트
        threshold_x (int): X축 방향 병합 임계값 (픽셀)
        threshold_y (int): Y축 방향 병합 임계값 (픽셀)

    Returns:
        List[Dict]: 병합된 바운딩 박스 리스트
    """
    if len(bboxes) <= 1:
        return bboxes

    merged = []
    used = set()

    for i, bbox1 in enumerate(bboxes):
        if i in used:
            continue

        # 현재 바운딩 박스
        current_bbox = bbox1["bbox"].copy()
        current_text = bbox1.get("text", "")
        merged_indices = {i}

        # 다른 바운딩 박스와 비교
        for j, bbox2 in enumerate(bboxes):
            if j <= i or j in used:
                continue

            # 두 바운딩 박스가 인접한지 확인
            x_distance = abs(bbox1["bbox"]["x"] - bbox2["bbox"]["x"])
            y_distance = abs(bbox1["bbox"]["y"] - bbox2["bbox"]["y"])

            if x_distance <= threshold_x and y_distance <= threshold_y:
                # 병합
                x_min = min(current_bbox["x"], bbox2["bbox"]["x"])
                y_min = min(current_bbox["y"], bbox2["bbox"]["y"])
                x_max = max(
                    current_bbox["x"] + current_bbox["width"],
                    bbox2["bbox"]["x"] + bbox2["bbox"]["width"]
                )
                y_max = max(
                    current_bbox["y"] + current_bbox["height"],
                    bbox2["bbox"]["y"] + bbox2["bbox"]["height"]
                )

                current_bbox = {
                    "x": x_min,
                    "y": y_min,
                    "width": x_max - x_min,
                    "height": y_max - y_min
                }
                current_text += " " + bbox2.get("text", "")
                merged_indices.add(j)

        # 병합된 바운딩 박스 추가
        merged_bbox = bbox1.copy()
        merged_bbox["bbox"] = current_bbox
        merged_bbox["text"] = current_text.strip()
        merged.append(merged_bbox)

        used.update(merged_indices)

    logger.info(f"바운딩 박스 병합: {len(bboxes)}개 → {len(merged)}개")
    return merged


def filter_incorrect_steps(
    grouped_bboxes: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    틀린 단계(Incorrect)의 바운딩 박스만 필터링
    Args:
        grouped_bboxes (Dict): 단계별로 그룹화된 바운딩 박스

    Returns:
        Dict: 틀린 단계의 바운딩 박스만 포함
    """
    filtered = {}

    for step_key, bboxes in grouped_bboxes.items():
        if not bboxes:
            continue

        # 첫 번째 바운딩 박스의 step_status 확인
        step_status = bboxes[0].get("step_status", "Unknown")

        # Incorrect 또는 Partial 단계만 포함
        if step_status in ["Incorrect", "Partial"]:
            filtered[step_key] = bboxes

    logger.info(f"틀린 단계 필터링: {len(grouped_bboxes)}개 → {len(filtered)}개")
    return filtered


def calculate_bbox_coverage(
    grouped_bboxes: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, float]:
    """
    각 단계의 바운딩 박스 커버리지(총 면적)를 계산

    Args:
        grouped_bboxes (Dict): 단계별로 그룹화된 바운딩 박스
    Returns:
        Dict[str, float]: 단계별 총 면적 {"step_1": 12345.0, ...}
    """
    coverage = {}

    for step_key, bboxes in grouped_bboxes.items():
        total_area = sum([
            bbox["bbox"]["width"] * bbox["bbox"]["height"]
            for bbox in bboxes
        ])
        coverage[step_key] = total_area

    return coverage

def draw_bboxes_on_image(
    image_path: str,
    bboxes: List[Dict[str, Any]],
    show_text: bool = False
) -> Image.Image:
    """
    이미지 위에 바운딩 박스 그리기

    Args:
        image_path (str): 원본 이미지 경로
        bboxes (List[Dict]): 바운딩 박스 리스트
            각 항목: {"bbox": {"x": ..., "y": ..., "width": ..., "height": ...},
                     "color": {"rgba": "rgba(255,255,0,0.2)", "border": "#FFD700"},
                     "text": "...", "feedback": "..."}
        show_text (bool): 바운딩 박스 내부에 텍스트 표시 여부

    Returns:
        Image.Image: 바운딩 박스가 그려진 이미지
    """
    # 원본 이미지 로드
    image = Image.open(image_path).convert("RGBA")

    # 오버레이 레이어 생성
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # 각 바운딩 박스 그리기
    for bbox_info in bboxes:
        bbox = bbox_info["bbox"]
        x = bbox["x"]
        y = bbox["y"]
        width = bbox["width"]
        height = bbox["height"]

        # 색상 정보 추출
        color_info = bbox_info.get("color", {})
        rgba_str = color_info.get("rgba", "rgba(255, 255, 0, 0.2)")
        border_color = color_info.get("border", "#FFD700")

        # RGBA 문자열 파싱 (예: "rgba(255, 255, 0, 0.2)" -> (255, 255, 0, 51))
        rgba_values = rgba_str.replace("rgba(", "").replace(")", "").split(",")
        r = int(rgba_values[0].strip())
        g = int(rgba_values[1].strip())
        b = int(rgba_values[2].strip())
        a = int(float(rgba_values[3].strip()) * 255)  # 0.2 -> 51

        # 채우기 (형광펜 효과)
        draw.rectangle(
            [(x, y), (x + width, y + height)],
            fill=(r, g, b, a),
            outline=None
        )

        # 선택적 텍스트 표시
        if show_text and bbox_info.get("text"):
            text = bbox_info["text"][:20]  # 최대 20자
            try:
                # 작은 폰트 사용
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            # 텍스트 배경
            text_bbox = draw.textbbox((x + 5, y + 5), text, font=font)
            draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
            draw.text((x + 5, y + 5), text, fill=(0, 0, 0, 255), font=font)

    # 원본 이미지와 오버레이 합성
    result = Image.alpha_composite(image, overlay)

    return result.convert("RGB")


def image_to_base64(image: Image.Image) -> str:
    """
    PIL Image를 Base64 문자열 변환

    Args:
        image (Image.Image): PIL Image 객체

    Returns:
        str: Base64 인코딩된 이미지 문자열
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def create_bbox_overlay_html(
    image_path: str,
    bboxes: List[Dict[str, Any]],
    width: int = 600
) -> str:
    """
    바운딩 박스가 오버레이된 이미지의 HTML을 생성
    (Streamlit의 st.markdown으로 렌더링 가능)

    Args:
        image_path (str): 원본 이미지 경로
        bboxes (List[Dict]): 바운딩 박스 리스트
        width (int): 이미지 너비 (픽셀)

    Returns:
        str: HTML 문자열
    """
    # 바운딩 박스가 그려진 이미지 생성
    result_image = draw_bboxes_on_image(image_path, bboxes, show_text=False)

    # Base64로 변환
    img_base64 = image_to_base64(result_image)

    # HTML 생성
    html = f'''
    <div style="text-align: center;">
        <img src="data:image/png;base64,{img_base64}" width="{width}" />
    </div>
    '''

    return html

def create_interactive_bbox_overlay(
    image_path: str,
    grouped_bboxes: Dict[str, List[Dict[str, Any]]],
    selected_steps: List[str],
    width: int = 600
) -> str:
    """
    선택된 단계의 바운딩 박스만 표시하는 인터랙티브 오버레이를 생성합니다.

    Args:
        image_path (str): 원본 이미지 경로
        grouped_bboxes (Dict): 단계별로 그룹화된 바운딩 박스
        selected_steps (List[str]): 표시할 단계 리스트 (예: ["step_1", "step_2"])
        width (int): 이미지 너비

    Returns:
        str: HTML 문자열
    """
    # 선택된 단계의 바운딩 박스만 필터링
    filtered_bboxes = []
    for step_key in selected_steps:
        if step_key in grouped_bboxes:
            filtered_bboxes.extend(grouped_bboxes[step_key])

    # 바운딩 박스가 없으면 원본 이미지만 표시
    if not filtered_bboxes:
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode()
        html = f'''
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="{width}" />
        </div>
        '''
        return html

    # 바운딩 박스 오버레이
    return create_bbox_overlay_html(image_path, filtered_bboxes, width)


# 테스트 함수
def test_bbox_utils():
    """
    바운딩 박스 유틸리티 테스트
    """
    # 더미 OCR 결과
    ocr_bboxes = [
        {"text": "문제1", "bbox": {"x": 100, "y": 50, "width": 200, "height": 30}, "level": "block"},
        {"text": "x + 2 = 5", "bbox": {"x": 100, "y": 100, "width": 150, "height": 40}, "level": "block"},
        {"text": "x = 3", "bbox": {"x": 100, "y": 200, "width": 100, "height": 30}, "level": "block"},
        {"text": "답: 3", "bbox": {"x": 100, "y": 300, "width": 80, "height": 25}, "level": "block"},
    ]

    # 더미 단계 평가
    step_evaluations = [
        {"step_number": 1, "step_name": "문제 이해", "status": "Correct"},
        {"step_number": 2, "step_name": "방정식 풀이", "status": "Incorrect", "feedback": "계산 오류"},
        {"step_number": 3, "step_name": "답 작성", "status": "Correct"},
    ]

    # 그룹화 테스트
    grouped = group_bboxes_by_step(ocr_bboxes, step_evaluations)

    print("\n=== 단계별 바운딩 박스 그룹화 ===")
    for step_key, bboxes in grouped.items():
        print(f"\n{step_key}: {len(bboxes)}개")
        for bbox in bboxes:
            print(f"  - {bbox['text']} (색상: {bbox['color']['name']})")

    # 틀린 단계만 필터링
    incorrect_only = filter_incorrect_steps(grouped)

    print("\n=== 틀린 단계만 필터링 ===")
    for step_key, bboxes in incorrect_only.items():
        print(f"\n{step_key}: {len(bboxes)}개")
        for bbox in bboxes:
            print(f"  - {bbox['text']}: {bbox['feedback']}")

    # 오버레이 테스트
    print("\n=== 오버레이 생성 테스트 ===")
    test_overlay_bboxes = [
        {
            "bbox": {"x": 100, "y": 50, "width": 200, "height": 30},
            "color": {"rgba": "rgba(255, 255, 0, 0.2)", "border": "#FFD700"},
            "text": "테스트 텍스트",
            "feedback": "피드백 예시"
        }
    ]
    image_path = "resource/solve/223174-1.png"
    if os.path.exists(image_path):
        result = draw_bboxes_on_image(image_path, test_overlay_bboxes)
        result.save("test_overlay.png")
        print("오버레이 이미지 생성 완료: test_overlay.png")

        # 인터랙티브 오버레이 테스트
        html_output = create_interactive_bbox_overlay(
            image_path,
            grouped,
            ["step_2"]
        )
        with open("test_interactive_overlay.html", "w", encoding="utf-8") as f:
            f.write(html_output)
        print("인터랙티브 오버레이 HTML 생성 완료: test_interactive_overlay.html")

    else:
        print(f"테스트 이미지를 찾을 수 없습니다: {image_path}")


if __name__ == "__main__":
    test_bbox_utils()