"""
공통 유틸리티 함수 모음
"""
import re


def fix_json_escaping(text):
    """
    Gemini 응답에서 잘못된 백슬래시 이스케이프를 수정합니다.
    LaTeX 수식 등에 포함된 백슬래시를 올바르게 이스케이프합니다.

    Args:
        text (str): Gemini의 JSON 응답 텍스트

    Returns:
        str: 수정된 JSON 텍스트
    """
    # JSON 문자열 값 내부의 모든 백슬래시를 올바르게 이스케이프
    def escape_backslashes_in_string(match):
        string_content = match.group(0)
        quote = string_content[0]
        content = string_content[1:-1]

        # 방법: 모든 백슬래시를 임시로 플레이스홀더로 교체 -> 이중 백슬래시로 변환 -> 유효한 이스케이프는 원래대로
        # 더 간단한 방법: 백슬래시를 순차적으로 처리

        # 1단계: 모든 \를 임시 플레이스홀더로 변환
        temp_placeholder = "###BACKSLASH###"
        content = content.replace("\\", temp_placeholder)

        # 2단계: 플레이스홀더를 \\로 변환 (JSON에서 \를 표현)
        content = content.replace(temp_placeholder, "\\\\")

        return f'{quote}{content}{quote}'

    # JSON 문자열 값을 찾는 정규식
    # 이중 따옴표 문자열만 처리 (JSON 표준)
    fixed_text = re.sub(
        r'"(?:[^"\\]|\\.)*"',
        escape_backslashes_in_string,
        text
    )

    return fixed_text