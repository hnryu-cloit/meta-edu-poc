import json
import pandas as pd


def create_curriculum_selection_prompt():
    """
    1-1단계: 문제 이미지를 분석하여 "./resource/curriculum.csv"에서 적절한 교육과정 선택

    Returns:
        str: curriculum 선택용 프롬프트
    """
    # CSV 파일 로드
    curriculum_df = pd.read_csv('resource/curriculum.csv', encoding='utf-8-sig')
    curriculum_list = curriculum_df.to_dict('records')

    # 교육과정 정보를 문자열로 변환
    curriculum_str = "\n".join([
        f"- 학년: {row['학년']}, 교과목: {row['교과목']}, 대단원: {row['대단원']}, 중단원: {row['중단원']}, 소단원: {row['소단원']}"
        for row in curriculum_list
    ])

    prompt = f"""
        당신은 수학 교육과정 전문가입니다.
        주어진 수학 문제 이미지를 분석하여, 아래 교육과정 목록에서 가장 적합한 항목을 선택해야 합니다.
        
        **사용 가능한 교육과정 목록:**
        {curriculum_str}
        
        **과제:**
        제공된 수학 문제 이미지를 분석하고, 위의 목록에서 이 문제에 가장 적합한 교육과정 정보를 선택하세요.
        
        **중요 지침:**
        1. 반드시 위에 제공된 목록에 있는 값만 사용하세요.
        2. 문제의 내용, 난이도, 사용된 수학 개념을 고려하여 선택하세요.
        3. 대단원, 중단원, 소단원은 위의 교육과정 목록에서 정확히 일치하는 항목을 찾으세요.
        
        **출력 형식:**
        반드시 다음 JSON 형식으로만 출력하세요:
        
        {{
          "학년": "선택한 학년",
          "교과목": "선택한 교과목",
          "대단원": "선택한 대단원",
          "중단원": "선택한 중단원",
          "소단원": "선택한 소단원",
          "선택_이유": "이 교육과정을 선택한 이유 (1-2문장)"
        }}
        
        **주의사항:**
        - 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
        - 모든 필드는 위의 목록에서 정확히 복사하여 사용하세요.
        """
    return prompt


def create_achievement_selection_prompt(curriculum_info):
    """
    1-2단계: 선택된 curriculum 정보를 바탕으로 "./resource/achievement_standards.csv"에서 성취기준 선택

    Args:
        curriculum_info (dict): 1-1단계에서 선택한 curriculum 정보

    Returns:
        str: achievement_standards 선택용 프롬프트
    """

    achievement_df = pd.read_csv('resource/achievement_standards.csv', encoding='utf-8-sig')

    # 학년과 교과목 기반으로 필터링
    grade = curriculum_info.get('학년', '')

    # 학년별로 필터링 (예: "고3" -> 고3 관련 성취기준만)
    # 교과목으로도 필터링 시도
    filtered_df = achievement_df.copy()

    # 간단한 학년 매핑 (필요시 확장)
    if '초' in grade:
        filtered_df = filtered_df[filtered_df['성취기준 코드'].str.startswith('2수')]
    elif '중' in grade:
        filtered_df = filtered_df[filtered_df['성취기준 코드'].str.startswith('9수')]
    elif '고1' in grade:
        filtered_df = filtered_df[filtered_df['성취기준 코드'].str.startswith('10')]
    elif '고2' in grade:
        filtered_df = filtered_df[filtered_df['성취기준 코드'].str.startswith(('10', '수1', '수2'))]
    elif '고3' in grade:
        filtered_df = filtered_df[filtered_df['성취기준 코드'].str.startswith('12')]

    achievement_list = filtered_df.to_dict('records')

    if not achievement_list:
        # 필터링 결과가 없으면 전체 목록 사용
        achievement_list = achievement_df.to_dict('records')

    achievement_str = "\n".join([
        f"- 코드: {row['성취기준 코드']}, 단원: {row['단원']}, 성취기준: {row['성취기준']}, 내용: {row['성취기준 내용']}"
        for row in achievement_list[:100]  # 최대 100개
    ])

    prompt = f"""
당신은 수학 교육과정 전문가입니다.

**이미 선택된 교육과정 정보:**
- 학년: {curriculum_info.get('학년', 'N/A')}
- 교과목: {curriculum_info.get('교과목', 'N/A')}
- 대단원: {curriculum_info.get('대단원', 'N/A')}
- 중단원: {curriculum_info.get('중단원', 'N/A')}
- 소단원: {curriculum_info.get('소단원', 'N/A')}

**해당 학년의 사용 가능한 성취기준 목록:**
{achievement_str}

**과제:**
주어진 수학 문제 이미지와 위에서 선택된 교육과정 정보를 고려하여,
위의 성취기준 목록에서 가장 적합한 성취기준 코드를 선택하세요.

**중요 지침:**
1. 반드시 위에 제공된 성취기준 목록에 있는 코드만 사용하세요.
2. 선택된 대단원, 중단원, 소단원과 관련성이 높은 성취기준을 선택하세요.
3. 문제의 내용과 성취기준 내용을 비교하여 가장 적합한 것을 선택하세요.

**출력 형식:**
반드시 다음 JSON 형식으로만 출력하세요:

{{
  "성취기준_코드": "선택한 성취기준 코드",
  "선택_이유": "이 성취기준을 선택한 이유 (1-2문장)"
}}

**주의사항:**
- 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
- 성취기준 코드는 위의 목록에서 정확히 복사하여 사용하세요.
"""
    return prompt

def create_metadata_extraction_prompt(curriculum, achievement_standards, consideration, total_points=10):
    """
    1)문제('./resource/question)과
    2)모범 답안('./resource/commentary) 분석하여 메타데이터 추출하는 프름포트 생성

    Args:
        curriculum (dict): 교육과정 데이터
        achievement_standards (dict): 성취기준 데이터
        consideration (dict): 고려사항 데이터
        total_points (int): 문제의 총 배점 (기본값: 10)

    Returns:
        str: 메타데이터 추출용 프롬프트
    """
    prompt = f"""
        당신은 수학 교육 전문가입니다. 
        주어진 수학 문제와 모범답안을 분석하여 체계적인 채점 가이드(메타데이터)를 생성해야 합니다.
        
        **교육과정 정보:**
        - 학년: {curriculum.get('학년', 'N/A')}
        - 교과목: {curriculum.get('교과목', 'N/A')}
        - 대단원: {curriculum.get('대단원', 'N/A')}
        - 중단원: {curriculum.get('중단원', 'N/A')}
        - 소단원: {curriculum.get('소단원', 'N/A')}

        **성취기준:**
        - 단원: {achievement_standards.get('단원', 'N/A')}
        - 성취기준: {achievement_standards.get('성취기준', 'N/A')}
        - 성취기준 코드: {achievement_standards.get('성취기준 코드', 'N/A')}
        - 성취기준 내용: {achievement_standards.get('성취기준 내용', 'N/A')}
        - 성취기준 해설: {achievement_standards.get('성취기준 해설', 'N/A')}
        
        **고려사항:**
        - 성취기준 대표코드: {consideration.get('성취기준 대표코드', 'N/A')}
        - 학습 내용 요약: {consideration.get('학습 내용 요약', 'N/A')}
        - 고려사항: {consideration.get('고려사항', 'N/A')}
        
        **과제:**
        이제 제공된 두 이미지를 분석하세요:
        1. 첫 번째 이미지: 수학 문제
        2. 두 번째 이미지: 모범답안 풀이
        
        모범답안을 분석하여 다음과 같은 채점 가이드를 생성하세요:
        
        **중요:** 출력은 반드시 다음 JSON 형식이어야 합니다:

        {{
          "curriculum_mapping": {{
            "대단원": "추출한 대단원 정보",
            "중단원": "추출한 중단원 정보",
            "소단원": "추출한 소단원 정보",
            "성취기준_코드": "추출한 성취기준 코드 정보"
          }},
          "problem_analysis": {{
            "problem_type": "문제 유형 (계산, 방정식 풀이, 증명, 응용, 그래프 해석/작도, 도형, 함수, 확률/통계, 개념 이해, 추론/논리 중 하나)",
            "difficulty": "난이도 (기초/기본/응용/심화)",
            "difficulty_reason": "난이도 설정 이유 (기초: 공식 직접 대입, 기본: 개념 이해 필요, 응용: 여러 개념 연결, 심화: 창의적 문제해결)",
            "required_concepts": ["필요한 개념1", "필요한 개념2", "필요한 개념3"],
            "problem_intent": "문제 출제 의도"
          }},
          "solution_steps": [
            {{
              "step_number": 1,
              "step_name": "첫 번째 단계 명칭",
              "description": "이 단계에서 수행해야 할 작업",
              "key_concept": "이 단계의 핵심 개념",
              "expected_action": "학생이 수행해야 할 구체적 행동",
              "points": 해당 단계 점수,
              "common_errors": ["흔한 실수1", "흔한 실수2"]
            }},
            {{
              "step_number": 2,
              "step_name": "두 번째 단계 명칭",
              "description": "이 단계에서 수행해야 할 작업",
              "key_concept": "이 단계의 핵심 개념",
              "expected_action": "학생이 수행해야 할 구체적 행동",
              "points": 해당 단계 점수,
              "common_errors": ["흔한 실수1", "흔한 실수2"]
            }}
          ],
          "total_points": {total_points},
          "correct_answer": "정답",
          "alternative_solutions": [
            {{
              "method_name": "대안 풀이법 1 이름",
              "description": "대안 풀이법 설명",
              "is_valid": true
            }}
          ],
          "grading_considerations": [
            "채점시 고려해야 할 사항1",
            "채점시 고려해야 할 사항2",
            "동치 답안 처리 방법"
          ]
        }}
        
        **요구사항:**
        1. 풀이 과정을 최소 3개 이상의 단계로 나누어 분석하세요.
        2. 각 단계별로 명확한 채점 기준과 배점을 제시하세요.
        3. **중요: 모든 단계의 점수 합계는 반드시 {total_points}점이 되어야 합니다.** 각 단계의 중요도에 따라 점수를 적절히 배분하세요.
        4. **중요: "성취기준_코드"는 반드시 위에 제공된 성취기준 정보의 "성취기준 코드" 값({achievement_standards.get('성취기준 코드', 'N/A')})을 그대로 사용하세요. 절대 새로운 코드를 생성하지 마세요.**
        5. 동치 답안이나 대안 풀이법이 있다면 반드시 명시하세요.
        6. 학생들이 자주 범하는 실수나 오개념을 예측하여 포함하세요.
        7. 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
        """
    return prompt

def create_analysis_prompt(curriculum, achievement_standards, consideration, metadata):
    """
    학생의 풀이를 분석하고 채점하는 프롬프트를 생성합니다.

    Args:
        curriculum (dict): 교육과정 데이터
        achievement_standards (dict): 성취기준 데이터
        consideration (dict): 고려사항 데이터
        metadata (dict): 메타데이터 추출 단계에서 생성된 채점 가이드

    Returns:
        str: 학생 풀이 분석용 프롬프트
    """

    prompt = f"""
        당신은 수학 전문가 입니다. 학생의 필기 풀이를 분석하고 공정하게 채점해야 합니다.

        **채점 가이드(메타데이터):**
        {json.dumps(metadata, ensure_ascii=False, indent=2)}

        **과제:**
        이제 제공된 두 이미지를 분석하세요:
        1. 첫 번째 이미지: 수학 문제
        2. 두 번째 이미지: 학생의 필기 풀이

        위의 채점 가이드를 기준으로 학생의 풀이를 단계별로 평가하세요.

        **중요 평가 원칙:**
        1. **모범답안과 다른 풀이도 인정:** 학생이 모범답안과 다른 방법으로 풀었더라도, 수학적으로 타당하다면 동일하게 점수를 부여하세요.
        2. **부분 점수 부여:** 최종 답이 틀렸더라도 중간 과정이 올바르다면 부분 점수를 부여하세요.
        3. **오류 분석:** 학생이 실수한 부분은 다음과 같이 구분하세요:
           - **계산 실수:** 개념은 이해했으나 단순 계산 오류
           - **개념 오류:** 수학적 개념을 잘못 이해한 경우
           - **풀이 중단:** 특정 단계에서 더 이상 진행하지 못한 경우
        4. **근거 명시:** 모든 점수에 대해 명확한 근거를 제시하세요.
        5. **수학적 방법 검증:** 학생이 사용한 수학적 공식, 정리, 개념이 올바른지 검증하세요.

        **출력 형식:**
        반드시 다음 JSON 형식으로만 출력하세요:

        {{
          "student_approach": "학생이 사용한 풀이 방법 요약",
          "mathematical_methods_used": [
            {{
              "method_name": "사용된 수학적 방법/공식/정리 명칭",
              "is_valid": true/false,
              "validation_comment": "해당 방법이 이 문제에 적절한지 또는 올바르게 적용되었는지에 대한 평가"
            }}
          ],
          "is_alternative_method": true/false,
          "step_by_step_evaluation": [
            {{
              "step_number": 1,
              "step_name": "단계 명칭",
              "student_work": "학생이 이 단계에서 수행한 작업",
              "status": "Correct/Incorrect/Partial/NotAttempted",
              "points_earned": 획득 점수,
              "points_possible": 배점,
              "evaluation": "이 단계에 대한 평가",
              "error_type": "계산 실수/개념 오류/풀이 중단/없음",
              "feedback": "학생에게 제공할 구체적 피드백"
            }}
          ],
          "final_score": 총 획득점수,
          "total_possible": 총 배점,
          "correct_answer": "정답",
          "student_answer": "학생이 제시한 답",
          "answer_match": true/false,
          "overall_evaluation": {{
            "strengths": ["학생의 강점1", "학생의 강점2"],
            "weaknesses": ["개선이 필요한 부분1", "개선이 필요한 부분2"],
            "summary": "전체적인 평가 요약"
          }},
          "detailed_feedback": "학생이 이해할 수 있도록 작성된 상세한 피드백. 왜 이 점수가 나왔는지, 어떤 부분을 개선해야 하는지 명확히 설명.",
          "improvement_suggestions": [
            "구체적인 개선 방안1",
            "구체적인 개선 방안2",
            "구체적인 개선 방안3"
          ]
        }}

        **주의 사항:**
        - 순수 하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
        - 학생의 풀이가 읽기 어렵거나 불완전하더라도 최대한 이해하려 노력하고 부분 점수를 공정하게 부여하세요.
        - 필기가 흐려서 판독이 어려운 경우 그 사실을 피드백에 명시하세요.
        """
    return prompt


def create_vision_analysis_prompt(metadata, box_map):
    """
    Google Vision API 기반으로 시각화된(ID가 표시된) 이미지를 분석하여,
    오류가 발생한 정확한 위치(Box ID)를 식별하는 프롬프트를 생성합니다.

    Args:
        metadata (dict): 채점 가이드 (메타데이터)
        box_map (list): 바운딩 박스 ID와 텍스트 매핑 정보
                        [{"id": 0, "text": "...", "box": [...]}, ...]

    Returns:
        str: Vision 기반 정밀 분석용 프롬프트
    """
    
    # 텍스트 매핑 정보를 문자열로 변환 (컨텍스트 제공용)
    box_map_str = json.dumps(box_map, ensure_ascii=False, indent=1)

    prompt = f"""
        당신은 수학 전문가이자 정밀한 풀이 분석가입니다.
        제공된 이미지는 학생의 필기 풀이 위에 컴퓨터 비전 기술로 텍스트 블록을 감지하고 **고유 ID(숫자)**를 붙인 것입니다.
        
        **입력 자료:**
        1. **채점 가이드(메타데이터):**
        {json.dumps(metadata, ensure_ascii=False, indent=2)}

        2. **텍스트 블록 ID 매핑 정보(참고용):**
        {box_map_str}

        **과제:**
        제공된 **'ID가 표시된 풀이 이미지'**를 시각적으로 정밀하게 분석하여 다음을 수행하세요:

        1. **단계별 풀이 검증:** 메타데이터의 solution_steps에 정의된 각 단계를 기준으로 학생의 풀이를 평가하세요.
        2. **오류 위치 핀포인트:** 만약 풀이 과정에 오류(계산 실수, 개념 오류 등)가 있다면, **해당 오류가 처음 시작된 정확한 위치의 'Box ID'**를 찾으세요.
           - 이미지를 보고 오류가 있는 수식이나 숫자가 포함된 박스의 ID 번호를 확인하세요.
           - 오류가 없다면 Box ID는 null 입니다.

        **출력 형식:**
        반드시 다음 JSON 형식으로만 출력하세요.
        **중요:** 이 프롬프트는 오류 위치 핀포인트에 집중합니다. 점수는 계산하지 마세요.

        {{
          "student_approach": "학생이 사용한 풀이 방법 요약",
          "is_correct": true/false,
          "step_by_step_analysis": [
             {{
                "step_number": 메타데이터의 step_number,
                "step_name": "메타데이터의 step_name",
                "student_work": "학생이 이 단계에서 수행한 작업 설명",
                "status": "Correct/Incorrect/Partial/NotAttempted",
                "related_box_ids": [해당 단계와 관련된 Box ID들],
                "evaluation": "이 단계에 대한 평가"
             }}
          ],
          "first_error_location": {{
             "has_error": true/false,
             "error_step_number": 오류가 발생한 step_number (없으면 null),
             "error_box_id": 오류가 포함된 가장 구체적인 Box ID (숫자, 없으면 null),
             "reason": "오류라고 판단한 근거"
          }},
          "overall_feedback": "학생 풀이에 대한 전반적인 피드백"
        }}

        **중요 지침:**
        1. **이 분석의 목적은 오류가 어디서 시작되었는지 찾는 것입니다. 점수는 계산하지 마세요.**
        2. step_by_step_analysis의 각 항목은 메타데이터의 solution_steps와 1:1 매핑되어야 합니다.
        3. step_number와 step_name은 메타데이터에서 정의된 값을 정확히 사용하세요.
        4. status는 "Correct/Incorrect/Partial/NotAttempted" 중 하나만 사용하세요.
        5. `error_box_id`는 이미지에 표시된 **숫자 ID**와 정확히 일치해야 합니다.
        6. 텍스트 매핑 정보를 참고하되, 최종 판단은 **이미지**를 보고 하세요.
        7. 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
    """
    return prompt


def create_step_validation_prompt(metadata, box_map):
    """
    학생 풀이의 각 스텝을 검증하고 LaTeX 근거와 함께 bbox ID를 출력하는 프롬프트 생성

    Args:
        metadata (dict): 채점 가이드 (메타데이터)
        box_map (list): 바운딩 박스 ID와 텍스트 매핑 정보
                        [{"id": 0, "text": "...", "box": [...]}, ...]

    Returns:
        str: 스텝별 검증 프롬프트
    """

    box_map_str = json.dumps(box_map, ensure_ascii=False, indent=1)

    prompt = f"""
        당신은 수학 전문가입니다.
        제공된 이미지는 학생의 필기 풀이 위에 **바운딩 박스 ID(숫자)**가 표시된 것입니다.

        **입력 자료:**
        1. **채점 가이드(메타데이터):**
        {json.dumps(metadata, ensure_ascii=False, indent=2)}

        2. **텍스트 블록 ID 매핑 정보 (참고용):**
        {box_map_str}

        **과제:**
        메타데이터의 solution_steps에 정의된 각 단계를 기준으로 학생의 풀이를 검증하세요.
        각 스텝마다 다음을 수행해야 합니다:

        1. **학생이 작성한 내용을 LaTeX 형식으로 추출**: 이미지에서 해당 스텝에 해당하는 수식/계산을 LaTeX로 표현하세요.
        2. **정답 여부 확인**: 해당 스텝의 풀이가 올바른지 확인하세요.
        3. **검증 근거 제시**: LaTeX로 어떻게 확인했는지 구체적인 수학적 근거를 제시하세요.
        4. **관련 Box ID 명시**: 해당 스텝의 풀이가 포함된 모든 Box ID를 나열하세요.

        **출력 형식:**
        반드시 다음 JSON 형식으로만 출력하세요.
        **중요:** step_validations의 각 항목은 메타데이터의 solution_steps와 1:1 매핑되어야 합니다.

        {{
          "overall_summary": "전체 풀이에 대한 간단한 요약",
          "step_validations": [
            {{
              "step_number": 메타데이터의 step_number,
              "step_name": "메타데이터의 step_name",
              "student_work_latex": "학생이 작성한 내용을 LaTeX 형식으로 표현 (예: $x = \\\\frac{{-b \\\\pm \\\\sqrt{{b^2-4ac}}}}{{2a}}$)",
              "is_correct": true/false,
              "validation_evidence": "어떻게 확인했는지에 대한 구체적인 수학적 근거. LaTeX 수식을 포함하여 단계별로 설명",
              "related_box_ids": [관련된 Box ID 배열],
              "error_description": "틀렸다면 어떤 오류인지 설명 (맞으면 null)",
              "points_earned": 획득 점수,
              "points_possible": 메타데이터의 배점
            }}
          ],
          "final_score": 총 획득 점수,
          "total_possible": 총 배점,
          "overall_feedback": "전체적인 피드백"
        }}

        **중요 지침:**
        1. **이 분석의 목적은 실제 채점 및 점수 부여입니다. 각 단계별 점수를 정확히 계산하세요.**
        2. step_validations의 각 항목은 메타데이터의 solution_steps와 1:1 매핑되어야 합니다.
        3. step_number와 step_name은 메타데이터에서 정의된 값을 정확히 사용하세요.
        4. points_possible은 메타데이터의 해당 단계 points 값을 사용하세요.
        5. points_earned는 학생이 실제로 획득한 점수를 엄격하게 계산하세요.
        6. `student_work_latex`는 반드시 LaTeX 형식으로 작성하세요. 예: $\\\\frac{{1}}{{2}}$, $x^2 + 2x + 1$
        7. `validation_evidence`에는 "학생이 $x = 3$을 대입했고, $f(3) = 3^2 + 1 = 10$으로 계산했습니다. 정답은 $f(3) = 10$이므로 올바릅니다."와 같이 LaTeX를 포함한 구체적인 검증 과정을 작성하세요.
        8. `related_box_ids`는 이미지에서 해당 스텝의 풀이가 포함된 모든 Box ID를 포함해야 합니다.
        9. 학생의 풀이가 모범답안과 다르더라도, 수학적으로 올바르다면 정답으로 인정하세요.
        10. 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.

        **주의사항:**
        - Box ID는 이미지에 표시된 숫자와 정확히 일치해야 합니다.
        - LaTeX 형식을 사용할 때 백슬래시는 이중으로 사용하세요 (\\\\frac, \\\\sqrt 등).
        - 텍스트 매핑 정보를 참고하되, 최종 판단은 이미지를 보고 하세요.
    """
    return prompt
