import json

def create_metadata_extraction_prompt(curriculum, achievement_standards, consideration):
    """
    문제와 모범답안 이미지를 분석하여 메타데이터를 추출하는 프롬프트를 생성합니다.

    Args:
        curriculum (dict): 교육과정 데이터
        achievement_standards (dict): 성취기준 데이터
        consideration (dict): 고려사항 데이터

    Returns:
        str: 메타데이터 추출용 프롬프트
    """

    prompt = f"""
당신은 수학 교육 전문가입니다. 주어진 수학 문제와 모범답안을 분석하여 체계적인 채점 가이드(메타데이터)를 생성해야 합니다.

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
    "성취기준_코드": "해당 성취기준 코드"
  }},
  "problem_analysis": {{
    "problem_type": "문제 유형 (예: 방정식 풀이, 증명, 계산 등)",
    "difficulty": "난이도 (상/중/하)",
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
      "points": 해당단계점수,
      "common_errors": ["흔한 실수1", "흔한 실수2"]
    }},
    {{
      "step_number": 2,
      "step_name": "두 번째 단계 명칭",
      "description": "이 단계에서 수행해야 할 작업",
      "key_concept": "이 단계의 핵심 개념",
      "expected_action": "학생이 수행해야 할 구체적 행동",
      "points": 해당단계점수,
      "common_errors": ["흔한 실수1", "흔한 실수2"]
    }}
  ],
  "total_points": 총배점,
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
3. 동치 답안이나 대안 풀이법이 있다면 반드시 명시하세요.
4. 학생들이 자주 범하는 실수나 오개념을 예측하여 포함하세요.
5. 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
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

    # metadata가 비어있는 경우를 대비한 기본 프롬프트
    if not metadata:
        return create_prompt(curriculum, achievement_standards, consideration)

    prompt = f"""
당신은 수학 문제 채점 전문가입니다. 학생의 필기 풀이를 분석하고 공정하게 채점해야 합니다.

**교육과정 정보:**
- 학년: {curriculum.get('학년', 'N/A')}
- 교과목: {curriculum.get('교과목', 'N/A')}
- 대단원: {curriculum.get('대단원', 'N/A')}
- 중단원: {curriculum.get('중단원', 'N/A')}

**성취기준:**
- 성취기준 코드: {achievement_standards.get('성취기준 코드', 'N/A')}
- 성취기준 내용: {achievement_standards.get('성취기준 내용', 'N/A')}

**채점 가이드 (메타데이터):**
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

**출력 형식:**
반드시 다음 JSON 형식으로만 출력하세요:

{{
  "student_approach": "학생이 사용한 풀이 방법 요약",
  "is_alternative_method": true/false,
  "step_by_step_evaluation": [
    {{
      "step_number": 1,
      "step_name": "단계 명칭",
      "student_work": "학생이 이 단계에서 수행한 작업",
      "status": "Correct/Incorrect/Partial/NotAttempted",
      "points_earned": 획득점수,
      "points_possible": 배점,
      "evaluation": "이 단계에 대한 평가",
      "error_type": "계산 실수/개념 오류/풀이 중단/없음",
      "feedback": "학생에게 제공할 구체적 피드백"
    }}
  ],
  "final_score": 총획득점수,
  "total_possible": 총배점,
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

**주의사항:**
- 순수하게 JSON만 출력하세요. JSON 외의 텍스트를 포함하지 마세요.
- 학생의 풀이가 읽기 어렵거나 불완전하더라도 최대한 이해하려 노력하고 부분 점수를 공정하게 부여하세요.
- 필기가 흐려서 판독이 어려운 경우 그 사실을 피드백에 명시하세요.
"""
    return prompt


def create_prompt(curriculum, achievement_standards, consideration):
    """
    Creates a detailed prompt for the Gemini model to analyze a student's math solution
    and return the analysis in a JSON format.

    (레거시 함수 - 하위 호환성을 위해 유지)

    Args:
        curriculum (dict): Data about the curriculum.
        achievement_standards (dict): Data about the achievement standards for the problem.
        consideration (dict): Data about the problem's intent and other considerations.

    Returns:
        str: A formatted prompt string.
    """

    prompt = f"""
        You are an expert AI assistant for grading middle school math problems.
        Your task is to analyze a student's handwritten solution based on a given problem and a set of evaluation criteria.

        Please perform the following steps:
        1.  Review the problem's context, including the curriculum, achievement standards, and evaluation criteria.
        2.  Examine the provided image of the original math problem.
        3.  Carefully analyze the student's handwritten solution provided as a separate image.
        4.  Compare the student's work against the model answer and scoring guide.
        5.  Score the solution step-by-step, identifying any errors, conceptual misunderstandings, or parts where the student stopped.
        6.  Provide a final score and a detailed, constructive explanation for the score.

        **Problem Context:**

        **1. Curriculum Information:**
        - **Target Grade:** {curriculum.get('대상')}
        - **Unit/Topic:** {curriculum.get('단원/주제')}
        - **Instructional Intent:** {curriculum.get('수업설계의도')}

        **2. Achievement Standards & Evaluation Criteria:**
        - **Achievement Code:** {achievement_standards.get('성취코드')}
        - **Standard:** {achievement_standards.get('성취기준')}
        - **Content:** {achievement_standards.get('성취기준 내용')}
        - **Evaluation Elements:** {achievement_standards.get('평가요소')}
        - **Scoring:** {achievement_standards.get('배점')}
        - **Model Answer/Scoring Guide:** {achievement_standards.get('정답')}

        **3. Problem Intent:**
        - **Intent:** {consideration.get('문항 설계 의도')}

        **Analysis Task:**

        Now, analyze the student's solution based on the two images that will be provided: the problem image and the student's solution image.

        **IMPORTANT**: Your entire output must be a single JSON object. Do not include any text outside of the JSON structure.

        Provide your analysis in the following JSON format:
        {{
          "step_by_step_analysis": [
            {{
              "step": "Description of student's step 1",
              "status": "Correct/Incorrect/Partial",
              "reasoning": "Reasoning/Feedback for this step"
            }},
            {{
              "step": "Description of student's step 2",
              "status": "Correct/Incorrect/Partial",
              "reasoning": "Reasoning/Feedback for this step"
            }}
          ],
          "final_score": "<Score>",
          "total_score": "{achievement_standards.get('배점')}",
          "overall_feedback": "A summary of the student's performance, highlighting strengths and areas for improvement."
        }}

        If the student's approach is different from the model answer but is still mathematically valid, create a new scoring guide for that approach and evaluate the solution against it, still following the JSON output format.
        """
    return prompt
