# 수학 문제 자동 채점 시스템 PoC

## 1. PoC 개요
"일관된" 채점 기준으로, "최소한"의 사용자 개입으로, 학생들이 푼 수학 문제를 채점 수행하는 것을 목표로 합니다.

### 핵심 포인트
- 문제별 객관적 채점기준(메타데이터) 생성
- 학생 필기 풀이와 모범 답안(채점기준) 간 정합성 평가
- Gemini 기반 자동 채점 및 근거 제공

## 2. PoC 시나리오

### 입력 데이터(고객사 제공)
1. 수학문제 이미지 (`resource/question/{문제번호}.png`)
2. 모범답안 이미지 (`resource/commentary/{문제번호}.png`)
3. 학생 문제풀이 필기 이미지 (`resource/solve/{문제번호}-{회차}.png`)
4. 학생 답안 정보 (`resource/list.csv`)

### 주요 단계

#### 1단계: 개별 수학 문제의 메타데이터 추출
- 교육과정(별책 수학과 교육과정 2022년) 기준 분석
- 출제자 의도 및 채점 기준 정의
- 단계별 점수 체계 생성
- 최소 3개 이상의 풀이 단계로 분해
- 동치 답안 및 대안 풀이법 정의

#### 2단계: 학생 문제풀이(필기) 분석
- 추출된 메타데이터 판단 기준을 기준으로 채점
- 각 단계(부분) 별 점수화
- 점수에 대한 근거(description 명시)
  - 풀이 중단 시 → '다음 단계로 진행하지 못한 이유'
  - 계산 오류 시 → '계산 실수 / 개념 오류 등 구체적 근거'
- 모범답안과 다른 풀이도 수학적으로 타당하면 인정

### 3. 최종 인수조건

#### 평가 기준
1. 고객사로 부터 전달받은 "모범답안 풀이를 기준으로" 채점가이드를 생성하여 메타데이터 축적
2. 일관된 채점가이드(메타데이터)와 학생풀이의 부합하는 판정

#### 예외 처리
현재까지 전달 받는 데이터의 경우, "별도의 채점 기준이 없는" 상태에서 문제 풀이 된 풀이 자료로,
동일 수학 문제에 대해 N회독에 해당하는 풀이가 존재할 수 있습니다.

따라서, 학생의 풀이가 "모범 답안(기준)"과 다른 풀이 인 경우에는 Gemini의 추론 능력으로 별도의 채점 가이드를 생성하여
점수 및 근거를 description으로 제안합니다.

#### 최종 결과물
1. 각 학생 풀이 별 점수 및 근거
2. 최종적으로 엔드 유저(학생, 선생님)이 해당 평가 description을 통해 "왜 이 점수가 나왔는지" 명확히 이해할 수 있도록 설계

---

## 프로젝트 구조

```
meta-edu-poc/
├── common/
│   ├── gemini.py          # Gemini API 클라이언트
│   ├── logger.py          # 로깅 설정
│   └── prompt.py          # 프롬프트 템플릿
├── services/
│   └── problem_service.py # 문제 처리 비즈니스 로직
├── resource/
│   ├── question/          # 문제 이미지
│   ├── commentary/        # 모범답안 이미지
│   ├── solve/             # 학생 풀이 이미지
│   ├── list.csv           # 처리할 풀이 목록
│   ├── achievement_standards.csv  # 성취기준
│   ├── consideration.csv  # 고려사항
│   └── curriculum.csv     # 교육과정
├── process_all.py         # 배치 처리 스크립트
├── requirements.txt
└── README.md
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 정보를 입력하세요:

```env
API_KEY=your_google_api_key
PROJECT_ID=your_gcp_project_id
LOCATION=your_gcp_location
```

### 3. 데이터 준비

다음 디렉토리에 파일들을 배치하세요:
- `resource/question/`: 문제 이미지 파일들
- `resource/commentary/`: 모범답안 이미지 파일들
- `resource/solve/`: 학생 풀이 이미지 파일들
- `resource/list.csv`: 처리할 풀이 목록

### 4. 배치 처리 실행

```bash
python main.py
```

실행 결과는 `results/batch_{timestamp}/` 디렉토리에 저장됩니다:
- `metadata/`: 문제별 메타데이터 (채점 가이드)
- `analysis/`: 학생별 풀이 분석 결과
- `summary.json`: 전체 처리 결과 요약

## 주요 기능

### 1. 메타데이터 추출 (common/prompt.py:create_metadata_extraction_prompt)
- 문제와 모범답안 이미지 분석
- 교육과정 매핑
- 풀이 단계별 채점 기준 생성
- 동치 답안 및 대안 풀이법 정의

### 2. 학생 풀이 분석 (common/prompt.py:create_analysis_prompt)
- 메타데이터 기반 단계별 채점
- 오류 유형 분석 (계산 실수 / 개념 오류 / 풀이 중단)
- 부분 점수 부여
- 구체적 피드백 생성

### 3. 배치 처리 (process_all.py)
- list.csv의 모든 풀이 자동 처리
- 문제별 메타데이터 캐싱
- 처리 통계 및 오류 로깅
- 결과 파일 자동 저장

## 출력 형식

### 메타데이터 (metadata/{문제번호}_metadata.json)
```json
{
  "curriculum_mapping": { ... },
  "problem_analysis": { ... },
  "solution_steps": [ ... ],
  "total_points": 100,
  "correct_answer": "...",
  "alternative_solutions": [ ... ],
  "grading_considerations": [ ... ]
}
```

### 분석 결과 (analysis/{문제번호}_{풀이파일명}_analysis.json)
```json
{
  "problem_id": "223174",
  "solution_file": "223174-1.png",
  "student_answer": "...",
  "expected_result": "PASS/FAIL",
  "analysis": {
    "student_approach": "...",
    "is_alternative_method": false,
    "step_by_step_evaluation": [ ... ],
    "final_score": 85,
    "total_possible": 100,
    "overall_evaluation": { ... },
    "detailed_feedback": "...",
    "improvement_suggestions": [ ... ]
  },
  "timestamp": "2025-11-28T..."
}
```

## 주의사항

1. **CSV 파일 인코딩**: 모든 CSV 파일은 UTF-8-BOM 인코딩을 사용해야 합니다.
2. **이미지 파일명**:
   - 문제: `{문제번호}.png`
   - 모범답안: `{문제번호}.png`
   - 학생풀이: `{문제번호}-{회차}.png`
3. **API 비용**: Gemini API 사용에 따른 비용이 발생할 수 있습니다.
4. **처리 시간**: 문제당 평균 10-30초 소요됩니다.

## 문제 해결

### 이미지 파일을 찾을 수 없음
- 파일 경로와 이름이 올바른지 확인
- list.csv의 파일명과 실제 파일명이 일치하는지 확인

### JSON 파싱 오류
- Gemini 응답이 JSON 형식이 아닐 수 있음
- 로그 파일에서 실제 응답 내용 확인
- 프롬프트 수정 필요할 수 있음

### API 오류
- .env 파일의 API_KEY, PROJECT_ID, LOCATION 확인
- GCP 프로젝트에서 Gemini API가 활성화되어 있는지 확인
- API 할당량 확인