import pandas as pd
from pathlib import Path
import json
from common.logger import init_logger
from common.gemini import Gemini
from utils.utils import fix_json_escaping


# 로거 초기화
logger = init_logger()


def _get_problem_data_from_metadata(curriculum_mapping):
    """
    메타데이터의 curriculum_mapping을 바탕으로 CSV에서 상세 정보 조회

    Args:
        curriculum_mapping (dict): 메타데이터에서 추출한 교육과정 정보

    Returns:
        dict: problem_data (curriculum, achievement_standards, consideration)
    """
    try:
        curriculum_df = pd.read_csv('resource/curriculum.csv', encoding='utf-8-sig')
        achievement_df = pd.read_csv('resource/achievement_standards.csv', encoding='utf-8-sig')
        consideration_df = pd.read_csv('resource/consideration.csv', encoding='utf-8-sig')

        # 교육과정 정보 (curriculum_mapping에서 직접 가져옴)
        curriculum = {
            '학년': curriculum_mapping.get('학년', 'N/A'),
            '교과목': curriculum_mapping.get('교과목', 'N/A'),
            '대단원': curriculum_mapping.get('대단원', 'N/A'),
            '중단원': curriculum_mapping.get('중단원', 'N/A'),
            '소단원': curriculum_mapping.get('소단원', 'N/A')
        }

        # 성취기준 정보
        achievement_code = curriculum_mapping.get('성취기준_코드', '')
        achievement_row = achievement_df[achievement_df['성취기준 코드'] == achievement_code]
        if not achievement_row.empty:
            achievement_row = achievement_row.iloc[0]
            achievement_standards = {
                '단원': achievement_row.get('단원', 'N/A'),
                '성취기준': achievement_row.get('성취기준', 'N/A'),
                '성취기준 코드': achievement_row.get('성취기준 코드', 'N/A'),
                '성취기준 내용': achievement_row.get('성취기준 내용', 'N/A'),
                '성취기준 해설': achievement_row.get('성취기준 해설', '')
            }
        else:
            achievement_standards = {
                '단원': 'N/A',
                '성취기준': 'N/A',
                '성취기준 코드': achievement_code,
                '성취기준 내용': 'N/A',
                '성취기준 해설': ''
            }

        # 고려사항 정보 (성취기준 대표코드로 매칭)
        achievement_base_code = achievement_code.split('-')[0] if '-' in achievement_code else achievement_code
        consideration_rows = consideration_df[consideration_df['성취기준 대표코드'] == achievement_base_code]
        if not consideration_rows.empty:
            consideration_row = consideration_rows.iloc[0]
            consideration = {
                '단원': consideration_row.get('단원', 'N/A'),
                '성취기준 대표코드': consideration_row.get('성취기준 대표코드', 'N/A'),
                '학습 내용 요약': consideration_row.get('학습 내용 요약', 'N/A'),
                '고려사항': consideration_row.get('고려사항', 'N/A')
            }
        else:
            consideration = {
                '단원': 'N/A',
                '성취기준 대표코드': achievement_base_code,
                '학습 내용 요약': 'N/A',
                '고려사항': 'N/A'
            }

        return {
            'curriculum': curriculum,
            'achievement_standards': achievement_standards,
            'consideration': consideration
        }

    except Exception as e:
        logger.error(f"CSV 데이터 조회 중 오류: {e}")
        return None


def get_problem_data(problem_id):
    """
    문제 ID에 해당하는 교육과정 데이터를 반환합니다.
    (더미 함수 - 하위 호환성을 위해 유지)

    Args:
        problem_id (str): 문제 ID

    Returns:
        dict: 교육과정 데이터 (curriculum, achievement_standards, consideration)
              문제를 찾을 수 없는 경우 None 반환
    """
    # 더미 데이터 반환 (하위 호환성)
    return {
        'curriculum': {
            '학년': '고1',
            '교과목': '공통수학(상)',
            '대단원': '방정식과 부등식',
            '중단원': '복소수와 이차방정식',
            '소단원': ''
        },
        'achievement_standards': {
            '단원': '수와 연산',
            '성취기준': '네 자리 이하의 수',
            '성취기준 코드': '2수01-01',
            '성취기준 내용': '수의 필요성을 인식하면서 수 개념을 이해하고, 수를 세고 읽고 쓸 수 있다.',
            '성취기준 해설': ''
        },
        'consideration': {
            '단원': '수와 연산',
            '성취기준 대표코드': '2수01',
            '학습 내용 요약': '위치적 기수법 기초',
            '고려사항': '수의 분해와 합성 활동을 통하여 수 감각을 기른다.'
        }
    }


def get_problems_from_list():
    """
    resource/list.csv 처리할 손글씨 풀이 목록

    Returns:
        list: CSV 파일의 각 행에 해당하는 딕셔너리 리스트.
              파일을 찾지 못할 경우 빈 리스트를 반환합니다.
    """
    try:
        df = pd.read_csv('resource/list.csv', encoding='utf-8-sig')
        return df.to_dict('records')
    except FileNotFoundError:
        logger.error("resource/list.csv 파일을 찾을 수 없습니다.")
        return []


def analyze_student_solution(problem_id, student_solution_filename, metadata, ocr_engine_name: str = 'latex_ocr'):
    """
    추출된 채점 기준(메타데이터)을 바탕으로 학생의 풀이 이미지를 분석하고 채점합니다.

    Args:
        problem_id (str): 분석할 문제의 고유 ID.
        student_solution_filename (str): 학생 풀이 이미지 파일명.
        metadata (dict): `extract_metadata`를 통해 사전에 추출된 채점 기준.
        ocr_engine_name (str): OCR 엔진 이름 ('latex_ocr', 'nougat_latex').

    Returns:
        tuple: (analysis_result, success_bool)
               성공 시 (분석 결과 딕셔너리, True), 실패 시 (None, False).
    """
    from common.prompt import create_analysis_prompt
    from services.ocr_service import get_ocr_engine
    from utils.grade_visualizer import group_bboxes_by_step

    # 메타데이터에서 curriculum_mapping 정보를 가져와서 problem_data 생성
    curriculum_mapping = metadata.get('curriculum_mapping', {})
    problem_data = _get_problem_data_from_metadata(curriculum_mapping)
    if not problem_data:
        logger.error(f"문제 {problem_id}에 대한 데이터를 로드할 수 없습니다.")
        return None, False

    try:
        # 이미지 파일 경로 설정
        solution_image_path = Path(f"resource/solve/{student_solution_filename}")
        question_image_path = Path(f"resource/question/{problem_id}.png")

        # 이미지 파일 존재 여부 확인
        if not solution_image_path.exists():
            raise FileNotFoundError(f"풀이 이미지를 찾을 수 없습니다: {solution_image_path}")
        if not question_image_path.exists():
            raise FileNotFoundError(f"문제 이미지를 찾을 수 없습니다: {question_image_path}")

        # Gemini 클라이언트 생성
        gemini_client = Gemini()

        # 이미지 파일 업로드
        question_image_part = gemini_client.client.files.upload(file=str(question_image_path))
        solution_image_part = gemini_client.client.files.upload(file=str(solution_image_path))

        # 프롬프트 생성
        prompt = create_analysis_prompt(
            problem_data['curriculum'],
            problem_data['achievement_standards'],
            problem_data['consideration'],
            metadata
        )

        # 콘텐츠 구성
        contents = [
            prompt,
            "아래 이미지는 실제 문제지입니다.",
            question_image_part,
            "아래 이미지는 학생의 문제 풀이 과정입니다.",
            solution_image_part
        ]

        # Gemini API 호출 (재시도 로직 포함)
        response_text = gemini_client.call_extract_metadata(contents)

        # 백슬래시 이스케이프 수정
        response_text_fixed = fix_json_escaping(response_text)

        # JSON 파싱
        analysis_json = json.loads(response_text_fixed)

        # ========== OCR 데이터 추가==========
        try:
            logger.info(f"OCR 처리 시작 ({ocr_engine_name} 엔진): {student_solution_filename}")
            ocr_engine = get_ocr_engine(ocr_engine_name)
            
            if ocr_engine:
                ocr_results = ocr_engine.extract_latex(str(solution_image_path))
                
                # `group_bboxes_by_step` 함수는 이전 Google Vision 출력 형식에 맞춰져 있을 수 있음
                # 새로운 엔진의 출력은 [{ "text": "...", "bbox": [...] }] 형식이므로,
                # 호환성을 위해 `group_bboxes_by_step`에 맞게 입력을 조정하거나,
                # 현재는 단순히 전체 결과를 저장합니다.
                
                # NOTE: 현재의 LatexOCREngine과 NougatLatexOCREngine은 이미지 전체에 대한
                # 단일 결과를 반환하므로, 단계별 그룹화는 의미가 없을 수 있습니다.
                # bbox 형식을 group_bboxes_by_step이 기대하는 x,y,width,height로 변환
                compatible_ocr_results = []
                for res in ocr_results:
                    x_min, y_min, x_max, y_max = res['bbox']
                    compatible_ocr_results.append({
                        "text": res['text'],
                        "bbox": {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min},
                        "level": "latex" # 새로운 레벨 추가
                    })

                step_grouped_bboxes = {}
                if 'step_by_step_evaluation' in analysis_json:
                    step_evaluations = analysis_json['step_by_step_evaluation']
                    # 이전 group_bboxes_by_step 함수가 호환된다면 그대로 사용
                    step_grouped_bboxes = group_bboxes_by_step(compatible_ocr_results, step_evaluations)

                analysis_json['ocr_data'] = {
                    'engine': ocr_engine_name,
                    'all_latex': compatible_ocr_results,
                    'step_grouped_bboxes': step_grouped_bboxes,
                    'total_latex_count': len(compatible_ocr_results)
                }
                logger.info(f"OCR 완료: {len(compatible_ocr_results)}개 LaTeX 추출")
            else:
                raise ValueError(f"OCR 엔진 '{ocr_engine_name}'을 찾을 수 없습니다.")

        except Exception as ocr_error:
            logger.warning(f"OCR 처리 실패 (분석은 계속 진행): {ocr_error}")
            analysis_json['ocr_data'] = {
                'engine': ocr_engine_name,
                'all_latex': [],
                'step_grouped_bboxes': {},
                'total_latex_count': 0,
                'error': str(ocr_error)
            }
        # ==========================================

        logger.info(f"✓ 문제 {problem_id}, 풀이 {student_solution_filename} 분석 성공")
        return analysis_json, True

    except (FileNotFoundError, IOError) as e:
        logger.error(f"이미지 파일 처리 실패: {e}")
        return None, False
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}")
        return None, False
    except Exception as e:
        logger.error(f"분석 중 예외 발생: {e}", exc_info=True)
        return None, False


def analyze_problem_from_list(problem_id, student_solution_filename):
    """
    메타데이터 추출과 학생 풀이 분석을 순차적으로 실행합니다.

    Args:
        problem_id (str): 분석할 문제의 고유 ID.
        student_solution_filename (str): 학생 풀이 이미지 파일명.

    Returns:
        tuple: (analysis_result, success_bool)
    """
    from extract_metadata import MetadataExtractor

    # 1단계: 메타데이터 추출
    extractor = MetadataExtractor()
    metadata, success = extractor.extract_single_problem(problem_id)
    if not success:
        return None, False

    # 2단계: 학생 풀이 분석
    analysis, success = analyze_student_solution(problem_id, student_solution_filename, metadata)
    if not success:
        return None, False

    return analysis, True


def get_problems():
    """
    'resource' 디렉토리를 스캔하여 이용 가능한 문제와 풀이 이미지 목록을 구성합니다.

    'resource/question' 폴더에서 문제 이미지를 찾고, 'resource/solve' 폴더에서
    '문제ID-풀이번호.png' 형식의 풀이 이미지들을 찾아 각 문제에 연결합니다.

    Returns:
        list: 각 문제의 ID, 문제 이미지 경로, 풀이 이미지 목록을 포함하는
              딕셔너리들의 리스트. ID를 기준으로 정렬하여 반환됩니다.
    """
    problem_resource_path = Path("resource")
    question_dir = problem_resource_path / "question"
    solution_dir = problem_resource_path / "solve"

    problems = {}

    # 문제 이미지 스캔
    for f in question_dir.glob("*.png"):
        problem_id = f.stem
        problems[problem_id] = {
            "id": problem_id,
            "question_image": str(Path(question_dir.name) / f.name).replace("\\\\", "/"),
            "solutions": []
        }

    # 풀이 이미지 스캔
    for f in solution_dir.glob("*.png"):
        try:
            problem_id, solution_num = f.stem.split('-')
            if problem_id in problems:
                problems[problem_id]["solutions"].append({
                    "id": solution_num,
                    "solution_image": str(Path(solution_dir.name) / f.name).replace("\\\\", "/"),
                })
        except ValueError:
            # '문제ID-풀이번호' 형식이 아닌 파일은 무시
            logger.debug(f"파일명 형식이 맞지 않아 건너뜁니다: {f.name}")
            continue

    # ID 기준으로 문제와 풀이를 정렬
    for problem in problems.values():
        problem['solutions'].sort(key=lambda s: s['id'])

    return sorted(list(problems.values()), key=lambda p: p['id'])


def analyze_solution(problem_id, solution_image_path, question_image_path):
    """
    학생의 풀이를 분석(메타데이터 사전 추출 과정 없음)

    Args:
        problem_id (str): 분석할 문제의 고유 ID.
        solution_image_path (str): 'resource/' 기준 학생 풀이 이미지 상대 경로.
        question_image_path (str): 'resource/' 기준 원본 문제 이미지 상대 경로.

    Returns:
        tuple: (analysis_result, success_bool)
    """
    from common.prompt import create_analysis_prompt

    problem_data = get_problem_data(problem_id)
    if not problem_data:
        return None, False

    try:
        # 이미지 파일 경로 설정
        full_question_path = Path("resource") / question_image_path
        full_solution_path = Path("resource") / solution_image_path

        # 이미지 파일 존재 여부 확인
        if not full_question_path.exists():
            raise FileNotFoundError(f"문제 이미지를 찾을 수 없습니다: {full_question_path}")
        if not full_solution_path.exists():
            raise FileNotFoundError(f"풀이 이미지를 찾을 수 없습니다: {full_solution_path}")

        # Gemini 클라이언트 생성
        gemini_client = Gemini()

        # 이미지 파일 업로드
        question_image_part = gemini_client.client.files.upload(file=str(full_question_path))
        solution_image_part = gemini_client.client.files.upload(file=str(full_solution_path))

        # 메타데이터 없이 분석을 시도하므로, 빈 메타데이터를 전달
        metadata = {}

        # 프롬프트 생성
        prompt = create_analysis_prompt(
            problem_data['curriculum'],
            problem_data['achievement_standards'],
            problem_data['consideration'],
            metadata
        )

        # 콘텐츠 구성
        contents = [
            prompt,
            "아래 이미지는 실제 문제지입니다.",
            question_image_part,
            "아래 이미지는 학생의 문제 풀이 과정입니다.",
            solution_image_part
        ]

        # Gemini API 호출 (재시도 로직 포함)
        response_text = gemini_client.call_extract_metadata(contents)

        # 백슬래시 이스케이프 수정
        response_text_fixed = fix_json_escaping(response_text)

        # JSON 파싱
        analysis_json = json.loads(response_text_fixed)
        logger.info(f"✓ 문제 {problem_id} 분석 성공")
        return analysis_json, True

    except (FileNotFoundError, IOError) as e:
        logger.error(f"이미지 파일 처리 실패: {e}")
        return None, False
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {e}")
        return None, False
    except Exception as e:
        logger.error(f"분석 중 예외 발생: {e}", exc_info=True)
        return None, False