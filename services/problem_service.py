import pandas as pd
from pathlib import Path
import json
import logging
from common.gemini import Gemini

gemini_client = Gemini()
logger = logging.getLogger(__name__)


def get_problems_from_list():
    """
    Reads the list of problems to be processed from resource/list.csv.
    """
    try:
        df = pd.read_csv('resource/list.csv', encoding='utf-8-sig')
        # add a unique id to each row
        df['id'] = df.index
        return df.to_dict('records')
    except FileNotFoundError:
        logger.error("resource/list.csv not found.")
        return []


def extract_metadata(problem_id):
    """
    Extracts metadata for a given problem by analyzing the question and commentary images.
    """
    from common.prompt import create_metadata_extraction_prompt

    problem_data = get_problem_data(problem_id)
    if not problem_data:
        return {"error": f"Could not load metadata for problem {problem_id}"}, 404

    question_image_path = f"resource/question/{problem_id}.png"
    commentary_image_path = f"resource/commentary/{problem_id}.png"

    full_question_path = Path(question_image_path)
    full_commentary_path = Path(commentary_image_path)

    if not full_question_path.exists() or not full_commentary_path.exists():
        return {"error": "Image file not found on server"}, 404

    try:
        question_image_part = {"mime_type": "image/png", "data": full_question_path.read_bytes()}
        commentary_image_part = {"mime_type": "image/png", "data": full_commentary_path.read_bytes()}
    except IOError as e:
        logger.error(f"File reading error: {e}")
        return {"error": "Could not read image files"}, 500

    prompt = create_metadata_extraction_prompt(problem_data['curriculum'], problem_data['achievement_standards'],
                                             problem_data['consideration'])

    contents = [
        prompt,
        "아래 이미지는 실제 문제지입니다.",
        question_image_part,
        "아래 이미지는 해설 이미지입니다.",
        commentary_image_part
    ]

    try:
        response_text = gemini_client.call_gemini_multimodal(contents)
        metadata = json.loads(response_text)
        return metadata, 200
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from Gemini response. Response text: {response_text}")
        return {"error": "Failed to parse the metadata from the AI. The response was not valid JSON."}, 500
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return {"error": "An unexpected error occurred while extracting metadata."}, 500


def analyze_student_solution(problem_id, student_solution_filename, metadata):
    """
    Analyzes a student's solution based on the extracted metadata.
    """
    from common.prompt import create_analysis_prompt

    problem_data = get_problem_data(problem_id)
    if not problem_data:
        return {"error": f"Could not load metadata for problem {problem_id}"}, 404

    solution_image_path = f"resource/solve/{student_solution_filename}"
    question_image_path = f"resource/question/{problem_id}.png"

    full_solution_path = Path(solution_image_path)
    full_question_path = Path(question_image_path)

    if not full_solution_path.exists() or not full_question_path.exists():
        return {"error": "Image file not found on server"}, 404

    try:
        solution_image_part = {"mime_type": "image/png", "data": full_solution_path.read_bytes()}
        question_image_part = {"mime_type": "image/png", "data": full_question_path.read_bytes()}
    except IOError as e:
        logger.error(f"File reading error: {e}")
        return {"error": "Could not read image files"}, 500

    prompt = create_analysis_prompt(problem_data['curriculum'], problem_data['achievement_standards'],
                                    problem_data['consideration'], metadata)

    contents = [
        prompt,
        "아래 이미지는 실제 문제지입니다.",
        question_image_part,
        "아래 이미지는 학생의 문제 풀이 과정입니다.",
        solution_image_part
    ]

    try:
        response_text = gemini_client.call_gemini_multimodal(contents)
        analysis_json = json.loads(response_text)
        return analysis_json, 200
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from Gemini response. Response text: {response_text}")
        return {"error": "Failed to parse the analysis from the AI. The response was not valid JSON."}, 500
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return {"error": "An unexpected error occurred while analyzing the solution."}, 500


def analyze_problem_from_list(problem_id, student_solution_filename):
    """
    Orchestrates the two-step analysis for a problem from the list.
    """
    # Step 1: Extract metadata
    metadata, status_code = extract_metadata(problem_id)
    if status_code != 200:
        return metadata, status_code

    # Step 2: Analyze student solution
    analysis, status_code = analyze_student_solution(problem_id, student_solution_filename, metadata)
    if status_code != 200:
        return analysis, status_code

    return analysis, 200


def get_problems():
    """
    Scans the resource directory to find all available problems and their solutions.
    Returns a sorted list of problem dictionaries.
    """
    problem_resource_path = Path("resource")
    question_dir = problem_resource_path / "question"
    solution_dir = problem_resource_path / "solve"  # Corrected from "png"

    problems = {}

    # Scan for question images
    for f in question_dir.glob("*.png"): # Corrected from "question-*.png"
        problem_id = f.stem
        problems[problem_id] = {
            "id": problem_id,
            "question_image": str(Path(question_dir.name) / f.name).replace("\\", "/"),
            "solutions": []
        }

    # Scan for solution images
    for f in solution_dir.glob("*.png"):
        try:
            problem_id = f.stem.split('-')[0]
            solution_num = f.stem.split('-')[1]
            if problem_id in problems:
                problems[problem_id]["solutions"].append({
                    "id": solution_num,
                    "solution_image": str(Path(solution_dir.name) / f.name).replace("\\", "/"),
                })
        except IndexError:
            # Ignore files that don't match the 'problem_id-solution_num' format
            continue

    # Sort problems by ID and solutions by their number for consistent ordering
    for problem in problems.values():
        problem['solutions'].sort(key=lambda s: s['id'])

    return sorted(list(problems.values()), key=lambda p: p['id'])


def get_problem_data(problem_id):
    """
    Loads all relevant CSV data for a given problem ID.

    Note: CSV 파일에 문제번호 컬럼이 없으므로, 일반적인 교육과정 정보를 로드합니다.
    향후 문제번호별 매핑 정보가 추가되면 수정이 필요합니다.

    Returns a dictionary containing curriculum data or None if not found.
    """
    try:
        # CSV 파일 로드 (encoding 명시)
        df_achievement = pd.read_csv('resource/achievement_standards.csv', encoding='utf-8-sig')
        df_consideration = pd.read_csv('resource/consideration.csv', encoding='utf-8-sig')
        df_curriculum = pd.read_csv('resource/curriculum.csv', encoding='utf-8-sig')

        # 문제번호 컬럼이 있는 경우 해당 행 조회
        if '문제번호' in df_achievement.columns:
            problem_id_int = int(problem_id)
            achievement_standards = df_achievement.query(f"`문제번호` == {problem_id_int}").to_dict('records')[0]
            consideration = df_consideration.query(f"`문제번호` == {problem_id_int}").to_dict('records')[0]
            curriculum = df_curriculum.query(f"`문제번호` == {problem_id_int}").to_dict('records')[0]
        else:
            # 문제번호 컬럼이 없는 경우: 첫 번째 행을 기본값으로 사용
            # (실제로는 문제 이미지 분석을 통해 교육과정을 매칭해야 함)
            logger.warning(f"CSV 파일에 '문제번호' 컬럼이 없습니다. 일반적인 데이터를 사용합니다.")

            # 고등학교 공통수학 데이터를 기본으로 사용
            curriculum_data = df_curriculum[df_curriculum['교과목'].str.contains('공통수학', na=False)]
            if len(curriculum_data) == 0:
                curriculum_data = df_curriculum

            achievement_standards = df_achievement.iloc[0].to_dict() if len(df_achievement) > 0 else {}
            consideration = df_consideration.iloc[0].to_dict() if len(df_consideration) > 0 else {}
            curriculum = curriculum_data.iloc[0].to_dict() if len(curriculum_data) > 0 else {}

        return {
            "achievement_standards": achievement_standards,
            "consideration": consideration,
            "curriculum": curriculum,
        }
    except (FileNotFoundError, IndexError, ValueError) as e:
        logger.error(f"Error loading data for problem {problem_id}: {e}")
        return None


def analyze_solution(problem_id, solution_image_path, question_image_path):
    """
    Analyzes a student's solution by calling the Gemini model via the Gemini service class.
    """
    from common.prompt import create_analysis_prompt

    problem_data = get_problem_data(problem_id)
    if not problem_data:
        return {"error": f"Could not load metadata for problem {problem_id}"}, 404

    full_question_path = Path("resource") / question_image_path
    full_solution_path = Path("resource") / solution_image_path

    if not full_question_path.exists() or not full_solution_path.exists():
        return {"error": "Image file not found on server"}, 404

    # Prepare images for the model
    try:
        question_image_part = {"mime_type": "image/png", "data": full_question_path.read_bytes()}
        solution_image_part = {"mime_type": "image/png", "data": full_solution_path.read_bytes()}
    except IOError as e:
        logger.error(f"File reading error: {e}")
        return {"error": "Could not read image files"}, 500

    # This function is now a two-step process.
    # For direct analysis, we'll just pass empty metadata.
    metadata = {}

    # Create the prompt
    prompt = create_analysis_prompt(problem_data['curriculum'], problem_data['achievement_standards'],
                                    problem_data['consideration'], metadata)

    # Build the contents list for the multimodal call
    contents = [
        prompt,
        "아래 이미지는 실제 문제지입니다.",
        question_image_part,
        "아래 이미지는 학생의 문제 풀이 과정입니다.",
        solution_image_part
    ]

    try:
        # Use the Gemini class to make the API call
        response_text = gemini_client.call_gemini_multimodal(contents)
        
        # The response text should be a JSON string. We parse it to send a proper JSON object.
        analysis_json = json.loads(response_text)
        return analysis_json, 200

    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from Gemini response. Response text: {response_text}")
        return {"error": "Failed to parse the analysis from the AI. The response was not valid JSON."}, 500
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return {"error": "An unexpected error occurred while analyzing the solution."}, 500