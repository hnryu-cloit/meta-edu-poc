"""
메타데이터 추출 전용 스크립트

수학 문제와 모범답안(./resource/commentary)를 분석하여 채점 가이드(메타데이터)를 추출
    1. 단일 문제 처리 또는 배치 처리 지원
    2. JSON 형식으로 결과 저장
    3. 커맨드라인에서 독립적으로 실행 가능
"""

import json
import argparse

from pathlib import Path
from datetime import datetime
from common.logger import init_logger
from common.gemini import Gemini
from common.prompt import create_metadata_extraction_prompt
from common.utils import fix_json_escaping
import pandas as pd


class MetadataExtractor:
    """메타데이터 추출 클래스"""

    def __init__(self, output_dir="metadata"):
        """
        Args:
            output_dir(str): 메타데이터 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = init_logger()
        self.gemini = Gemini()

        # 통계
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

    def extract_curriculum_mapping(self, problem_id):
        """
        1단계: 문제 이미지를 분석하여 적절한 교육과정 정보 추출 (2단계로 세분화)

        Args:
            problem_id (str): 문제 ID

        Returns:
            tuple: (curriculum_mapping_dict, success_bool)
        """
        self.logger.info(f"[1단계] 교육과정 정보 추출 시작: 문제 {problem_id}")

        try:
            from common.prompt import create_curriculum_selection_prompt, create_achievement_selection_prompt

            # 이미지 파일 경로 설정
            question_image_path = Path(f"resource/question/{problem_id}.png")

            # 이미지 파일 존재 여부 확인
            if not question_image_path.exists():
                raise FileNotFoundError(f"문제 이미지를 찾을 수 없습니다: {question_image_path}")

            # 이미지 파일 업로드
            question_image_part = self.gemini.client.files.upload(file=str(question_image_path))

            # ===== 1-1단계: curriculum 선택 =====
            self.logger.info(f"  [1-1단계] curriculum.csv에서 교육과정 선택 중...")
            curriculum_prompt = create_curriculum_selection_prompt()
            curriculum_contents = [
                curriculum_prompt,
                "아래 이미지는 분석할 수학 문제입니다.",
                question_image_part
            ]

            # Gemini API 호출
            curriculum_response = self.gemini.call_extract_metadata(curriculum_contents)
            # 백슬래시 이스케이프 수정
            curriculum_response_fixed = fix_json_escaping(curriculum_response)
            curriculum_info = json.loads(curriculum_response_fixed)

            self.logger.info(f"  ✓ [1-1단계] curriculum 선택 완료")
            self.logger.info(f"    학년: {curriculum_info.get('학년', 'N/A')}, 교과목: {curriculum_info.get('교과목', 'N/A')}")
            self.logger.info(f"    대단원: {curriculum_info.get('대단원', 'N/A')} > 중단원: {curriculum_info.get('중단원', 'N/A')}")

            # ===== 1-2단계: achievement_standards에서 성취기준 선택 =====
            self.logger.info(f"  [1-2단계] achievement_standards.csv에서 성취기준 코드 선택 중...")
            achievement_prompt = create_achievement_selection_prompt(curriculum_info)
            achievement_contents = [
                achievement_prompt,
                "아래 이미지는 분석할 수학 문제입니다.",
                question_image_part
            ]

            # Gemini API 호출
            achievement_response = self.gemini.call_extract_metadata(achievement_contents)
            # 백슬래시 이스케이프 수정
            achievement_response_fixed = fix_json_escaping(achievement_response)
            achievement_info = json.loads(achievement_response_fixed)

            self.logger.info(f"  ✓ [1-2단계] 성취기준 코드 선택 완료")
            self.logger.info(f"    성취기준 코드: {achievement_info.get('성취기준_코드', 'N/A')}")

            # ===== 1-1단계와 1-2단계 결과 병합 =====
            curriculum_mapping = {
                "학년": curriculum_info.get('학년', ''),
                "교과목": curriculum_info.get('교과목', ''),
                "대단원": curriculum_info.get('대단원', ''),
                "중단원": curriculum_info.get('중단원', ''),
                "소단원": curriculum_info.get('소단원', ''),
                "성취기준_코드": achievement_info.get('성취기준_코드', '')
            }

            self.logger.info(f"✓ [1단계] 교육과정 정보 추출 성공: {problem_id}")
            return curriculum_mapping, True

        except (FileNotFoundError, IOError) as e:
            error_msg = f"파일 처리 실패: {e}"
            self.logger.error(error_msg)
            return None, False
        except json.JSONDecodeError as e:
            error_msg = f"JSON 파싱 실패: {e}"
            self.logger.error(error_msg)
            try:
                if 'curriculum_response' in locals():
                    self.logger.error(f"원본 응답 내용:\n{curriculum_response}")
                    self.logger.error(f"수정된 응답 내용:\n{curriculum_response_fixed}")
                elif 'achievement_response' in locals():
                    self.logger.error(f"원본 응답 내용:\n{achievement_response}")
                    self.logger.error(f"수정된 응답 내용:\n{achievement_response_fixed}")
            except Exception as log_error:
                self.logger.error(f"로그 출력 실패: {log_error}")
            return None, False
        except Exception as e:
            error_msg = f"교육과정 정보 추출 중 예외 발생: {e}"
            self.logger.error(error_msg, exc_info=True)
            return None, False

    def extract_single_problem(self, problem_id, total_points=10):
        """
        단일 문제의 메타데이터를 추출 (2단계 처리)

        Args:
            problem_id (str): 문제 ID
            total_points (int): 문제의 총 배점 (기본값: 10)

        Returns:
            tuple: (metadata_dict, success_bool)
        """
        self.logger.info(f"메타데이터 추출 시작: 문제 {problem_id}")

        try:
            # ===== 1단계: 교육과정 정보 추출 =====
            curriculum_mapping, success = self.extract_curriculum_mapping(problem_id)
            if not success or not curriculum_mapping:
                raise ValueError(f"교육과정 정보 추출 실패: {problem_id}")

            # CSV에서 상세 정보 가져오기
            problem_data = self._get_problem_data_from_mapping(curriculum_mapping)
            if not problem_data:
                raise ValueError(f"교육과정 매핑 데이터를 찾을 수 없습니다: {problem_id}")

            # ===== 2단계: 메타데이터 추출 =====
            self.logger.info(f"[2단계] 메타데이터 추출 시작: 문제 {problem_id}")

            # 이미지 파일 경로 설정
            question_image_path = Path(f"resource/question/{problem_id}.png")
            commentary_image_path = Path(f"resource/commentary/{problem_id}.png")

            # 이미지 파일 존재 여부 확인
            if not question_image_path.exists():
                raise FileNotFoundError(f"문제 이미지를 찾을 수 없습니다: {question_image_path}")
            if not commentary_image_path.exists():
                raise FileNotFoundError(f"모범답안 이미지를 찾을 수 없습니다: {commentary_image_path}")

            # 이미지 파일 업로드
            question_image_part = self.gemini.client.files.upload(file=str(question_image_path))
            commentary_image_part = self.gemini.client.files.upload(file=str(commentary_image_path))

            # 프롬프트 생성
            prompt = create_metadata_extraction_prompt(
                problem_data['curriculum'],
                problem_data['achievement_standards'],
                problem_data['consideration'],
                total_points
            )

            # Gemini API 호출 콘텐츠 구성
            contents = [
                prompt,
                "아래 이미지는 실제 문제지입니다.",
                question_image_part,
                "아래 이미지는 해설 이미지입니다.",
                commentary_image_part
            ]

            # Gemini API 호출
            response_text = self.gemini.call_extract_metadata(contents)

            # 백슬래시 이스케이프 수정
            response_text_fixed = fix_json_escaping(response_text)

            # JSON 파싱
            metadata = json.loads(response_text_fixed)

            # ===== 3단계: 1단계와 2단계 결과 병합 =====
            # 2단계에서 나온 curriculum_mapping을 1단계 결과로 교체
            metadata['curriculum_mapping'] = {
                "대단원": curriculum_mapping.get('대단원', ''),
                "중단원": curriculum_mapping.get('중단원', ''),
                "소단원": curriculum_mapping.get('소단원', ''),
                "성취기준_코드": curriculum_mapping.get('성취기준_코드', '')
            }

            self.logger.info(f"✓ [2단계] 메타데이터 추출 성공: {problem_id}")
            self.logger.info(f"✓ 최종 메타데이터 생성 완료: {problem_id}")
            return metadata, True

        except (FileNotFoundError, IOError) as e:
            error_msg = f"파일 처리 실패: {e}"
            self.logger.error(error_msg)
            self.stats['errors'].append({"problem_id": problem_id, "error": error_msg})
            return None, False
        except json.JSONDecodeError as e:
            error_msg = f"JSON 파싱 실패: {e}"
            self.logger.error(error_msg)
            try:
                if 'response_text' in locals():
                    self.logger.error(f"원본 응답 내용:\n{response_text}")
                    self.logger.error(f"수정된 응답 내용:\n{response_text_fixed}")
            except Exception as log_error:
                self.logger.error(f"로그 출력 실패: {log_error}")
            self.stats['errors'].append({"problem_id": problem_id, "error": error_msg})
            return None, False
        except Exception as e:
            error_msg = f"메타데이터 추출 중 예외 발생: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.stats['errors'].append({"problem_id": problem_id, "error": error_msg})
            return None, False

    def _get_problem_data_from_mapping(self, curriculum_mapping):
        """
        1단계에서 추출한 curriculum_mapping을 바탕으로 CSV에서 상세 정보 조회

        Args:
            curriculum_mapping (dict): 1단계에서 추출한 교육과정 정보

        Returns:
            dict: problem_data (curriculum, achievement_standards, consideration)
        """
        try:
            curriculum_df = pd.read_csv('resource/curriculum.csv', encoding='utf-8-sig')
            achievement_df = pd.read_csv('resource/achievement_standards.csv', encoding='utf-8-sig')
            consideration_df = pd.read_csv('resource/consideration.csv', encoding='utf-8-sig')

            # 교육과정 정보
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
                # 여러 행이 있을 수 있으므로 첫 번째 행 사용
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
            self.logger.error(f"CSV 데이터 조회 중 오류: {e}")
            return None

    def save_metadata(self, problem_id, metadata):
        """
        메타데이터를 JSON 파일로 저장

        Args:
            problem_id (str): 문제 ID
            metadata (dict): 메타데이터
        """
        filepath = self.output_dir / f"{problem_id}_metadata.json"

        # 추가 정보 포함
        output_data = {
            "problem_id": problem_id,
            "extracted_at": datetime.now().isoformat(),
            "metadata": metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"메타데이터 저장 완료: {filepath}")

    def process_single_problem(self, problem_id, total_points=10):
        """
        단일 문제에 대한 메타데이터 추출, 저장, 통계 현황 저장
        Args:
            problem_id (str): 처리할 문제의 ID
            total_points (int): 문제의 총 배점 (기본값: 10)

        Returns:
            bool: 모든 과정 성공(True), 그렇지 않으면 (False)
        """
        # 이미 메타데이터 파일이 있으면 건너뛰기
        metadata_file = self.output_dir / f"{problem_id}_metadata.json"
        if metadata_file.exists():
            self.logger.info(f"메타데이터가 이미 존재합니다. 건너뜁니다: {problem_id}")
            return True

        self.stats['total'] += 1
        metadata, success = self.extract_single_problem(problem_id, total_points)

        if success and metadata:
            self.save_metadata(problem_id, metadata)
            self.stats['success'] += 1
            return True
        else:
            self.stats['failed'] += 1
            return False

    def extract_from_list(self, list_csv_path="resource/list.csv"):
        """
        list.csv에 명시된 문제(qid) 메타데이터 추출

        Args:
            list_csv_path (str): list.csv 파일 경로
        """

        self.logger.info("메타데이터 추출 시작")

        try:
            df = pd.read_csv(list_csv_path, encoding='utf-8-sig')
            # 고유한 문제 ID 추출
            unique_problems = df['qid'].unique()

            self.logger.info(f"총 {len(unique_problems)}개 문제의 메타데이터를 추출합니다.")

            for idx, problem_id in enumerate(unique_problems, 1):
                self.logger.info(f"\n진행률: {idx}/{len(unique_problems)}")
                self.process_single_problem(str(problem_id))

            self.print_summary()
            self.save_summary()

        except FileNotFoundError:
            self.logger.error(f"파일을 찾을 수 없습니다: {list_csv_path}")
        except Exception as e:
            self.logger.error(f"배치 처리 중 오류 발생: {e}", exc_info=True)

    def extract_from_directory(self, question_dir="resource/question"):
        """
        ./resource/question 폴더 내 문제 이미지 메타데이터 추출

        Args:
            question_dir (str): 문제 이미지 디렉토리
        """
        self.logger.info("="*80)
        self.logger.info("디렉토리 기반 메타데이터 추출 시작")
        self.logger.info("="*80)

        question_path = Path(question_dir)
        if not question_path.exists():
            self.logger.error(f"디렉토리를 찾을 수 없습니다: {question_dir}")
            return

        # PNG 파일 찾기
        problem_files = list(question_path.glob("*.png"))
        self.logger.info(f"총 {len(problem_files)}개 문제를 발견했습니다.")

        for idx, problem_file in enumerate(problem_files, 1):
            problem_id = problem_file.stem
            self.logger.info(f"\n진행률: {idx}/{len(problem_files)}")
            self.extract_and_save(problem_id)

        self.print_summary()
        self.save_summary()

    def print_summary(self):
        """ 메타데이터 처리 결과 출력"""
        self.logger.info("\n" + "="*30)
        self.logger.info("메타데이터 추출 완료 - 최종 결과")
        self.logger.info("="*30)
        self.logger.info(f"총 처리: {self.stats['total']}")
        self.logger.info(f"성공: {self.stats['success']}")
        self.logger.info(f"실패: {self.stats['failed']}")

        if self.stats['errors']:
            self.logger.info(f"\n실패한 케이스 ({len(self.stats['errors'])}개):")
            for error in self.stats['errors']:
                self.logger.error(f"  - 문제 {error['problem_id']}: {error['error']}")

    def save_summary(self):
        """메타데이터 처리 결과 JSON 파일 저장"""
        summary_file = self.output_dir / "_extraction_summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "output_directory": str(self.output_dir)
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"\n요약 정보 저장: {summary_file}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="수학 문제 메타데이터 추출 스크립트"
    )
    parser.add_argument(
        '--problem-id',
        type=str,
        help='단일 문제 ID(예: 223174)'
    )
    parser.add_argument(
        '--from-list',
        action='store_true',
        help='resource/list.csv의 모든 문제 처리'
    )
    parser.add_argument(
        '--from-directory',
        action='store_true',
        help='resource/question 디렉토리의 모든 문제 처리'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='metadata',
        help='메타데이터를 저장할 디렉토리 (기본값: metadata)'
    )
    parser.add_argument(
        '--total-points',
        type=int,
        default=10,
        help='문제의 총 배점 (기본값: 10)'
    )

    args = parser.parse_args()
    output_dir = f"{args.output_dir}"

    extractor = MetadataExtractor(output_dir=output_dir)

    # 처리 모드 선택
    if args.problem_id:
        # 단일 문제 처리
        extractor.process_single_problem(args.problem_id, args.total_points)
        extractor.print_summary()
        extractor.save_summary()
    elif args.from_list:
        # list.csv 기반 배치 처리
        extractor.extract_from_list()
    elif args.from_directory:
        # 디렉토리 기반 배치 처리
        extractor.extract_from_directory()
    else:
        print("처리 모드를 선택하세요:")
        print("  --problem-id <문제번호>  : 단일 문제 처리")
        print("  --from-list              : list.csv의 모든 문제 처리")
        print("  --from-directory         : question 디렉토리의 모든 문제 처리")
        print("\n기본값으로 list.csv 기반 처리를 시작합니다...\n")
        extractor.extract_from_list()


if __name__ == "__main__":
    main()