"""
통합 Vision 기반 정밀 채점 스크립트 (main_update.py)

main.py의 모든 기능 +
Google Vision API + Gemini를 결합한 고급 분석:
1. OCR + bbox 추출
2. visualized 이미지 생성 (bbox ID 표시)
3. 오류 위치 핀포인트 분석
4. 스텝별 LaTeX 검증 분석

사용법:
    python main_update.py
    python main_update.py --output results/my_custom_batch
    python main_update.py --ocr-engine google
"""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from common.logger import init_logger
from common.gemini import Gemini
from common.prompt import create_vision_analysis_prompt, create_step_validation_prompt
from utils.ocr_visualizer import visualize_bboxes_with_ids
from utils.utils import fix_json_escaping
from services.problem_service import get_problems_from_list
from services.ocr_service import get_ocr_engine


class IntegratedVisionGradingProcessor:
    """
    통합 Vision 기반 채점 프로세서

    main.py의 기능 + Vision 기반 고급 분석을 통합한 프로세서
    """

    def __init__(self, output_dir_name: str = None, metadata_dir: str = "metadata", ocr_engine_name: str = "google"):
        """
        Args:
            output_dir_name (str): 출력 디렉토리 이름 (선택)
            metadata_dir (str): 메타데이터 디렉토리
            ocr_engine_name (str): 사용할 OCR 엔진 ('google', 'nougat_latex', 'latex_ocr')
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir_name:
            self.output_base = Path(output_dir_name)
        else:
            self.output_base = Path(f"results/new/batch_{self.timestamp}")

        # 디렉토리 구조 설정
        self.dirs = {
            "root": self.output_base,
            "analysis": self.output_base / "analysis",
            "bbox": self.output_base / "bbox",
            "visualized": self.output_base / "visualized"
        }

        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = Path(metadata_dir)
        self.logger = init_logger()
        self.gemini = Gemini()
        self.ocr_engine_name = ocr_engine_name

        # OCR 엔진 초기화
        self._initialize_ocr_engine()

        # 메타데이터 캐시 및 통계
        self.metadata_cache = {}
        self.stats = {
            "total_problems": 0,
            "total_solutions": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

        # 메타데이터 로드 (필수)
        self._load_metadata()

        # 메타데이터가 하나도 없으면 에러
        if not self.metadata_cache:
            raise ValueError(
                f"메타데이터 디렉토리에 메타데이터 파일이 없습니다: {self.metadata_dir}\n"
                f"메타데이터를 먼저 추출해주세요. (extract_metadata.py 실행)"
            )

    def _initialize_ocr_engine(self):
        """OCR 엔진 초기화"""
        self.logger.info(f"OCR 엔진 초기화: {self.ocr_engine_name}")

        try:
            self.ocr_engine = get_ocr_engine(self.ocr_engine_name)
            self.logger.info(f"✓ OCR 엔진 초기화 성공: {self.ocr_engine_name}")
        except Exception as e:
            self.logger.error(f"OCR 엔진 초기화 실패: {self.ocr_engine_name} - {e}")

            # google 실패 시 대체 엔진 시도
            if self.ocr_engine_name == "google":
                self.logger.info("대체 엔진(nougat_latex)으로 전환을 시도합니다.")
                try:
                    self.ocr_engine = get_ocr_engine("nougat_latex")
                    self.ocr_engine_name = "nougat_latex"
                    self.logger.info("✓ 대체 엔진 초기화 성공: nougat_latex")
                except Exception as e2:
                    self.logger.error(f"대체 엔진 초기화 실패: {e2}")
                    self.logger.warning("Mock 엔진을 사용합니다.")
                    self.ocr_engine = get_ocr_engine("mock")
                    self.ocr_engine_name = "mock"
            else:
                raise

    def _load_metadata(self):
        """메타데이터 로드"""
        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"메타데이터 디렉토리를 찾을 수 없습니다: {self.metadata_dir}\n"
                f"메타데이터를 먼저 추출해주세요. (extract_metadata.py 실행)"
            )

        self.logger.info(f"메타데이터 로드 중: {self.metadata_dir}")
        for f in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    pid = data.get('problem_id', f.stem.replace('_metadata', ''))
                    self.metadata_cache[pid] = data.get('metadata', data)
                self.logger.info(f"  ✓ 로드 완료: {pid}")
            except Exception as e:
                self.logger.warning(f"  ✗ 메타데이터 로드 실패: {f} - {e}")

    def process_solution(self, student_data: Dict[str, Any]) -> bool:
        """
        단일 학생 풀이 처리

        1. OCR + bbox 추출 (Google Vision 또는 대체 엔진)
        2. visualized 이미지 생성
        3. 오류 위치 핀포인트 분석
        4. 스텝별 LaTeX 검증 분석
        5. 모든 결과 통합 저장

        Args:
            student_data (Dict[str, Any]): 학생 풀이 데이터

        Returns:
            bool: 성공 여부
        """
        problem_id = str(student_data['qid'])
        solution_filename = student_data['학생풀이']
        solution_path = os.path.join("resource/solve", solution_filename)

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"처리 시작 - 문제: {problem_id}, 풀이: {solution_filename}")
        self.logger.info(f"{'='*80}")

        # 파일 존재 확인
        if not os.path.exists(solution_path):
            self.logger.error(f"파일 없음: {solution_path}")
            self.stats['failed'] += 1
            self.stats['errors'].append({
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "error": "파일을 찾을 수 없음"
            })
            return False

        # 메타데이터 확인
        if problem_id not in self.metadata_cache:
            self.logger.error(f"메타데이터 없음: {problem_id}")
            self.stats['failed'] += 1
            self.stats['errors'].append({
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "error": "메타데이터를 찾을 수 없음"
            })
            return False

        try:
            metadata = self.metadata_cache[problem_id]
            solution_stem = Path(solution_filename).stem

            # ========================================
            # 1단계: OCR + bbox 추출
            # ========================================
            self.logger.info("1단계: OCR + bbox 추출 중...")
            ocr_results, raw_response = self._extract_ocr_and_bbox(solution_path)

            if not ocr_results:
                raise ValueError("OCR 결과가 비어있습니다")

            # bbox 정보 저장
            bbox_path = self.dirs["bbox"] / f"{problem_id}_{solution_stem}_bbox.json"
            with open(bbox_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "annotations": ocr_results,
                    "raw_response": raw_response
                }, f, ensure_ascii=False, indent=2)
            self.logger.info(f"  ✓ bbox 저장: {bbox_path}")

            # ========================================
            # 2단계: Visualized 이미지 생성
            # ========================================
            self.logger.info("2단계: Visualized 이미지 생성 중...")
            vis_image_path = self.dirs["visualized"] / solution_filename
            vis_map_path = self.dirs["visualized"] / f"{problem_id}_{solution_stem}_map.json"

            id_map = visualize_bboxes_with_ids(solution_path, ocr_results, str(vis_image_path))

            with open(vis_map_path, 'w', encoding='utf-8') as f:
                json.dump(id_map, f, ensure_ascii=False, indent=2)
            self.logger.info(f"  ✓ 시각화 이미지: {vis_image_path}")
            self.logger.info(f"  ✓ ID 매핑: {vis_map_path}")

            # ========================================
            # 3단계: 오류 위치 핀포인트 분석
            # ========================================
            self.logger.info("3단계: 오류 위치 핀포인트 분석 중...")
            vision_analysis = self._analyze_error_location(str(vis_image_path), metadata, id_map)
            self.logger.info("  ✓ 오류 위치 분석 완료")

            # ========================================
            # 4단계: 스텝별 LaTeX 검증 분석
            # ========================================
            self.logger.info("4단계: 스텝별 LaTeX 검증 분석 중...")
            step_validation = self._validate_steps_with_latex(str(vis_image_path), metadata, id_map)
            self.logger.info("  ✓ 스텝별 검증 완료")

            # ========================================
            # 5단계: 통합 결과 저장
            # ========================================
            self.logger.info("5단계: 통합 결과 저장 중...")
            analysis_path = self.dirs["analysis"] / f"{problem_id}_{solution_stem}_analysis.json"

            # 최종 점수는 step_validation에서만 가져옴
            final_score = step_validation.get('final_score', 'N/A')
            total_possible = step_validation.get('total_possible', 'N/A')

            final_output = {
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "student_answer": student_data.get('학생답안', ''),
                "expected_result": student_data.get('정답유무', ''),
                "ocr_engine": self.ocr_engine_name,
                "vision_analysis": vision_analysis,  # 오류 위치 핀포인트 (점수 제외)
                "step_validation": step_validation,  # 실제 채점 + 점수
                "final_score": final_score,  # step_validation의 점수 사용
                "total_possible": total_possible,
                "timestamp": datetime.now().isoformat()
            }

            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

            self.logger.info(f"✓ 처리 완료: {analysis_path}")

            # 간단한 요약 출력
            self.logger.info(f"  최종 점수: {final_score} / {total_possible}")

            if "first_error_location" in vision_analysis:
                error_info = vision_analysis["first_error_location"]
                if error_info.get("has_error"):
                    self.logger.info(f"  첫 번째 오류: Step {error_info.get('error_step_number')}, Box ID {error_info.get('error_box_id')}")

            self.stats['success'] += 1
            return True

        except Exception as e:
            error_msg = f"처리 중 예외 발생: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats['errors'].append({
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "error": error_msg
            })
            self.stats['failed'] += 1
            return False

    def _extract_ocr_and_bbox(self, image_path: str) -> Tuple[list, dict]:
        """
        OCR + bbox 추출

        Args:
            image_path (str): 이미지 경로

        Returns:
            Tuple[list, dict]: (ocr_results, raw_response)
        """
        if self.ocr_engine is None:
            raise RuntimeError("OCR 엔진이 초기화되지 않았습니다.")

        # Google Vision 엔진인 경우 bbox 추출
        if hasattr(self.ocr_engine, 'extract_text_with_bboxes'):
            ocr_results, raw_response = self.ocr_engine.extract_text_with_bboxes(
                image_path,
                levels=['block']
            )
        else:
            # 다른 엔진은 기본 LaTeX 추출
            self.logger.warning("bbox 추출을 지원하지 않는 엔진입니다. 기본 추출 모드로 전환합니다.")
            ocr_results = self.ocr_engine.extract_latex(image_path)
            raw_response = {"info": "Extracted using non-google engine"}

        return ocr_results, raw_response

    def _analyze_error_location(self, vis_image_path: str, metadata: dict, id_map: list) -> dict:
        """
        오류 위치 핀포인트 분석

        Args:
            vis_image_path (str): visualized 이미지 경로
            metadata (dict): 메타데이터
            id_map (list): bbox ID 매핑

        Returns:
            dict: 오류 위치 분석 결과
        """
        prompt = create_vision_analysis_prompt(metadata, id_map)

        response_text = self.gemini.call_gemini_image_text(
            prompt=prompt,
            image=vis_image_path
        )

        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            cleaned_text = fix_json_escaping(cleaned_text)
            analysis_result = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패 (오류 위치 분석): {e}")
            analysis_result = {
                "raw_response": response_text,
                "error": f"JSON parsing failed: {e}"
            }

        return analysis_result

    def _validate_steps_with_latex(self, vis_image_path: str, metadata: dict, id_map: list) -> dict:
        """
        스텝별 LaTeX 검증 분석

        Args:
            vis_image_path (str): visualized 이미지 경로
            metadata (dict): 메타데이터
            id_map (list): bbox ID 매핑

        Returns:
            dict: 스텝별 검증 결과
        """
        prompt = create_step_validation_prompt(metadata, id_map)

        response_text = self.gemini.call_gemini_image_text(
            prompt=prompt,
            image=vis_image_path
        )

        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            cleaned_text = fix_json_escaping(cleaned_text)
            validation_result = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON 파싱 실패 (스텝별 검증): {e}")
            validation_result = {
                "raw_response": response_text,
                "error": f"JSON parsing failed: {e}"
            }

        return validation_result

    def run(self):
        """전체 배치 처리 실행"""
        self.logger.info("="*80)
        self.logger.info("통합 Vision 기반 채점 배치 처리 시작")
        self.logger.info("="*80)

        problems = get_problems_from_list()
        if not problems:
            self.logger.error("처리할 문제가 없습니다. 'resource/list.csv'를 확인하세요.")
            return

        self.stats['total_solutions'] = len(problems)
        unique_problems = set(str(item['qid']) for item in problems)
        self.stats['total_problems'] = len(unique_problems)

        self.logger.info(f"총 {self.stats['total_problems']}개 문제, {self.stats['total_solutions']}개 풀이 처리 예정")
        self.logger.info(f"OCR 엔진: {self.ocr_engine_name}")

        for idx, student_data in enumerate(problems, 1):
            self.logger.info(f"\n풀이 진행률: {idx}/{self.stats['total_solutions']}")
            self.process_solution(student_data)

        self._save_summary()
        self._print_summary()

    def _save_summary(self):
        """요약 정보 저장"""
        summary_path = self.dirs["root"] / "summary.json"
        summary = {
            "summary_generated_at": datetime.now().isoformat(),
            "ocr_engine": self.ocr_engine_name,
            "statistics": self.stats,
            "output_directory": str(self.output_base)
        }

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"\n요약 정보 저장 완료: {summary_path}")

    def _print_summary(self):
        """최종 통계 출력"""
        self.logger.info("\n" + "="*80)
        self.logger.info("배치 처리 완료 - 최종 결과")
        self.logger.info("="*80)
        self.logger.info(f"OCR 엔진: {self.ocr_engine_name}")
        self.logger.info(f"총 문제 수: {self.stats['total_problems']}")
        self.logger.info(f"총 풀이 수: {self.stats['total_solutions']}")
        self.logger.info(f"성공: {self.stats['success']}")
        self.logger.info(f"실패: {self.stats['failed']}")

        if self.stats['errors']:
            self.logger.info(f"\n오류 발생 내역 ({len(self.stats['errors'])}건):")
            for error in self.stats['errors']:
                self.logger.error(
                    f"  - 문제 {error['problem_id']}, "
                    f"파일 {error['solution_file']}: {error['error']}"
                )


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="통합 Vision 기반 수학 문제 자동 채점 시스템"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="출력 디렉토리 이름 (선택, 기본값: results/new/batch_<timestamp>)"
    )
    parser.add_argument(
        '--metadata-dir',
        type=str,
        default='metadata',
        help="메타데이터 디렉토리 경로 (기본값: metadata)"
    )
    parser.add_argument(
        '--ocr-engine',
        type=str,
        choices=['google', 'nougat_latex', 'latex_ocr', 'mock'],
        default='google',
        help="사용할 OCR 엔진 (기본값: google)"
    )

    args = parser.parse_args()

    logger = init_logger()
    logger.info("="*80)
    logger.info("통합 Vision 기반 수학 문제 자동 채점 시스템")
    logger.info("="*80)
    logger.info(f"출력 디렉토리: {args.output if args.output else 'results/new/batch_<timestamp>'}")
    logger.info(f"메타데이터 디렉토리: {args.metadata_dir}")
    logger.info(f"OCR 엔진: {args.ocr_engine}")

    try:
        processor = IntegratedVisionGradingProcessor(
            output_dir_name=args.output,
            metadata_dir=args.metadata_dir,
            ocr_engine_name=args.ocr_engine
        )
        processor.run()
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"초기화 실패: {e}")
        logger.error("프로그램을 종료합니다.")
        return
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
        logger.error("프로그램을 종료합니다.")
        return


if __name__ == "__main__":
    main()
