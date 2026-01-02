"""
수학 문제 자동 채점 메인 스크립트

'resource/list.csv'에 명시된 모든 학생 풀이를 순회하며 자동 채점을 실행합니다.
전체 프로세스는 다음과 같습니다.
1. 각 문제별 채점 기준(메타데이터)이 존재한다는 전제하에 시작합니다.
2. 해당 문제의 메타데이터를 기반으로 학생의 풀이 이미지를 분석하고 채점합니다.
3. 채점 결과는 JSON 파일 형식으로 저장합니다.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from common.logger import init_logger
from services.problem_service import get_problems_from_list, analyze_student_solution as service_analyze


class MathGradingProcessor:
    """
    수학 문제 자동 채점 프로세서 클래스
    """

    def __init__(self, output_dir="results", metadata_dir="metadata", ocr_engine_name: str = 'latex_ocr'):
        """
        Args:
            output_dir (str): 채점 결과 파일을 저장할 최상위 디렉토리
            metadata_dir (str): 사전에 추출된 문제별 메타데이터 디렉토리 (필수)
            ocr_engine_name (str): 사용할 OCR 엔진 이름

        Raises:
            FileNotFoundError: metadata_dir이 존재하지 않을 경우
            ValueError: metadata_dir에 메타데이터 파일이 하나도 없을 경우
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = Path(metadata_dir)
        self.ocr_engine_name = ocr_engine_name
        self.logger = init_logger()
        self.logger.info(f"선택된 OCR 엔진: {self.ocr_engine_name}")

        # 메타데이터 디렉토리 존재 여부 확인 (필수)
        if not self.metadata_dir.exists():
            raise FileNotFoundError(
                f"메타데이터 디렉토리를 찾을 수 없습니다: {self.metadata_dir}\n"
                f"메타데이터를 먼저 추출해주세요. (extract_metadata.py 실행)"
            )

        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.stats: Dict[str, Any] = {
            "total_problems": 0,
            "total_solutions": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

        # 메타데이터 로드 (필수)
        self._load_existing_metadata()

        # 메타데이터가 하나도 없으면 에러
        if not self.metadata_cache:
            raise ValueError(
                f"메타데이터 디렉토리에 메타데이터 파일이 없습니다: {self.metadata_dir}\n"
                f"메타데이터를 먼저 추출해주세요. (extract_metadata.py 실행)"
            )

    def _load_existing_metadata(self) -> None:
        """
        지정된 디렉토리에서 기존 메타데이터 파일들을 로드하여 캐시에 저장합니다.
        """
        self.logger.info(f"기존 메타데이터 로드 중: {self.metadata_dir}")
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    problem_id = data.get('problem_id', metadata_file.stem.replace('_metadata', ''))
                    self.metadata_cache[problem_id] = data.get('metadata', data)
                self.logger.info(f"  ✓ 로드 완료: {problem_id}")
            except Exception as e:
                self.logger.warning(f"  ✗ 메타데이터 로드 실패: {metadata_file} - {e}")

    def get_metadata(self, problem_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        메타데이터를 캐시에서 가져옵니다.

        초기화 시 로드된 메타데이터 캐시에서 문제 ID에 해당하는 메타데이터를 조회합니다.
        메타데이터는 사전에 extract_metadata.py를 통해 추출되어 있어야 합니다.

        Args:
            problem_id (str): 조회할 문제의 고유 ID.

        Returns:
            Tuple[Optional[Dict[str, Any]], bool]:
                - 성공 시: (메타데이터 딕셔너리, True)
                - 실패 시: (None, False)
        """
        if problem_id in self.metadata_cache:
            self.logger.info(f"캐시에서 메타데이터 로드: {problem_id}")
            return self.metadata_cache[problem_id], True

        self.logger.error(
            f"메타데이터를 찾을 수 없습니다: {problem_id}\n"
            f"해당 문제의 메타데이터를 먼저 추출해주세요. "
            f"(extract_metadata.py --problem-id {problem_id})"
        )
        return None, False

    def save_analysis(self, problem_id: str, solution_filename: str, analysis: Dict[str, Any], student_data: Dict[str, Any]) -> None:
        """
        학생 풀이에 대한 최종 분석 결과를 JSON 파일로 저장합니다.

        Args:
            problem_id (str): 문제의 고유 ID.
            solution_filename (str): 학생 풀이 이미지 파일명.
            analysis (Dict[str, Any]): `analyze_student_solution`으로부터 받은 분석 결과.
            student_data (Dict[str, Any]): 원본 CSV 파일의 학생 관련 데이터 행.
        """
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        solution_name = Path(solution_filename).stem
        filepath = analysis_dir / f"{problem_id}_{solution_name}_analysis.json"

        result = {
            "problem_id": problem_id,
            "solution_file": solution_filename,
            "student_answer": student_data.get('학생답안', ''),
            "expected_result": student_data.get('정답유무', ''),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        self.logger.info(f"분석 결과 저장 완료: {filepath}")

    def process_single_solution(self, student_data: Dict[str, Any]) -> bool:
        """
        단일 학생 풀이 한 건에 대한 전체 채점 과정을 처리합니다.

        이 메소드는 다음 단계를 순차적으로 실행합니다.
        1. 문제의 메타데이터를 캐시에서 가져옵니다.
        2. 획득한 메타데이터를 이용해 학생 풀이를 분석합니다.
        3. 분석 결과를 파일로 저장하고 통계를 업데이트합니다.

        Args:
            student_data (Dict[str, Any]): 'list.csv'에서 읽어온 학생 풀이 한 행의 데이터.

        Returns:
            bool: 모든 과정이 성공하면 True, 하나라도 실패하면 False.
        """
        problem_id = str(student_data['qid'])
        solution_filename = student_data['학생풀이']

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"처리 시작 - 문제: {problem_id}, 풀이: {solution_filename}")
        self.logger.info(f"{'='*80}")

        try:
            # 1단계: 메타데이터 조회 (캐시에서)
            metadata, success = self.get_metadata(problem_id)
            if not success or not metadata:
                raise ValueError(f"메타데이터를 찾을 수 없습니다: {problem_id}")

            # 2단계: 학생 풀이 분석 (서비스 모듈 사용)
            self.logger.info(f"학생 풀이 분석 시작: {solution_filename}")
            analysis, success = service_analyze(
                problem_id,
                solution_filename,
                metadata,
                ocr_engine_name=self.ocr_engine_name
            )
            if not success or not analysis:
                raise ValueError(f"풀이 분석 실패: {problem_id} - {solution_filename}")

            # 3단계: 결과 저장
            self.save_analysis(problem_id, solution_filename, analysis, student_data)

            self.stats['success'] += 1
            self.logger.info(f"✓ 처리 완료: {problem_id} - {solution_filename}")
            self.logger.info(f"  최종 점수: {analysis.get('final_score', 'N/A')} / {analysis.get('total_possible', 'N/A')}")
            return True

        except Exception as e:
            error_msg = f"예외 발생: {str(e)}"
            self.logger.error(f"처리 중 예외 발생: {problem_id} - {solution_filename}")
            self.logger.error(error_msg, exc_info=True)
            self.stats['errors'].append({
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "error": error_msg
            })
            self.stats['failed'] += 1
            return False

    def run(self) -> None:
        """
        'list.csv'에 명시된 모든 학생 풀이에 대해 자동 채점 배치를 실행합니다.
        """
        self.logger.info("=" * 80)
        self.logger.info("수학 문제 자동 채점 배치 처리 시작")
        self.logger.info("=" * 80)

        problems_list = get_problems_from_list()
        if not problems_list:
            self.logger.error("처리할 문제가 없습니다. 'resource/list.csv'를 확인하세요.")
            return

        self.stats['total_solutions'] = len(problems_list)
        unique_problems = set(str(item['qid']) for item in problems_list)
        self.stats['total_problems'] = len(unique_problems)

        self.logger.info(f"총 {self.stats['total_problems']}개 문제, {self.stats['total_solutions']}개 풀이 처리 예정")

        for idx, student_data in enumerate(problems_list, 1):
            self.logger.info(f"\n풀이 진행률: {idx}/{self.stats['total_solutions']}")
            self.process_single_solution(student_data)

        self.print_summary()
        self.save_summary()

    def print_summary(self) -> None:
        """
        배치 처리 완료 후 최종 통계 요약을 콘솔에 출력합니다.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("배치 처리 완료 - 최종 결과")
        self.logger.info("=" * 80)
        self.logger.info(f"총 문제 수: {self.stats['total_problems']}")
        self.logger.info(f"총 풀이 수: {self.stats['total_solutions']}")
        self.logger.info(f"성공: {self.stats['success']}")
        self.logger.info(f"실패: {self.stats['failed']}")

        if self.stats['errors']:
            self.logger.info(f"\n오류 발생 내역 ({len(self.stats['errors'])}건):")
            for error in self.stats['errors']:
                self.logger.error(f"  - 문제 {error['problem_id']}, 파일 {error['solution_file']}: {error['error']}")

    def save_summary(self) -> None:
        """
        최종 통계 요약을 'summary.json' 파일로 저장합니다.
        """
        summary_file = self.output_dir / "summary.json"
        summary = {
            "summary_generated_at": datetime.now().isoformat(),
            "statistics": self.stats,
            "output_directory": str(self.output_dir)
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(f"\n요약 정보 저장 완료: {summary_file}")


def main() -> None:
    """
    자동 채점 스크립트의 메인 실행 함수
    """
    parser = argparse.ArgumentParser(description="수학 문제 자동 채점 시스템")
    parser.add_argument(
        '--ocr-engine',
        type=str,
        choices=['latex_ocr', 'nougat_latex'],
        default='nougat_latex',
        help="LaTeX 추출에 사용할 OCR 엔진 선택"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/batch_{timestamp}"
    metadata_dir = "metadata"

    logger = init_logger()
    logger.info("=" * 80)
    logger.info("수학 문제 자동 채점 시스템 시작")
    logger.info("=" * 80)
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"메타데이터 디렉토리: {metadata_dir}")

    try:
        processor = MathGradingProcessor(
            output_dir=output_dir,
            metadata_dir=metadata_dir,
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