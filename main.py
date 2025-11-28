"""
수학 문제 자동 채점 배치 처리 스크립트

이 스크립트는 resource/list.csv에 있는 모든 학생 풀이를 순회하며:
1. 문제별로 메타데이터(채점 가이드)를 추출
2. 각 학생 풀이를 메타데이터 기준으로 채점
3. 결과를 JSON 파일로 저장
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from common.logger import init_logger
from services.problem_service import (
    get_problems_from_list,
    extract_metadata,
    analyze_student_solution,
    get_problem_data
)

# 로거 초기화
logger = init_logger()


class MathGradingProcessor:
    """수학 문제 채점 배치 프로세서"""

    def __init__(self, output_dir="results"):
        """
        Args:
            output_dir (str): 결과 파일을 저장할 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 메타데이터 캐시 (같은 문제는 한 번만 추출)
        self.metadata_cache = {}

        # 처리 통계
        self.stats = {
            "total_problems": 0,
            "total_solutions": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

    def save_metadata(self, problem_id, metadata):
        """
        메타데이터를 파일로 저장

        Args:
            problem_id (str): 문제 ID
            metadata (dict): 추출된 메타데이터
        """
        metadata_dir = self.output_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        filepath = metadata_dir / f"{problem_id}_metadata.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"메타데이터 저장 완료: {filepath}")

    def save_analysis(self, problem_id, solution_filename, analysis, student_data):
        """
        학생 풀이 분석 결과를 파일로 저장

        Args:
            problem_id (str): 문제 ID
            solution_filename (str): 학생 풀이 파일명
            analysis (dict): 분석 결과
            student_data (dict): 학생 데이터 (list.csv의 행)
        """
        analysis_dir = self.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        # 솔루션 파일명에서 확장자 제거
        solution_name = Path(solution_filename).stem
        filepath = analysis_dir / f"{problem_id}_{solution_name}_analysis.json"

        # 학생 데이터와 분석 결과를 함께 저장
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

        logger.info(f"분석 결과 저장 완료: {filepath}")

    def get_or_extract_metadata(self, problem_id):
        """
        메타데이터를 캐시에서 가져오거나 새로 추출

        Args:
            problem_id (str): 문제 ID

        Returns:
            tuple: (metadata, status_code)
        """
        # 캐시 확인
        if problem_id in self.metadata_cache:
            logger.info(f"캐시에서 메타데이터 로드: {problem_id}")
            return self.metadata_cache[problem_id], 200

        # 메타데이터 추출
        logger.info(f"메타데이터 추출 시작: {problem_id}")
        metadata, status_code = extract_metadata(problem_id)

        if status_code == 200:
            # 캐시에 저장
            self.metadata_cache[problem_id] = metadata
            # 파일로 저장
            self.save_metadata(problem_id, metadata)
            logger.info(f"메타데이터 추출 완료: {problem_id}")
        else:
            logger.error(f"메타데이터 추출 실패: {problem_id}, 상태코드: {status_code}")

        return metadata, status_code

    def process_single_solution(self, student_data):
        """
        단일 학생 풀이 처리

        Args:
            student_data (dict): list.csv의 한 행 데이터

        Returns:
            bool: 성공 여부
        """
        problem_id = str(student_data['qid'])
        solution_filename = student_data['학생풀이']

        logger.info(f"\n{'='*80}")
        logger.info(f"처리 시작 - 문제: {problem_id}, 풀이: {solution_filename}")
        logger.info(f"{'='*80}")

        try:
            # 1단계: 메타데이터 추출 (캐시 사용)
            metadata, status_code = self.get_or_extract_metadata(problem_id)
            if status_code != 200:
                error_msg = f"메타데이터 추출 실패: {problem_id}"
                logger.error(error_msg)
                self.stats['errors'].append({
                    "problem_id": problem_id,
                    "solution_file": solution_filename,
                    "error": error_msg,
                    "metadata": metadata
                })
                self.stats['failed'] += 1
                return False

            # 2단계: 학생 풀이 분석
            logger.info(f"학생 풀이 분석 시작: {solution_filename}")
            analysis, status_code = analyze_student_solution(
                problem_id,
                solution_filename,
                metadata
            )

            if status_code != 200:
                error_msg = f"풀이 분석 실패: {solution_filename}"
                logger.error(error_msg)
                self.stats['errors'].append({
                    "problem_id": problem_id,
                    "solution_file": solution_filename,
                    "error": error_msg,
                    "analysis": analysis
                })
                self.stats['failed'] += 1
                return False

            # 3단계: 결과 저장
            self.save_analysis(problem_id, solution_filename, analysis, student_data)

            # 통계 업데이트
            self.stats['success'] += 1

            logger.info(f"✓ 처리 완료: {problem_id} - {solution_filename}")
            logger.info(f"  최종 점수: {analysis.get('final_score', 'N/A')} / {analysis.get('total_possible', 'N/A')}")

            return True

        except Exception as e:
            error_msg = f"예외 발생: {str(e)}"
            logger.error(f"처리 중 예외 발생: {problem_id} - {solution_filename}")
            logger.error(error_msg, exc_info=True)
            self.stats['errors'].append({
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "error": error_msg
            })
            self.stats['failed'] += 1
            return False

    def process_all(self):
        """
        list.csv의 모든 학생 풀이 처리
        """
        logger.info("="*80)
        logger.info("수학 문제 자동 채점 배치 처리 시작")
        logger.info("="*80)

        # list.csv 로드
        problems_list = get_problems_from_list()

        if not problems_list:
            logger.error("처리할 문제가 없습니다. resource/list.csv를 확인하세요.")
            return

        self.stats['total_solutions'] = len(problems_list)

        # 고유한 문제 ID 수 계산
        unique_problems = set(str(item['qid']) for item in problems_list)
        self.stats['total_problems'] = len(unique_problems)

        logger.info(f"총 {self.stats['total_problems']}개 문제, {self.stats['total_solutions']}개 풀이 처리 예정")

        # 각 풀이 처리
        for idx, student_data in enumerate(problems_list, 1):
            logger.info(f"\n진행률: {idx}/{self.stats['total_solutions']}")
            self.process_single_solution(student_data)

        # 최종 통계 출력
        self.print_summary()

        # 통계를 파일로 저장
        self.save_summary()

    def print_summary(self):
        """처리 결과 요약 출력"""
        logger.info("\n" + "="*80)
        logger.info("처리 완료 - 최종 결과")
        logger.info("="*80)
        logger.info(f"총 문제 수: {self.stats['total_problems']}")
        logger.info(f"총 풀이 수: {self.stats['total_solutions']}")
        logger.info(f"성공: {self.stats['success']}")
        logger.info(f"실패: {self.stats['failed']}")

        if self.stats['errors']:
            logger.info(f"\n실패한 케이스 ({len(self.stats['errors'])}개):")
            for error in self.stats['errors']:
                logger.error(f"  - 문제 {error['problem_id']}, 파일 {error['solution_file']}: {error['error']}")

    def save_summary(self):
        """처리 결과 요약을 JSON 파일로 저장"""
        summary_file = self.output_dir / "summary.json"
        summary = {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "output_directory": str(self.output_dir)
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"\n요약 정보 저장: {summary_file}")


def main():
    """메인 실행 함수"""
    # 결과 디렉토리 설정 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/batch_{timestamp}"

    # 프로세서 생성 및 실행
    processor = MathGradingProcessor(output_dir=output_dir)
    processor.process_all()


if __name__ == "__main__":
    main()