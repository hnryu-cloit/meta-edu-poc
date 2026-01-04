
"""
Vision 기반 정밀 채점 스크립트 (main_update.py)

Google Vision API와 Gemini를 결합하여,
풀이 과정의 오류 위치를 정확히 핀포인트(UX) 고급 분석을 수행합니다.

사용법:
    python main_update.py
    python main_update.py --output results/my_custom_batch
"""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from common.logger import init_logger
from common.gemini import Gemini
from common.prompt import create_vision_analysis_prompt
from utils.ocr_visualizer import visualize_bboxes_with_ids
from utils.utils import fix_json_escaping
from services.problem_service import get_problems_from_list
from services.ocr_service import get_ocr_engine

class VisionGradingProcessor:
    def __init__(self, output_dir_name: str = None, metadata_dir="metadata"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir_name:
             self.output_base = Path(output_dir_name)
        else:
             self.output_base = Path(f"results/batch_{self.timestamp}")
        
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
        
        try:
            self.ocr_engine = get_ocr_engine("google")
        except Exception as e:
            self.logger.error("Google Cloud Vision Engine 초기화 실패: {e}")
            self.logger.info("대체 엔진(nougat_latex)으로 전환을 시도합니다.")
            try:
                self.ocr_engine = get_ocr_engine("nougat_latex")
            except Exception as e2:
                self.logger.error(f"대체 엔진(nougat_latex) 초기화 실패: {e2}")
                self.logger.warning("모든 OCR 엔진 초기화에 실패했습니다. Mock(더미) 엔진을 사용합니다.")
                self.ocr_engine = get_ocr_engine("mock")

        self.metadata_cache = {}
        self._load_metadata()
        
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "errors": []
        }

    def _load_metadata(self):
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"메타데이터 디렉토리 없음: {self.metadata_dir}")
            
        for f in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    pid = data.get('problem_id', f.stem.replace('_metadata', ''))
                    self.metadata_cache[pid] = data.get('metadata', data)
            except Exception as e:
                self.logger.warning(f"메타데이터 로드 실패 {f}: {e}")

    def process_solution(self, student_data: Dict[str, Any]):
        problem_id = str(student_data['qid'])
        solution_filename = student_data['학생풀이']
        solution_path = os.path.join("resource/solve", solution_filename)
        
        self.logger.info(f"처리 중: {problem_id} - {solution_filename}")
        
        if not os.path.exists(solution_path):
            self.logger.error(f"파일 없음: {solution_path}")
            self.stats['failed'] += 1
            return

        if problem_id not in self.metadata_cache:
            self.logger.error(f"메타데이터 없음: {problem_id}")
            self.stats['failed'] += 1
            return

        try:
            # 1. OCR (Google Vision)
            if self.ocr_engine is None:
                raise RuntimeError("OCR 엔진이 초기화되지 않았습니다. Google Vision 설정을 확인하거나 대체 엔진을 구성하세요.")

            if hasattr(self.ocr_engine, 'extract_text_with_bboxes'):
                ocr_results, raw_response = self.ocr_engine.extract_text_with_bboxes(solution_path, levels=['block'])
            else:
                self.logger.warning("Google Vision 엔진을 사용할 수 없어 기본 추출 모드로 전환합니다.")
                ocr_results = self.ocr_engine.extract_latex(solution_path)
                raw_response = {"info": "Extracted using non-google engine"}

                self.ocr_engine = get_ocr_engine("mock")

            # bbox 및 원본 응답 저장
            if not ocr_results:
                self.logger.warning("OCR 결과가 비어있습니다. Mock 엔진으로 전환하여 재시도합니다.")
                self.ocr_engine = get_ocr_engine("mock")
                if hasattr(self.ocr_engine, 'extract_text_with_bboxes'):
                    ocr_results, raw_response = self.ocr_engine.extract_text_with_bboxes(solution_path, levels=['block'])
                else:
                    ocr_results = self.ocr_engine.extract_latex(solution_path)
                    raw_response = {"info": "mock_data"}

            if not ocr_results:
                self.logger.error("OCR 결과가 여전히 비어있습니다. 처리를 중단합니다.")
                self.stats['failed'] += 1
                return

            bbox_path = self.dirs["bbox"] / f"{problem_id}_{Path(solution_filename).stem}.json"
            full_output = {
                "annotations": ocr_results,
                "raw_response": raw_response
            }
            with open(bbox_path, 'w', encoding='utf-8') as f:
                json.dump(full_output, f, ensure_ascii=False, indent=2)
            
            # 2. Visualization
            vis_image_path = self.dirs["visualized"] / solution_filename
            vis_map_path = self.dirs["visualized"] / f"{problem_id}_{Path(solution_filename).stem}_map.json"
            
            # 공통 visualizer 모듈 사용
            id_map = visualize_bboxes_with_ids(solution_path, ocr_results, str(vis_image_path))
            
            # 매핑 정보 저장
            with open(vis_map_path, 'w', encoding='utf-8') as f:
                json.dump(id_map, f, ensure_ascii=False, indent=2)

            # 3. Gemini Analysis
            prompt = create_vision_analysis_prompt(self.metadata_cache[problem_id], id_map)
            
            response_text = self.gemini.call_gemini_image_text(
                image=str(vis_image_path),
                prompt=prompt
            )
            
            try:
                # 마크다운 코드 블록 제거 및 공백 제거
                cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
                # JSON 이스케이프 수정 (LaTeX 백슬래시 등 처리)
                cleaned_text = fix_json_escaping(cleaned_text)
                
                analysis_result = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON 파싱 실패: {e}")
                self.logger.warning("원본 텍스트를 저장합니다.")
                analysis_result = {"raw_response": response_text, "error": f"JSON parsing failed: {e}"}

            # 분석 결과 저장
            analysis_path = self.dirs["analysis"] / f"{problem_id}_{Path(solution_filename).stem}_analysis.json"
            final_output = {
                "problem_id": problem_id,
                "solution_file": solution_filename,
                "student_data": student_data,
                "analysis": analysis_result,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"완료: {analysis_path}")
            self.stats['success'] += 1

        except Exception as e:
            self.logger.error(f"오류 발생: {e}", exc_info=True)
            self.stats['errors'].append(f"{solution_filename}: {str(e)}")
            self.stats['failed'] += 1

    def run(self):
        problems = get_problems_from_list()
        self.stats['total'] = len(problems)
        
        self.logger.info(f"Vision 기반 채점 시작: 총 {len(problems)}건")
        
        for p in problems:
            self.process_solution(p)
            
        self._save_summary()

    def _save_summary(self):
        summary_path = self.dirs["root"] / "summary.json"
        summary = {
            "timestamp": self.timestamp,
            "stats": self.stats,
            "output_directory": str(self.output_base)
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.logger.info(f"요약 저장 완료: {summary_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="출력 디렉토리 이름 (선택)")
    args = parser.parse_args()
    
    processor = VisionGradingProcessor(args.output)
    processor.run()

if __name__ == "__main__":
    main()
