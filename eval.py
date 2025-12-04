import argparse
import json
import os
from glob import glob

def display_file(path):
    """지정된 경로의 파일 내용을 출력합니다."""
    if not os.path.exists(path):
        print(f"오류: 파일 또는 디렉토리를 찾을 수 없습니다 '{path}'")
        return

    if os.path.isdir(path):
        print(f"'{path}'는 디렉토리입니다. 파일 경로를 지정해주세요.")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            # JSON 파일인 경우 예쁘게 출력
            if path.endswith('.json'):
                try:
                    data = json.load(f)
                    print(json.dumps(data, indent=4, ensure_ascii=False))
                except json.JSONDecodeError:
                    print("오류: JSON 형식이 올바르지 않습니다.")
                    # JSON 파싱 실패 시 일반 텍스트로 출력
                    f.seek(0)
                    print(f.read())
            else:
                print(f.read())
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")

def find_latest_result_dir():
    """results 디렉토리에서 가장 최근에 생성된 하위 디렉토리를 찾습니다."""
    result_dirs = glob("results/batch_*")
    if not result_dirs:
        return None
    return max(result_dirs, key=os.path.getmtime)

def main():
    """
    메인 함수: 인자를 파싱하여 metadata 또는 result 파일의 내용을 출력합니다.
    """
    parser = argparse.ArgumentParser(description="Display metadata or result files.")
    parser.add_argument("type", choices=['metadata', 'result'], help="출력할 파일 타입: 'metadata' 또는 'result'")
    parser.add_argument("filename", help="출력할 파일 이름. 'result' 타입의 경우, 분석 파일 이름을 지정합니다.")
    parser.add_argument("--dir", help="결과 디렉토리 경로 (예: results/batch_20251202_164454). 지정하지 않으면 가장 최근 디렉토리를 사용합니다.")

    args = parser.parse_args()

    if args.type == 'metadata':
        file_path = os.path.join('metadata', args.filename)
        display_file(file_path)
    elif args.type == 'result':
        result_dir = args.dir or find_latest_result_dir()
        if not result_dir:
            print("오류: 'results' 디렉토리를 찾을 수 없거나 비어있습니다.")
            return
        
        file_path = os.path.join(result_dir, 'analysis', args.filename)
        if not os.path.exists(file_path) and args.filename == 'summary.json':
             file_path = os.path.join(result_dir, args.filename)

        display_file(file_path)

if __name__ == "__main__":
    main()
