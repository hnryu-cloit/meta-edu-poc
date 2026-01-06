import streamlit as st
import json
import os
from glob import glob
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="수학 채점 시스템 (개선판)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown(u"""
<style>
    /* 메인 헤더 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 28px;
        font-size: 2rem;
        font-weight: 700;
    }

    /* 섹션 헤더 크기 조정 */
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    h4 {
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }

    /* 메트릭 카드 */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }

    /* 배지 스타일 */
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .danger-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }

    /* 사이드바 텍스트 크기 */
    .css-1544g2n {
        font-size: 1rem;
    }
</style>
    """, unsafe_allow_html=True)


def find_latest_result_dir():
    """가장 최근 결과 디렉토리 찾기"""
    result_dirs = glob("results/new/batch_*")
    if not result_dirs:
        return None
    return max(result_dirs, key=os.path.getmtime)


def load_all_analyses(result_dir):
    """모든 분석 결과 로드"""
    analysis_dir = os.path.join(result_dir, 'analysis')
    if not os.path.exists(analysis_dir):
        return []

    analyses = []
    for file in glob(os.path.join(analysis_dir, '*_analysis.json')):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                analyses.append(data)
        except Exception as e:
            st.warning(f"파일 로드 실패: {file}")

    return analyses

def load_metadata(problem_id):
    """특정 문제의 메타데이터 로드"""
    metadata_file = f"metadata/{problem_id}_metadata.json"
    if not os.path.exists(metadata_file):
        return None

    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def get_available_batches():
    """사용 가능한 배치 목록 가져오기"""
    result_dirs = glob("results/new/batch_*")
    return sorted(result_dirs, key=os.path.getmtime, reverse=True)

def get_available_metadata():
    """사용 가능한 메타데이터 파일 목록"""
    metadata_files = glob("metadata/*_metadata.json")
    return sorted([os.path.basename(f) for f in metadata_files])

def safe_int(value, default=0):
    """정수 변환 함수"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


with st.sidebar:
    st.markdown("### 🎓 수학 채점 시스템 (개선판)")
    st.markdown("---")

    # 배치 선택
    st.markdown("### 📂 폴더 선택")
    batches = get_available_batches()
    if batches:
        selected_batch = st.selectbox(
            "결과 디렉토리",
            batches,
            format_func=lambda x: f"{os.path.basename(x)} {'(최신)' if x == batches[0] else ''}",
            label_visibility="collapsed"
        )
    else:
        st.error("결과 디렉토리가 없습니다.")
        selected_batch = None

    st.markdown("---")

    # 필터
    st.markdown("### 🔍 필터")
    filter_pass = st.checkbox("PASS만 보기", value=False)
    filter_fail = st.checkbox("FAIL만 보기", value=False)

    score_range = st.slider(
        "점수 범위 (%)",
        min_value=0,
        max_value=100,
        value=(0, 100),
        step=10
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; font-size: 1rem; color: #888; padding: 0.5rem 0;'>
        Copyright © 2025<br>
        ITCEN CLOIT<br>
        All rights reserved.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================ #
# 메인 영역
# ============================================================================ #

# 헤더
st.markdown('<div class="main-header">🎓 수학 문제 자동 채점 시스템 (개선판)</div>', unsafe_allow_html=True)

# 탭 생성
tab1, tab2, tab3 = st.tabs([
    "📊 대시보드",
    "📝 문제별 학생풀이 분석",
    "💾 문제 메타데이터"
])


# ============================================================================ #
# 홈 탭
# ============================================================================ #

with tab1:
    if not selected_batch:
        st.warning("선택된 배치가 없습니다.")
    else:
        st.markdown(f"### 📊 전체 통계 요약")

        # 데이터 로드
        analyses = load_all_analyses(selected_batch)

        if not analyses:
            st.error("분석 결과가 없습니다.")
        else:
            # 필터 적용
            initial_filtered_analyses = analyses
            if filter_pass:
                initial_filtered_analyses = [a for a in initial_filtered_analyses if a.get('expected_result') == 'PASS']
            if filter_fail:
                initial_filtered_analyses = [a for a in initial_filtered_analyses if a.get('expected_result') == 'FAIL']

            # 점수 범위 필터
            score_filtered_analyses = []
            for a in initial_filtered_analyses:
                final_score = safe_int(a.get('final_score'))
                total_possible = safe_int(a.get('total_possible'))
                if total_possible > 0:
                    percentage = (final_score / total_possible) * 100
                    if score_range[0] <= percentage <= score_range[1]:
                        score_filtered_analyses.append(a)
            
            filtered_analyses = score_filtered_analyses

            # 기본 통계
            total_count = len(filtered_analyses)
            pass_count = sum(1 for a in filtered_analyses if a.get('expected_result') == 'PASS')
            fail_count = sum(1 for a in filtered_analyses if a.get('expected_result') == 'FAIL')

            scores = [safe_int(a.get('final_score')) for a in filtered_analyses]
            total_possibles = [safe_int(a.get('total_possible')) for a in filtered_analyses]

            if scores:
                avg_score = sum(scores) / len(scores) if scores else 0
                avg_total = sum(total_possibles) / len(total_possibles) if total_possibles else 0
                avg_percentage = (avg_score / avg_total * 100) if avg_total > 0 else 0
            else:
                avg_score = 0
                avg_total = 0
                avg_percentage = 0

            # 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(label="📝 총 풀이 수", value=total_count)
            with col2:
                st.metric(label="📈 평균 점수", value=f"{avg_score:.1f} / {avg_total:.1f}")
            with col3:
                st.metric(label="✅ PASS", value=f"{pass_count}개")
            with col4:
                st.metric(label="⚠️ FAIL", value=f"{fail_count}개")

            st.markdown("---")

            # 점수 분포 차트
            st.markdown("### 📊 점수 분포")
            if scores:
                score_ranges = {'90-100%': 0, '80-89%': 0, '70-79%': 0, '60-69%': 0, '60% 미만': 0}
                for score, total in zip(scores, total_possibles):
                    percentage = (score / total * 100) if total > 0 else 0
                    if percentage >= 90: score_ranges['90-100%'] += 1
                    elif percentage >= 80: score_ranges['80-89%'] += 1
                    elif percentage >= 70: score_ranges['70-79%'] += 1
                    elif percentage >= 60: score_ranges['60-69%'] += 1
                    else: score_ranges['60% 미만'] += 1

                fig = go.Figure(data=[go.Bar(
                    x=list(score_ranges.keys()),
                    y=list(score_ranges.values()),
                    text=list(score_ranges.values()),
                    textposition='auto',
                    marker_color=['#28a745', '#5cb85c', '#ffc107', '#fd7e14', '#dc3545']
                )])
                fig.update_layout(xaxis_title="점수 구간", yaxis_title="학생 수", height=400, showlegend=False)
                st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # 문제별 성적 테이블
            st.markdown("### 📢 문제별 성적")
            problem_stats = defaultdict(lambda: {'count': 0, 'scores': [], 'total_possibles': []})
            for analysis in filtered_analyses:
                problem_id = analysis.get('problem_id', 'Unknown')
                problem_stats[problem_id]['count'] += 1
                problem_stats[problem_id]['scores'].append(safe_int(analysis.get('final_score')))
                problem_stats[problem_id]['total_possibles'].append(safe_int(analysis.get('total_possible')))

            table_data = []
            for problem_id, stats in sorted(problem_stats.items()):
                if stats['scores']:
                    avg_s = sum(stats['scores']) / stats['count']
                    avg_t = sum(stats['total_possibles']) / stats['count']
                    avg_p = (avg_s / avg_t * 100) if avg_t > 0 else 0
                    table_data.append({
                        '문제 ID': problem_id,
                        '풀이 수': stats['count'],
                        '평균 점수': f"{avg_s:.2f} / {avg_t:.1f}",
                        '평균 비율': f"{avg_p:.1f}%"
                    })

            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, width='stretch', hide_index=True)


# ============================================================================ #
# 문제별 학생풀이 분석 탭
# ============================================================================ #

with tab2:
    if not selected_batch:
        st.warning("선택된 배치가 없습니다.")
    else:
        analyses = load_all_analyses(selected_batch)

        if not analyses:
            st.error("분석 결과가 없습니다.")
        else:
            # 문제 ID 목록
            problem_ids = sorted(list(set(a.get('problem_id') for a in analyses)))

            selected_problem = st.selectbox(
                "📍 문제 선택",
                problem_ids,
                format_func=lambda x: f"문제 {x}",
                key="problem_selector_update"
            )

            if selected_problem:
                problem_analyses = [a for a in analyses if a.get('problem_id') == selected_problem]

                st.markdown(f"### 📝 문제 {selected_problem} 상세 분석")

                # 문제 이미지 표시
                question_image_path = f"resource/question/{selected_problem}.png"
                if os.path.exists(question_image_path):
                    st.markdown("#### 📷 문제 원본")
                    col1, col2, col3 = st.columns([1, 5, 1])
                    with col2:
                        st.image(question_image_path, width='stretch')
                    st.markdown("---")

                # 각 풀이 상세
                st.markdown(f"### 👥 학생 풀이 목록 ({len(problem_analyses)}개)")

                for idx, analysis in enumerate(problem_analyses, 1):
                    solution_file = analysis.get('solution_file', 'Unknown')
                    final_score = safe_int(analysis.get('final_score'))
                    total_possible = safe_int(analysis.get('total_possible'), default=1)
                    percentage = (final_score / total_possible * 100) if total_possible > 0 else 0
                    if percentage >= 90: emoji = "✅"
                    elif percentage >= 70: emoji = "⚠️"
                    else: emoji = "❌"
                    expander_title = f"📄 {solution_file} - {final_score}/{total_possible}점 ({percentage:.1f}%) {emoji}"

                    with st.expander(expander_title, expanded=(idx == 1)):
                        
                        st.subheader("📝 학생 풀이 및 AI 기본 분석")
                        main_col1, main_col2 = st.columns([6, 4])

                        with main_col1:
                            st.markdown("##### 학생 풀이 이미지")
                            visualized_image_path = Path(selected_batch) / "visualized" / solution_file
                            if visualized_image_path.exists():
                                with st.expander("BBox 시각화 이미지 보기/숨기기", expanded=False):
                                    st.image(str(visualized_image_path), width='stretch')
                            else:
                                original_solve_path = Path("resource/solve") / solution_file
                                if original_solve_path.exists():
                                    st.image(str(original_solve_path), width='stretch')
                                else:
                                    st.warning("풀이 이미지를 찾을 수 없습니다.")

                        with main_col2:
                            st.markdown("##### AI 오류 분석")
                            error_location = analysis.get('first_error_location')
                            if error_location and error_location.get('has_error'):
                                st.error(
                                    f"**오류 발생 지점:** Step {error_location.get('error_step_number', 'N/A')}, "
                                    f"Box ID **{error_location.get('error_box_id', 'N/A')}**\n\n"
                                    f"**사유:** {error_location.get('reason', 'N/A')}"
                                )
                            else:
                                st.success("**오류 미발견:** AI가 풀이 과정에서 명백한 오류를 찾지 못했습니다.")

                        st.divider()

                        st.subheader("🤖 AI 종합 평가")
                        overall_eval = analysis.get('overall_evaluation', {})
                        st.info(f"**총평 요약:** {overall_eval.get('summary', '요약 정보 없음')}")
                        
                        eval_col1, eval_col2 = st.columns(2)
                        with eval_col1:
                            with st.container(border=True):
                                st.markdown("##### 강점 👍")
                                strengths = overall_eval.get('strengths', [])
                                if strengths:
                                    st.markdown("<ul style='margin-bottom: 0;'>", unsafe_allow_html=True)
                                    for item in strengths: st.markdown(f"<li style='margin-bottom: 0.2em; line-height: 1.2em;'>{item}</li>", unsafe_allow_html=True)
                                    st.markdown("</ul>", unsafe_allow_html=True)
                                else: st.caption("_발견된 강점 없음_")
                        with eval_col2:
                            with st.container(border=True):
                                st.markdown("##### 약점 👎")
                                weaknesses = overall_eval.get('weaknesses', [])
                                if weaknesses:
                                    st.markdown("<ul style='margin-bottom: 0;'>", unsafe_allow_html=True)
                                    for item in weaknesses: st.markdown(f"<li style='margin-bottom: 0.2em; line-height: 1.2em;'>{item}</li>", unsafe_allow_html=True)
                                    st.markdown("</ul>", unsafe_allow_html=True)
                                else: st.caption("_발견된 약점 없음_")

                        st.divider()

                        st.subheader("📊 단계별 채점 결과")
                        eval_details = analysis.get('step_by_step_evaluation', [])
                        if eval_details:
                            for step in eval_details:
                                status = step.get('status', 'NotAttempted')
                                if status == 'Correct': step_status = "✅"
                                elif status in ['Incorrect', 'Partial']: step_status = "❌"
                                else: step_status = "⚠️"

                                with st.container(border=True):
                                    st.markdown(f"**{step_status} {step.get('step_number')}단계: {step.get('step_name', '')}** ({safe_int(step.get('points_earned'))}/{safe_int(step.get('points_possible'))}점)")
                                    
                                    # LaTeX 렌더링 로직 개선
                                    student_work_latex = step.get('student_work_latex')
                                    if student_work_latex:
                                        with st.container(border=True):
                                            st.caption("학생 답안 (인식된 LaTeX)")
                                            cleaned_latex = student_work_latex.strip()
                                            if cleaned_latex.startswith('$$') and cleaned_latex.endswith('$$'):
                                                cleaned_latex = cleaned_latex[2:-2]
                                            elif cleaned_latex.startswith('$') and cleaned_latex.endswith('$'):
                                                cleaned_latex = cleaned_latex[1:-1]
                                            
                                            # 이중 백슬래시를 단일 백슬래시로 변경
                                            cleaned_latex = cleaned_latex.replace('\\\\', '\\')
                                            
                                            try:
                                                st.latex(cleaned_latex)
                                            except Exception as e:
                                                st.warning(f"LaTeX 렌더링 실패: {e}")
                                                st.code(student_work_latex, language='latex') # 실패 시 원본 표시
                                    elif 'student_work' in step and step['student_work']:
                                        with st.container(border=True):
                                            st.caption("학생 답안 (인식된 텍스트)")
                                            st.text(step.get('student_work', ''))

                                    evaluation = step.get('evaluation', 'N/A').replace('\\', '')
                                    st.markdown(f"**평가:** {evaluation}")
                                    if status != 'Correct':
                                        feedback = step.get('feedback', 'N/A').replace('\\', '')
                                        st.warning(f"**피드백:** {feedback}")
                        else:
                            st.info("단계별 채점 결과가 없습니다.")
                        st.divider()

                        st.subheader("🌱 개선 제안 및 상세 피드백")
                        sugg_col1, sugg_col2 = st.columns(2)
                        with sugg_col1:
                            st.markdown("##### 개선 제안")
                            suggestions = analysis.get('improvement_suggestions', [])
                            if suggestions:
                                st.markdown("<ul style='margin-bottom: 0;'>", unsafe_allow_html=True)
                                for suggestion in suggestions:
                                    st.markdown(f"<li style='margin-bottom: 0.2em; line-height: 1.2em;'>{suggestion}</li>", unsafe_allow_html=True)
                                st.markdown("</ul>", unsafe_allow_html=True)
                            else:
                                st.caption("개선 제안 없음")

                        with sugg_col2:
                            st.markdown("##### 상세 피드백")
                            if analysis.get('detailed_feedback'):
                                st.info(analysis['detailed_feedback'].replace('\\\\', '\\'))
                            else:
                                st.caption("상세 피드백 없음")

                        if st.checkbox("전체 채점 결과 보기 (JSON)", key=f"json_{solution_file}"):
                            st.json(analysis)



# ============================================================================ #
# 메타데이터 탭
# ============================================================================ #

with tab3:
    st.markdown("### 🚀 메타데이터")

    metadata_files = get_available_metadata()

    if not metadata_files:
        st.error("메타데이터 파일이 없습니다.")
    else:
        selected_metadata = st.selectbox(
            "📍 문제 선택",
            metadata_files,
            format_func=lambda x: x.replace('_metadata.json', '')
        )

        if selected_metadata:
            problem_id = selected_metadata.replace('_metadata.json', '')
            metadata = load_metadata(problem_id)

            if metadata:
                # 문제 이미지 표시
                question_image_path = f"resource/question/{problem_id}.png"
                if os.path.exists(question_image_path):
                    st.markdown("### 📷 문제 이미지")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(question_image_path, width='stretch')
                    st.markdown("---")

                meta = metadata.get('metadata', {})

                # 교육과정 정보
                st.markdown("### 📚 교육과정 정보")
                curriculum = meta.get('curriculum_mapping', {})

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**대단원:** {curriculum.get('대단원', 'N/A')}")
                    st.info(f"**중단원:** {curriculum.get('중단원', 'N/A')}")
                with col2:
                    st.info(f"**소단원:** {curriculum.get('소단원', 'N/A')}")
                    st.info(f"**성취기준:** {curriculum.get('성취기준_코드', 'N/A')}")

                st.markdown("---")

                # 문제 분석
                st.markdown("### 📝 문제 분석")
                problem_analysis = meta.get('problem_analysis', {})

                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**문제 유형:** {problem_analysis.get('problem_type', 'N/A')}")
                with col2:
                    st.info(f"**난이도:** {problem_analysis.get('difficulty', 'N/A')}")

                required_concepts = problem_analysis.get('required_concepts', [])
                if required_concepts:
                    st.info(f"**필요 개념:** {', '.join(required_concepts)}")

                if 'problem_intent' in problem_analysis:
                    st.info(f"**출제 의도:** {problem_analysis['problem_intent']}")

                st.markdown("---")

                # 풀이 단계
                st.markdown("### 📋 풀이 단계")
                solution_steps = meta.get('solution_steps', [])

                for step in solution_steps:
                    with st.expander(f"**{step.get('step_number')}단계: {step.get('step_name')}** (배점: {step.get('points')}점)"):
                        st.markdown(f"**설명:** {step.get('description')}")
                        st.markdown(f"**핵심 개념:** {step.get('key_concept')}")
                        st.markdown(f"**기대 행동:** {step.get('expected_action')}")

                        common_errors = step.get('common_errors', [])
                        if common_errors:
                            st.warning("**흔한 오류:**")
                            for error in common_errors:
                                st.write(f"- {error}")

                st.markdown("---")

                # JSON 원본
                if st.button("📄 JSON 원본 보기"):
                    st.json(metadata)


# ============================================================================ #
# 푸터
# ============================================================================ #

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Copyright © 2025ITCEN CLOIT. All rights reserved.
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    import sys
    import subprocess

    if len(sys.argv) == 1:
        subprocess.run([
            "streamlit", "run", __file__,
            "--server.headless", "true"
        ])