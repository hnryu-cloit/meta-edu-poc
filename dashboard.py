"""
수학 문제 자동 채점 시스템 대시보드
"""

import streamlit as st
import json
import os
from glob import glob
from collections import defaultdict
import plotly.graph_objects as go
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="수학 채점 시스템",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 설정
st.markdown("""
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
    result_dirs = glob("results/old/batch_*")
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
    result_dirs = glob("results/old/batch_*")
    return sorted(result_dirs, key=os.path.getmtime, reverse=True)


def get_available_metadata():
    """사용 가능한 메타데이터 파일 목록"""
    metadata_files = glob("metadata/*_metadata.json")
    return sorted([os.path.basename(f) for f in metadata_files])


with st.sidebar:
    st.markdown("### 🎓 수학 채점 시스템")
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
        "점수 범위",
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


# ============================================================================
# 메인 영역
# ============================================================================

# 헤더
st.markdown('<div class="main-header">🎓 수학 문제 자동 채점 시스템</div>', unsafe_allow_html=True)

# 탭 생성
tab1, tab2, tab3 = st.tabs([
    "🏠 홈",
    "📝 문제별 학생풀이 분석",
    "💾 문제 메타데이터"
])


# ============================================================================
# 홈 탭
# ============================================================================

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
            filtered_analyses = analyses
            if filter_pass:
                filtered_analyses = [a for a in filtered_analyses if a.get('expected_result') == 'PASS']
            if filter_fail:
                filtered_analyses = [a for a in filtered_analyses if a.get('expected_result') == 'FAIL']

            # 점수 범위 필터
            filtered_analyses = [
                a for a in filtered_analyses
                if 'analysis' in a and
                   score_range[0] <= (a['analysis']['final_score'] / a['analysis']['total_possible'] * 100) <= score_range[1]
            ]

            # 기본 통계
            total_count = len(filtered_analyses)
            pass_count = sum(1 for a in filtered_analyses if a.get('expected_result') == 'PASS')
            fail_count = sum(1 for a in filtered_analyses if a.get('expected_result') == 'FAIL')

            scores = [a['analysis']['final_score'] for a in filtered_analyses if 'analysis' in a]
            total_possibles = [a['analysis']['total_possible'] for a in filtered_analyses if 'analysis' in a]

            if scores:
                avg_score = sum(scores) / len(scores)
                avg_total = sum(total_possibles) / len(total_possibles)
                avg_percentage = (avg_score / avg_total * 100) if avg_total > 0 else 0
            else:
                avg_score = 0
                avg_total = 10
                avg_percentage = 0

            # 메트릭 카드
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="📝 총 풀이 수",
                    value=total_count
                )

            with col2:
                st.metric(
                    label="📈 평균 점수",
                    value=f"{avg_score:.1f}/{avg_total:.0f}",
                )

            with col3:
                st.metric(
                    label="✅ PASS",
                    value=f"{pass_count}개",
                )

            with col4:
                st.metric(
                    label="⚠️ FAIL",
                    value=f"{fail_count}개",
                )

            st.markdown("---")

            # 점수 분포 차트
            st.markdown("### 📊 점수 분포")

            if scores:
                score_ranges = {
                    '90-100%': 0,
                    '80-89%': 0,
                    '70-79%': 0,
                    '60-69%': 0,
                    '60% 미만': 0
                }

                for score, total_possible in zip(scores, total_possibles):
                    percentage = (score / total_possible * 100) if total_possible > 0 else 0
                    if percentage >= 90:
                        score_ranges['90-100%'] += 1
                    elif percentage >= 80:
                        score_ranges['80-89%'] += 1
                    elif percentage >= 70:
                        score_ranges['70-79%'] += 1
                    elif percentage >= 60:
                        score_ranges['60-69%'] += 1
                    else:
                        score_ranges['60% 미만'] += 1

                fig = go.Figure(data=[
                    go.Bar(
                        x=list(score_ranges.keys()),
                        y=list(score_ranges.values()),
                        text=list(score_ranges.values()),
                        textposition='auto',
                        marker_color=['#28a745', '#5cb85c', '#ffc107', '#fd7e14', '#dc3545']
                    )
                ])

                fig.update_layout(
                    xaxis_title="점수 구간",
                    yaxis_title="학생 수",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig, width='stretch')

            st.markdown("---")

            # 문제별 성적 테이블
            st.markdown("### 📢 문제별 성적")

            problem_stats = defaultdict(lambda: {'count': 0, 'scores': [], 'total_possibles': []})
            for analysis in filtered_analyses:
                problem_id = analysis.get('problem_id', 'Unknown')
                problem_stats[problem_id]['count'] += 1
                if 'analysis' in analysis:
                    problem_stats[problem_id]['scores'].append(analysis['analysis']['final_score'])
                    problem_stats[problem_id]['total_possibles'].append(analysis['analysis']['total_possible'])

            table_data = []
            for problem_id in sorted(problem_stats.keys()):
                stats = problem_stats[problem_id]
                if stats['scores']:
                    avg_score = sum(stats['scores']) / len(stats['scores'])
                    avg_total = sum(stats['total_possibles']) / len(stats['total_possibles'])
                    avg_percentage = (avg_score / avg_total * 100) if avg_total > 0 else 0

                    table_data.append({
                        '문제 ID': problem_id,
                        '풀이 수': stats['count'],
                        '평균 점수': f"{avg_score:.2f}/{avg_total:.0f}",
                        '평균 비율': f"{avg_percentage:.1f}%"
                    })

            if table_data:
                df = pd.DataFrame(table_data)
                st.dataframe(df, width='stretch', hide_index=True)


# ============================================================================
# 문제별 분석 탭
# ============================================================================

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
                format_func=lambda x: f"문제 {x}"
            )

            if selected_problem:
                problem_analyses = [a for a in analyses if a.get('problem_id') == selected_problem]

                st.markdown(f"### 📝 문제 {selected_problem} 상세 정보")

                # 문제 이미지 표시
                question_image_path = f"resource/question/{selected_problem}.png"
                if os.path.exists(question_image_path):
                    st.markdown("#### 📷 문제 이미지")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(question_image_path, width='stretch')
                    st.markdown("---")

                # 문제 기본 정보
                if problem_analyses and 'analysis' in problem_analyses[0]:
                    first = problem_analyses[0]['analysis']
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.info(f"**정답:** {first.get('correct_answer', 'N/A')}")
                    with col2:
                        st.info(f"**배점:** {first.get('total_possible', 'N/A')}점")
                    with col3:
                        st.info(f"**풀이 수:** {len(problem_analyses)}개")

                st.markdown("---")

                # 각 풀이 상세
                st.markdown(f"### 👥 학생 풀이 목록 ({len(problem_analyses)}개)")

                for idx, analysis in enumerate(problem_analyses, 1):
                    solution_file = analysis.get('solution_file', 'Unknown')
                    expected = analysis.get('expected_result', 'N/A')

                    if 'analysis' in analysis:
                        final_score = analysis['analysis']['final_score']
                        total_possible = analysis['analysis']['total_possible']
                        percentage = (final_score / total_possible * 100) if total_possible > 0 else 0
                        is_alternative = analysis['analysis'].get('is_alternative_method', False)

                        # 점수에 따른 색상
                        if percentage >= 90:
                            badge_class = "success-badge"
                            emoji = "✅"
                        elif percentage >= 70:
                            badge_class = "warning-badge"
                            emoji = "⚠️"
                        else:
                            badge_class = "danger-badge"
                            emoji = "❌"

                        with st.expander(f"📄 {solution_file} - {final_score}/{total_possible}점 ({percentage:.1f}%) {emoji}", expanded=(idx == 1)):
                            # 학생 풀이 이미지 표시
                            solution_image_path = f"resource/solve/{solution_file}"
                            if os.path.exists(solution_image_path):
                                st.markdown("#### 📷 학생 풀이")

                                # ========== 바운딩 박스 오버레이 ==========
                                # OCR 데이터 확인
                                ocr_data = analysis.get('ocr_data', {})
                                grouped_bboxes = ocr_data.get('step_grouped_bboxes', {})

                                if grouped_bboxes:
                                    # 단계별 토글 버튼
                                    st.markdown("##### 🎯 틀린 단계 표시")

                                    # 틀린 단계만 필터링
                                    incorrect_steps = []
                                    step_info_map = {}

                                    if 'step_by_step_evaluation' in analysis:
                                        for step_eval in analysis['step_by_step_evaluation']:
                                            step_num = step_eval.get('step_number', 0)
                                            step_key = f"step_{step_num}"
                                            step_status = step_eval.get('status', 'Unknown')

                                            # Incorrect 또는 Partial 단계만 포함
                                            if step_status in ['Incorrect', 'Partial']:
                                                incorrect_steps.append(step_key)
                                                step_info_map[step_key] = {
                                                    'name': step_eval.get('step_name', f'단계 {step_num}'),
                                                    'feedback': step_eval.get('feedback', ''),
                                                    'number': step_num
                                                }

                                    # 틀린 단계가 있으면 토글 버튼 표시
                                    if incorrect_steps:
                                        st.info(f"💡 틀린 단계를 선택하면 해당 영역이 형광펜으로 표시됩니다.")

                                        # 각 틀린 단계별 체크박스
                                        selected_steps = []
                                        cols = st.columns(len(incorrect_steps))

                                        for col_idx, step_key in enumerate(incorrect_steps):
                                            step_info = step_info_map[step_key]
                                            step_num = step_info['number']
                                            step_name = step_info['name']

                                            # 단계별 색상 가져오기
                                            from utils.grade_visualizer import get_step_color
                                            from utils.grade_visualizer import create_interactive_bbox_overlay
                                            step_color = get_step_color(step_num)

                                            with cols[col_idx]:
                                                # 색상 표시와 함께 체크박스
                                                checkbox_label = f"{step_num}단계"
                                                is_checked = st.checkbox(
                                                    checkbox_label,
                                                    value=False,
                                                    key=f"step_toggle_{solution_file}_{step_key}"
                                                )

                                                # 색상 표시
                                                st.markdown(
                                                    f'<div style="background-color: {step_color["rgba"]}; '
                                                    f'border: 2px solid {step_color["border"]}; '
                                                    f'padding: 5px; border-radius: 5px; text-align: center; '
                                                    f'font-size: 0.8rem;">{step_color["name"]}</div>',
                                                    unsafe_allow_html=True
                                                )

                                                if is_checked:
                                                    selected_steps.append(step_key)

                                        # 선택된 단계의 바운딩 박스 표시
                                        if selected_steps:
                                            st.markdown("##### 🔍 선택된 단계의 오류 영역")

                                            # 이미지 오버레이 생성
                                            from utils.grade_visualizer import create_interactive_bbox_overlay

                                            overlay_html = create_interactive_bbox_overlay(
                                                solution_image_path,
                                                grouped_bboxes,
                                                selected_steps,
                                                width=800
                                            )

                                            st.markdown(overlay_html, unsafe_allow_html=True)

                                            # 선택된 단계의 피드백 표시
                                            for step_key in selected_steps:
                                                step_info = step_info_map[step_key]
                                                step_num = step_info['number']
                                                step_name = step_info['name']
                                                feedback = step_info['feedback']

                                                st.warning(f"**{step_num}단계 ({step_name})**: {feedback}")
                                        else:
                                            # 선택된 단계가 없으면 원본 이미지 표시
                                            img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                                            with img_col2:
                                                st.image(solution_image_path, width='stretch')
                                    else:
                                        # 틀린 단계가 없으면 원본 이미지만 표시
                                        img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                                        with img_col2:
                                            st.image(solution_image_path, width='stretch')
                                else:
                                    # OCR 데이터가 없으면 원본 이미지만 표시
                                    img_col1, img_col2, img_col3 = st.columns([1, 3, 1])
                                    with img_col2:
                                        st.image(solution_image_path, width='stretch')
                                # ==========================================

                                st.markdown("---")

                            # 수학적 방법 검증 표시
                            if 'mathematical_methods_used' in analysis['analysis']:
                                st.markdown("#### 🔍 사용된 수학적 방법 검증")
                                methods = analysis['analysis']['mathematical_methods_used']
                                for method in methods:
                                    method_name = method.get('method_name', 'N/A')
                                    is_valid = method.get('is_valid', False)
                                    validation_comment = method.get('validation_comment', '')

                                    if is_valid:
                                        st.success(f"✓ **{method_name}**: {validation_comment}")
                                    else:
                                        st.error(f"✗ **{method_name}**: {validation_comment}")
                                st.markdown("---")

                            # 학생 풀이 LaTeX 표시
                            if 'student_solution_latex' in analysis['analysis']:
                                st.markdown("#### 📐 학생 풀이 (수식)")
                                student_latex = analysis['analysis']['student_solution_latex']
                                try:
                                    st.latex(student_latex)
                                except Exception as e:
                                    st.warning(f"LaTeX 렌더링 실패: {e}")
                                    st.code(student_latex, language='latex')
                                st.markdown("---")

                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**예상 결과:** {expected}")
                                st.markdown(f"**대안 풀이:** {'예' if is_alternative else '아니오'}")

                            with col2:
                                st.markdown(f"**최종 점수:** {final_score}/{total_possible}")
                                st.markdown(f"**정답률:** {percentage:.1f}%")

                            st.markdown("#### 📊 단계별 평가")

                            # 단계별 평가 테이블
                            if 'step_by_step_evaluation' in analysis['analysis']:
                                for step in analysis['analysis']['step_by_step_evaluation']:
                                    step_status = "✓" if step.get('status') == 'Correct' else "✗"
                                    step_color = "green" if step.get('status') == 'Correct' else "red"

                                    st.markdown(f"**{step_status} {step.get('step_number')}단계:** {step.get('step_name')}")
                                    st.markdown(f"- 점수: {step.get('points_earned')}/{step.get('points_possible')}점")
                                    st.markdown(f"- 평가: {step.get('evaluation')}")

                                    # 학생 작업 LaTeX 표시
                                    if 'student_work_latex' in step:
                                        st.markdown("**학생 풀이 (수식):**")
                                        try:
                                            st.latex(step['student_work_latex'])
                                        except Exception:
                                            st.code(step['student_work_latex'], language='latex')

                                    if step.get('status') != 'Correct':
                                        st.warning(f"💡 피드백: {step.get('feedback')}")

                            st.markdown("#### 💬 전체 피드백")
                            if 'detailed_feedback' in analysis['analysis']:
                                st.info(analysis['analysis']['detailed_feedback'])


# ============================================================================
# 메타데이터 탭
# ============================================================================

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


# ============================================================================
# 푸터
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Copyright © 2025 ITCEN CLOIT. All rights reserved.
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
