import streamlit as st
import time
import warnings

st.set_page_config(
    page_title="AI Math Assistant - MATHBOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------- CSS 커스터마이징 -------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 12px !important;
        margin: 10px 0 !important;
    }

    /* 헤더 */
    .header {
        background: white;
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e0e0e0;
        border-radius: 12px 12px 0 0;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header h2 { margin: 0; font-size: 1.6em; }
    .header .icons { font-size: 1.8em; }

    /* 퀵 액션 버튼 */
    .quick-btn button {
        background: #e3e3e3 !important;
        border: none !important;
        border-radius: 30px !important;
        color: #333 !important;
    }

    /* 분석/피드백 박스 */
    .analysis-box {
        background: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .feedback-box {
        background: #f0f8ff;
        border: 1px solid #cce5ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .good { color: #006400; font-weight: bold; }
    .improve { color: #b22222; font-weight: bold; }

    /* 스크롤 가능한 채팅 영역 */
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- 헤더 -------------------
st.markdown("""
<div class="header">
    <h2>🤖 AI 수학 도우미, MATHBOT</h2>
    <div class="icons">📊 📚 👤</div>
</div>
""", unsafe_allow_html=True)

# ------------------- 채팅 히스토리 초기화 -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# 처음 실행 시 초기 메시지 추가
if len(st.session_state.messages) == 0:
    st.session_state.messages.extend([
        {"role": "assistant", "content": """안녕하세요! AI 수학 도우미, MATHBOT입니다.  
                  문제 풀이를 검증하거나, 모르는 개념을 질문하거나, 유사 문제를 풀어볼 수 있어요.  
                  무엇을 도와드릴까요?""", "quick_buttons": True},
        {"role": "user", "content": "이차함수 문제를 풀었는데 확인해주세요."},
        {"role": "assistant", "content": "업로드된 이미지", "is_image": True},
        {"role": "assistant", "content": "분석 시작", "show_analysis": True}
    ])

# ------------------- 채팅 메시지 출력 -------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
            if "content" in msg:
                st.markdown(msg["content"])

            # 퀵 버튼 (첫 메시지에만)
            if msg.get("quick_buttons"):
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.button("문제 풀이 검증", key="q1", help="풀이가 맞는지 확인")
                with col2: st.button("개념 설명", key="q2")
                with col3: st.button("유사 문제", key="q3")
                with col4: st.button("오답 분석", key="q4")

            # 이미지 플레이스홀더
            if msg.get("is_image"):
                col1, col2 = st.columns(2)
                with col1:
                    st.info("📷 [업로드된 문제 이미지.png]")
                with col2:
                    st.info("📝 [업로드된 풀이 이미지.png]")

            # 분석 애니메이션 + 결과 (한 번만 실행되게)
            if msg.get("show_analysis") and not st.session_state.get("analysis_done"):
                st.write("네! 문제와 풀이를 확인했습니다. AI가 분석을 시작합니다.")

                with st.spinner("🔍 문제 분석 중..."):
                    progress = st.progress(0)
                    status = st.empty()
                    for i in range(100):
                        time.sleep(0.03)
                        progress.progress(i + 1)
                        if i < 30:
                            status.text("✓ 문제 텍스트 추출 완료")
                        elif i < 70:
                            status.text("✓ 수식 인식 및 파싱 완료")
                        else:
                            status.text("✓ 풀이 과정 논리 검증 중...")

                    time.sleep(0.5)
                    progress.empty()
                    status.empty()

                st.success("✓ 풀이 검증 완료!")

                # 결과 출력
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### ✓ 풀이 검증 결과")
                    st.success("**정답 여부: ✅ 정답!**")
                    st.metric("정확도", "92%", "우수")
                    st.write("**단계별 검증**")
                    st.write("1. 문제 이해: ✅")
                    st.write("2. 공식 적용: ✅")
                    st.write("3. 계산 과정: ✅")
                    st.warning("4. 답안 작성: ⚠️ (단위 누락)")

                with col2:
                    st.markdown("### 💡 AI 맞춤 피드백")
                    st.markdown("<p class='good'>💯 잘한 점</p>", unsafe_allow_html=True)
                    st.write("• 이차함수 공식을 정확히 적용했어요.")
                    st.write("• 계산 과정이 체계적이고 논리적입니다.")
                    st.markdown("---")
                    st.markdown("<p class='improve'>📝 개선 사항</p>", unsafe_allow_html=True)
                    st.write("• 최종 답에 단위(cm²)를 꼭 쓰세요!")
                    st.write("• 검산 과정을 추가하면 실수를 줄일 수 있어요.")

                st.markdown("### 🎯 추천 학습")
                c1, c2, c3 = st.columns(3)
                c1.button("유사 문제 3개 풀기", use_container_width=True)
                c2.button("관련 개념 다시 보기", use_container_width=True)
                c3.button("오답 노트에 저장", use_container_width=True)

                # 분석 완료 플래그
                st.session_state.analysis_done = True

# ------------------- 사용자 입력 -------------------
if prompt := st.chat_input("메시지를 입력하거나 사진/파일을 업로드하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"네, '{prompt}' 에 대해 도와드릴게요! 조금만 기다려주세요..."
    })

    st.rerun()