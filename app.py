"""
Dockerized AI Agent — Streamlit UI
LangGraph + Llama 3 via Groq
Runs identically everywhere via Docker
"""

import os
import streamlit as st

st.set_page_config(
    page_title="Dockerized AI Agent",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main, .stApp { background-color: #0f1117; }
.hero { font-size:2.5rem; font-weight:800;
  background:linear-gradient(135deg,#0db7ed,#6c63ff);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#888; font-size:1rem; margin-bottom:1.5rem; }
.card { background:#1e2130; border:1px solid #2d3148; border-radius:12px; padding:20px 24px; margin:10px 0; }
.tool-badge { display:inline-block; background:#0db7ed22; border:1px solid #0db7ed55;
  color:#0db7ed; border-radius:6px; padding:3px 10px; margin:3px; font-size:.8rem; }
.msg-user { background:#1a1d2e; border-left:3px solid #6c63ff; border-radius:8px; padding:12px 16px; margin:8px 0; }
.msg-ai { background:#1a2a1a; border-left:3px solid #43b89c; border-radius:8px; padding:12px 16px; margin:8px 0; }
.docker-badge { background:#0db7ed22; border:1px solid #0db7ed; border-radius:20px;
  padding:4px 14px; color:#0db7ed; font-size:.85rem; font-weight:600; }
div[data-testid="stSidebar"] { background:#151825; }
.stButton>button { background:linear-gradient(135deg,#0db7ed,#6c63ff);
  color:#fff; border:none; border-radius:8px; padding:12px 28px; font-weight:700; width:100%; }
.stTextInput input, .stTextArea textarea { background:#1e2130 !important;
  color:#e0e0e0 !important; border:1px solid #2d3148 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🐳 Docker Agent")
    st.markdown("---")

    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    elif os.getenv("GROQ_API_KEY"):
        st.success("✅ API key loaded from environment")

    st.markdown("---")
    st.markdown("### 🔧 Available Tools")
    tools_info = [
        ("🧮", "Calculator", "Math expressions"),
        ("🐍", "Python Runner", "Execute code"),
        ("📊", "Text Analyzer", "Text statistics"),
        ("📚", "Knowledge Base", "AI/tech topics"),
    ]
    for icon, name, desc in tools_info:
        st.markdown(f"{icon} **{name}** — {desc}")

    st.markdown("---")
    st.markdown("### 🐳 Docker Info")
    st.code("docker pull rohan/ai-agent\ndocker run -p 8501:8501 \\\n  -e GROQ_API_KEY=gsk_... \\\n  rohan/ai-agent", language="bash")

    st.markdown("---")
    st.markdown("**Stack:** LangGraph · Llama 3 · Docker")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero">🐳 Dockerized AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">A LangGraph + Llama 3 agent packaged with Docker — runs the same way everywhere</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<span class="docker-badge">🐳 Docker Ready</span>', unsafe_allow_html=True)
with col2:
    st.markdown('<span class="docker-badge">🔄 Runs Anywhere</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="docker-badge">⚡ Groq Powered</span>', unsafe_allow_html=True)

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("---")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Example prompts
examples = [
    "🧮 What is sqrt(144) + sin(30) * 100?",
    "🐍 Write and run Python code to generate fibonacci numbers up to 100",
    "📊 Analyze this text: Docker is a platform for containerizing applications that run consistently across environments",
    "📚 What is LangGraph and how does it work?",
    "🧮 Calculate compound interest: principal 10000, rate 7%, time 5 years",
    "🐍 Write Python code to find all prime numbers up to 50",
]

st.markdown("**💡 Try an example:**")
cols = st.columns(3)
for i, ex in enumerate(examples):
    with cols[i % 3]:
        label = ex[:45] + "..."
        if st.button(label, key=f"ex{i}"):
            st.session_state["user_input"] = ex


# Display chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            tools_used = msg.get("tools_used", [])
            tool_badges = " ".join([f'<span class="tool-badge">🔧 {t}</span>' for t in tools_used])
            st.markdown(f'<div class="msg-ai">🤖 <b>Agent:</b><br>{msg["content"]}{("<br>" + tool_badges) if tool_badges else ""}</div>', unsafe_allow_html=True)

# Input
st.markdown("---")
user_input = st.text_input(
    "Ask the agent anything",
    value=st.session_state.get("user_input", ""),
    placeholder="e.g. Calculate 15% tip on $85.50, or run Python code to sort a list...",
    key="input_box",
)
if "user_input" in st.session_state:
    del st.session_state["user_input"]

send_col, clear_col = st.columns([2, 1])
with send_col:
    send = st.button("🚀 Send to Agent", use_container_width=True)
with clear_col:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Run Agent ─────────────────────────────────────────────────────────────────
if send and user_input.strip():
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ Add your Groq API key in the sidebar.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("🤖 Agent thinking..."):
        try:
            from src.agents.agent import run_agent
            state = run_agent(user_input)

            answer = state.get("final_answer", "No response generated")
            tools_used = list(set(state.get("tool_calls_made", [])))
            iterations = state.get("iterations", 1)

            st.session_state.chat_history.append({
                "role": "agent",
                "content": answer,
                "tools_used": tools_used,
                "iterations": iterations,
            })

            st.rerun()

        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            st.exception(e)

# ── Docker Instructions ───────────────────────────────────────────────────────
with st.expander("🐳 How to run with Docker", expanded=False):
    st.markdown("### Run this agent anywhere with Docker")

    tab1, tab2, tab3 = st.tabs(["🐳 Docker", "🔧 docker-compose", "☁️ Cloud Deploy"])

    with tab1:
        st.markdown("**Build and run:**")
        st.code("""# Build the image
docker build -t dockerized-ai-agent .

# Run the container
docker run -p 8501:8501 \\
  -e GROQ_API_KEY=gsk_your_key_here \\
  dockerized-ai-agent

# Open http://localhost:8501""", language="bash")

    with tab2:
        st.markdown("**Using docker-compose (recommended):**")
        st.code("""# Add your key to .env file:
# GROQ_API_KEY=gsk_your_key_here

docker-compose up

# Open http://localhost:8501""", language="bash")

    with tab3:
        st.markdown("**Deploy to any cloud:**")
        st.code("""# Render.com
docker build -t ai-agent .
docker tag ai-agent registry.render.com/your-app
docker push registry.render.com/your-app

# Railway
railway up

# Fly.io
fly deploy""", language="bash")

# ── Landing if no chat ────────────────────────────────────────────────────────
if not st.session_state.chat_history and not send:
    st.markdown("---")
    st.markdown("### 🔧 Agent Tools")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="card">
<b>🧮 Calculator</b><br>
<span style="color:#888">Evaluates math expressions using Python's math library. Supports trig, sqrt, log, and more.</span><br><br>
<code style="color:#0db7ed">sqrt(144) + sin(3.14) * 2</code>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card">
<b>📊 Text Analyzer</b><br>
<span style="color:#888">Analyzes any text: word count, sentence count, top words, reading time.</span><br><br>
<code style="color:#0db7ed">Analyze: "Docker containers are..."</code>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card">
<b>🐍 Python Runner</b><br>
<span style="color:#888">Writes and executes Python code in a sandboxed subprocess. Returns output.</span><br><br>
<code style="color:#0db7ed">Generate fibonacci numbers up to 200</code>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card">
<b>📚 Knowledge Base</b><br>
<span style="color:#888">Searches local knowledge about Docker, LangGraph, Groq, Streamlit, FastAPI, and AI agents.</span><br><br>
<code style="color:#0db7ed">What is RAG and how does it work?</code>
</div>""", unsafe_allow_html=True)
