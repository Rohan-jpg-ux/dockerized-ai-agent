"""
Multi-Tool AI Agent — Dockerized
LangGraph + Llama 3 via Groq
Tools: web search simulation, calculator, code runner, text analyzer
"""

import os
import re
import json
import math
import subprocess
import tempfile
import sys
from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Example: '2 + 2', 'sqrt(16)', 'sin(3.14)'"""
    try:
        safe_expr = expression.replace("^", "**")
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
        result = eval(safe_expr, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def run_python(code: str) -> str:
    """Execute a Python code snippet and return the output. Good for data processing, algorithms."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=15
        )
        os.unlink(path)
        output = result.stdout or result.stderr
        return output[:2000] if output else "No output"
    except subprocess.TimeoutExpired:
        return "Code execution timed out (15s limit)"
    except Exception as e:
        return f"Execution error: {str(e)}"


@tool
def text_analyzer(text: str) -> str:
    """Analyze text: word count, sentence count, most common words, readability stats."""
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    word_freq = {}
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been"}
    for w in words:
        w = re.sub(r'[^a-z]', '', w)
        if w and w not in stop_words:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
    return json.dumps({
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_words": len(set(words)),
        "avg_word_length": round(avg_word_len, 2),
        "top_5_words": top_words,
        "estimated_reading_time_sec": round(len(words) / 3.5),
    }, indent=2)


@tool
def knowledge_base(query: str) -> str:
    """Search a local knowledge base about AI, Docker, and programming topics."""
    kb = {
        "docker": "Docker is a platform for containerizing applications. Key commands: docker build, docker run, docker-compose up. Images are blueprints; containers are running instances.",
        "langgraph": "LangGraph is a framework for building stateful, multi-actor AI applications using graph-based workflows. Nodes are functions, edges define flow.",
        "llama": "Llama 3 is Meta's open-source LLM. Llama 3.3 70B is available via Groq API with ultra-fast inference speeds.",
        "groq": "Groq provides ultra-fast LLM inference. Free tier available at console.groq.com. Supports Llama 3, Mixtral, and Gemma models.",
        "streamlit": "Streamlit is a Python framework for building data apps. Run with: streamlit run app.py. Deploys free on share.streamlit.io.",
        "fastapi": "FastAPI is a modern Python web framework. Fast, async, automatic Swagger docs. Run with: uvicorn main:app --reload",
        "python": "Python is a high-level programming language known for simplicity. Used in AI/ML, web dev, data science, and automation.",
        "agent": "An AI agent is a system that perceives its environment, makes decisions, and takes actions to achieve goals. LangGraph helps build reliable agents.",
        "rag": "Retrieval Augmented Generation (RAG) combines a vector database with an LLM to answer questions from your own documents.",
        "default": "I have knowledge about Docker, LangGraph, Llama, Groq, Streamlit, FastAPI, Python, AI agents, and RAG systems.",
    }
    query_lower = query.lower()
    for key, value in kb.items():
        if key in query_lower:
            return value
    return kb["default"]


TOOLS = [calculator, run_python, text_analyzer, knowledge_base]
TOOL_MAP = {t.name: t for t in TOOLS}


# ─── State ────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    tool_calls_made: List[str]
    final_answer: Optional[str]
    iterations: int


# ─── LLM with tools ───────────────────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2048,
        api_key=api_key,
    )
    return llm.bind_tools(TOOLS)


# ─── Nodes ────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Main agent node — calls LLM with tools"""
    logger.info(f"🤖 Agent thinking... (iteration {state['iterations']})")
    llm = get_llm()

    system = """You are a helpful AI assistant with access to these tools:
- calculator: for math expressions
- run_python: to execute Python code
- text_analyzer: to analyze text statistics
- knowledge_base: to look up AI/tech topics

Use tools when they would help answer accurately. Be concise and helpful."""

    messages = [SystemMessage(content=system)] + state["messages"]
    response = llm.invoke(messages)

    state["messages"] = state["messages"] + [response]
    state["iterations"] += 1
    return state


def tool_node(state: AgentState) -> AgentState:
    """Execute tool calls from the agent"""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info(f"🔧 Calling tool: {tool_name}")

        if tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(tool_args)
                state["tool_calls_made"].append(tool_name)
            except Exception as e:
                result = f"Tool error: {str(e)}"
        else:
            result = f"Unknown tool: {tool_name}"

        tool_results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        ))

    state["messages"] = state["messages"] + tool_results
    return state


def finalize_node(state: AgentState) -> AgentState:
    """Extract final answer from last message"""
    last = state["messages"][-1]
    if hasattr(last, "content") and last.content:
        state["final_answer"] = last.content
    else:
        state["final_answer"] = "I've processed your request."
    logger.info("✅ Agent complete")
    return state


# ─── Routing ──────────────────────────────────────────────────────────────────

def should_use_tools(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls and state["iterations"] < 5:
        return "tools"
    return "finalize"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_use_tools, {
        "tools": "tools",
        "finalize": "finalize",
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_agent(user_input: str) -> AgentState:
    graph = build_agent()
    initial: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "user_input": user_input,
        "tool_calls_made": [],
        "final_answer": None,
        "iterations": 0,
    }
    return graph.invoke(initial)
