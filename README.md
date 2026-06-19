# Dockerized AI Agent

A LangGraph + Llama 3 agent packaged with Docker — runs the same way everywhere.

## The Problem This Solves

AI demos often break when shared due to different Python versions, missing dependencies, or environment issues. Docker fixes this — one image runs identically on any machine, any cloud, any OS.

## Run in 3 Commands

git clone https://github.com/Rohan-jpg-ux/dockerized-ai-agent.git
cd dockerized-ai-agent
cp .env.example .env
docker-compose up

Open http://localhost:8501 — done.

## Agent Tools

- Calculator — Math expressions
- Python Runner — Execute Python code
- Text Analyzer — Word/sentence stats
- Knowledge Base — AI/tech topics

## Tests

pytest tests/ -v

Built with Docker + LangGraph + Llama 3.3 + Streamlit
