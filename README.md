# LLM Eval Framework

An end-to-end benchmarking framework that evaluates LLMs on code generation tasks.

## What it does
- Runs 4 models (GPT-4o, Gemini 1.5, LLaMA 3, Mistral) on 300 coding problems
- Scores outputs using test execution, semantic similarity, and LLM-as-judge
- Clusters failure patterns to surface where each model breaks down
- Displays results on a live leaderboard dashboard

## Stack
Python · FastAPI · React · SQLite · scikit-learn

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```
