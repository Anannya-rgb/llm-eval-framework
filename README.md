# LLM Eval Framework

An end-to-end benchmarking framework that evaluates LLMs on code generation tasks using three scoring methods.

## What it does
- Runs LLMs on a curated set of coding problems
- Scores outputs three ways: test execution, semantic similarity, and LLM-as-judge
- Uses Gemini 2.5 Flash as an independent judge to avoid self-evaluation bias
- Surfaces failure patterns and quality insights across models

## Scoring methodology
| Method | Weight | What it measures |
|---|---|---|
| Test execution | 50% | Does the code actually work? Runs code against test cases |
| LLM-as-judge | 30% | Code quality, efficiency, edge case handling |
| Semantic similarity | 20% | How similar the solution is to the reference in meaning |

## Key findings so far
- LLaMA 3.3 70B scores perfectly on correctness for easy-medium problems
- Recursive solutions pass correctness tests but are flagged by the judge for inefficiency
- - Dataset quality matters — discovered and fixed an incorrect expected answer in prob_004 where we wrote [2,3] instead of the correct [0,2], highlighting how benchmark bugs can silently corrupt evaluation results
- LLM-as-judge (Gemini) provides more technically precise feedback than execution scoring alone, correctly identifying time complexity tradeoffs like O(2^n) vs O(n).

## Known limitations
- Test cases evaluate correctness only, not efficiency — a recursive fibonacci passes the same as an iterative one
- Currently evaluating 5 problems — expanding to 300 is the next milestone
- Judge model (Gemini) and evaluated model (LLaMA) use different providers to avoid self-evaluation bias

## Stack
Python · LiteLLM · Groq · Google AI Studio · sentence-transformers · scikit-learn · FastAPI (coming) · React (coming)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

## Run the pipeline
```bash
python eval/runner.py     # run model on problems
python eval/scorer.py     # score with execution + similarity
python eval/judge.py      # score with LLM-as-judge
```

## Project structure
```
llm-eval-framework/
├── data/
│   ├── problems/         # benchmark dataset
│   └── results/          # scored outputs per model
├── eval/
│   ├── runner.py         # sends problems to models
│   ├── scorer.py         # execution + similarity scoring
│   └── judge.py          # LLM-as-judge scoring
├── analysis/             # failure clustering (coming)
└── dashboard/            # leaderboard UI (coming)
```