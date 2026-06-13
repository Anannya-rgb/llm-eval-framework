#measures quality through an external AI judge - LLM-as-Judge
import os
import json
from dotenv import load_dotenv
import litellm
from litellm import completion

load_dotenv()

litellm.api_key = os.getenv("GROQ_API_KEY")

JUDGE_PROMPT = """You are an expert Python programmer and code reviewer.

You will be given:
1. A coding problem
2. A reference solution (correct answer)
3. A model's solution (what we are evaluating)

Your job is to evaluate the model's solution and give it a score from 0 to 10.

Scoring guide:
- 10: Perfect. Correct, efficient, clean, handles edge cases
- 7-9: Good. Correct but minor issues (style, slight inefficiency)
- 4-6: Partial. Right idea but has bugs or misses edge cases
- 1-3: Poor. Wrong approach or mostly incorrect
- 0: Completely wrong or empty

You MUST respond with ONLY a JSON object in this exact format:
{{
  "score": <number from 0 to 10>,
  "reason": "<one sentence explaining the score>"
}}

No other text. Just the JSON.

---

Problem:
{problem}

Reference solution:
{reference}

Model's solution:
{solution}
"""

def judge_solution(problem, reference, model_output):
    prompt = JUDGE_PROMPT.format(
        problem=problem,
        reference=reference,
        solution=model_output
    )

    try:
        response = completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        raw = response.choices[0].message.content.strip()

        raw = raw.strip("```json").strip("```").strip()#to strip incase of anything other than json
        result = json.loads(raw)

        score = float(result["score"]) / 10.0 #score normalization - range of 0 to 1.0
        reason = result["reason"]

        return {
            "judge_score": round(score, 3),
            "judge_reason": reason
        }

    except Exception as e:
        print(f"Judge failed: {e}")
        return {
            "judge_score": 0.0,
            "judge_reason": "Judge failed to evaluate"
        }


def judge_all(scored_path):
    with open(scored_path, "r") as f:
        results = json.load(f)

    judged = []
    for i, result in enumerate(results):
        print(f"Judging {i+1}/{len(results)}: {result['problem_id']}")

        judgment = judge_solution(
            problem=result["prompt"],
            reference=result["ground_truth"],
            model_output=result["model_output"]
        )

        final_combined = (
            0.5 * result["exec_score"] +
            0.2 * result["similarity_score"] +
            0.3 * judgment["judge_score"]
        )

        judged.append({
            **result,
            **judgment,
            "final_combined_score": round(final_combined, 3)
        })

    return judged


if __name__ == "__main__":
    judged = judge_all("data/results/llama3_scored.json")

    output_path = "data/results/llama3_judged.json"
    with open(output_path, "w") as f:
        json.dump(judged, f, indent=2)

    print(f"\nDone. Judged results saved to {output_path}")
    print("\n--- Final Scores ---")
    for r in judged:
        print(f"{r['problem_id']} | final: {r['final_combined_score']} | reason: {r['judge_reason']}")