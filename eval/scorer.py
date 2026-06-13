import json
import os
import subprocess
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")#pre-trained neural network that has read billions of lines of text and code.certain things mean similar things even if they look different.


#this runs the model's code and checks if it produces the right answer.
def score_by_execution(model_output, test_cases, function_name):
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                inp = test["input"]
                expected = test["expected"]

                if isinstance(inp, list):
                    args = ", ".join(repr(a) for a in inp)
                else:
                    args = repr(inp)

                test_code = f"""{model_output}

try:
    result = {function_name}({args})
    print(repr(result))
except Exception as e:
    print("ERROR:", e)
"""
                f.write(test_code)
                tmp_path = f.name

            proc = subprocess.run(
                ["python", tmp_path],
                capture_output=True, text=True, timeout=5
            )
            if proc.stderr: 
                print(f"  STDERR: {proc.stderr.strip()}")

            output = proc.stdout.strip()
            expected_repr = repr(expected)

            if output == expected_repr:
                passed += 1
            else:
                print(f"  FAIL | expected: {expected_repr} | got: {output}")

        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return passed / total if total > 0 else 0.0

#This measures how similar the model's code looks to the reference solution in meaning.It never runs the code.
def score_by_similarity(model_output, ground_truth):
    emb1 = model.encode([model_output])
    emb2 = model.encode([ground_truth])#takes code as text and converts it into a list of 384 numbers representing meaning of code, vector.
    score = cosine_similarity(emb1, emb2)[0][0]#measures the angle between two vectors.
    return float(score)

def score_result(result):
    exec_score = score_by_execution(
        result["model_output"],
        result["test_cases"],
        result["function_name"]   
    )

    sim_score = score_by_similarity(
        result["model_output"],
        result["ground_truth"]
    )

    final_score = (0.7 * exec_score) + (0.3 * sim_score)

    return {
        **result,
        "exec_score": round(exec_score, 3),
        "similarity_score": round(sim_score, 3),
        "final_score": round(final_score, 3)
    }


def score_all(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)

    scored = []
    for i, result in enumerate(results):
        print(f"Scoring {i+1}/{len(results)}: {result['problem_id']}")
        scored.append(score_result(result))

    return scored


if __name__ == "__main__":
    scored = score_all("data/results/llama3_results.json")

    output_path = "data/results/llama3_scored.json"
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"\nDone. Scored results saved to {output_path}")

    print("\n--- Summary ---")
    for r in scored:
        print(f"{r['problem_id']} | exec: {r['exec_score']} | similarity: {r['similarity_score']} | final: {r['final_score']}")