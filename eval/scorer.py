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
def score_by_execution(model_output, test_cases):
    passed = 0
    total = len(test_cases)

    for test in test_cases:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                test_code = f"""
{model_output}

result = None
try:
    inp = {repr(test['input'])}
    if isinstance(inp, list):
        result = locals()[list(filter(lambda x: x.startswith('def '), {repr(model_output)}.split('\\n')))[0].split('(')[0].replace('def ', '')](*inp)
    else:
        result = locals()[list(filter(lambda x: x.startswith('def '), {repr(model_output)}.split('\\n')))[0].split('(')[0].replace('def ', '')](inp)
except Exception as e:
    result = None

print(result)
"""
                f.write(test_code)
                tmp_path = f.name

            proc = subprocess.run(#opens a brand new Python process and runs this file for safety in case of crash.
                ["python", tmp_path],
                capture_output=True, text=True, timeout=5
            )
            output = proc.stdout.strip()
            expected = str(test["expected"])

            if output == expected:
                passed += 1

        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    return passed / total if total > 0 else 0.0

#This measures how similar the model's code is to the reference solution in meaning.
def score_by_similarity(model_output, ground_truth):
    emb1 = model.encode([model_output])
    emb2 = model.encode([ground_truth])#takes code as text and converts it into a list of 384 numbers representing meaning of code, vector.
    score = cosine_similarity(emb1, emb2)[0][0]#measures the angle between two vectors.
    return float(score)


def score_result(result):
    exec_score = score_by_execution(
        result["model_output"],
        result["test_cases"]
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