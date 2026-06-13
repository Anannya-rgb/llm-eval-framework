import os
import json
from dotenv import load_dotenv

load_dotenv() #opens .env file and reads it 

import litellm #talks to all AI in a consistent manner
from litellm import completion#sends message to a model and gets the response back
litellm.api_key = os.getenv("GROQ_API_KEY")


#the function opens the json file of problems and loads them into Python as a list. 
def load_problems(path="data/problems/sample_problems.json"):
    with open(path, "r") as f:
        return json.load(f)#converts the raw text file into a Python list of dictionaries that helps to loop through.

def build_prompt(problem):
    return f"""You are an expert Python programmer.

Solve the following problem by writing ONLY the Python function.
No explanations, no markdown, no extra text. Just the function.

Problem:
{problem['prompt']}
"""

def run_model(problem, model_name):
    prompt = build_prompt(problem)#wraps the problem in instructions

    response = completion( #sends the prompt to the model and waits for a response
        model=model_name,#tells litellm which model to use 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 #since we do not want creative variations
    )

    output = response.choices[0].message.content.strip()
    #strips the markdown code fences out of the model's response before we store it anywhere. 
    # Clean code goes in, clean code comes out.
    output = output.replace("```python", "").replace("```", "").strip()

#This packages everything into one dictionary per problem — 
# the original question, the model's answer, the correct answer, the test cases.
#We keep all of this together because the scorer needs all of it later. 
    return {
        "problem_id": problem["id"],
        "function_name": problem["function_name"], 
        "model": model_name,
        "prompt": problem["prompt"],
        "model_output": output,
        "ground_truth": problem["ground_truth"],
        "test_cases": problem["test_cases"],
        "difficulty": problem["difficulty"],
        "tags": problem["tags"]
    }

def run_all(model_name, problems_path="data/problems/sample_problems.json"):
    problems = load_problems(problems_path)
    results = []

    for i, problem in enumerate(problems): #enumerate gives both the index i and the item problem at the same time
        print(f"Running problem {i+1}/{len(problems)}: {problem['id']}")
        result = run_model(problem, model_name)
        results.append(result)

    return results

#this means "only run this block if you're running this file directly." 
# If another file imports runner.py, this block won't execute. 
if __name__ == "__main__":
    results = run_all(model_name="groq/llama-3.3-70b-versatile")

    os.makedirs("data/results", exist_ok=True)# creates the data/results folder if it doesn't exist.
    output_path = "data/results/llama3_results.json"

    with open(output_path, "w") as f:#saves the results list to a JSON file.
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")