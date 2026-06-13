#problems from MBPP and HumanEval

import json
import os
from datasets import load_dataset

def load_mbpp(num_problems=20):
    print("Loading MBPP...")
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    
    problems = []
    count = 0
    
    for item in dataset:
        if count >= num_problems:
            break
            
        try:
            test_cases = []
            for test in item["test_list"]:
                test = test.strip().replace("assert ", "")
                parts = test.split("==", 1)
                if len(parts) != 2:
                    continue
                    
                call = parts[0].strip()
                expected = parts[1].strip()
                
                inner = call[call.index("(")+1 : call.rindex(")")]
                
                try:
                    parsed_input = eval(inner)
                    parsed_expected = eval(expected)
                except:
                    continue
                
                if not isinstance(parsed_input, tuple):
                    parsed_input = [parsed_input]
                else:
                    parsed_input = list(parsed_input)
                    
                test_cases.append({
                    "input": parsed_input,
                    "expected": parsed_expected
                })
            
            if len(test_cases) < 2:
                continue
            
            # sanitized version uses "prompt" not "text"
            code_lines = item["code"].strip().split("\n")
            func_line = next(
                (l for l in code_lines if l.startswith("def ")), None
            )
            if not func_line:
                continue
                
            function_name = func_line.split("(")[0].replace("def ", "").strip()
            
            problem = {
                "id": f"mbpp_{item['task_id']:03d}",
                "function_name": function_name,
                "prompt": f"Write a Python function called `{function_name}` that {item['prompt'].lower()}",
                "ground_truth": item["code"].strip(),
                "test_cases": test_cases[:4],
                "difficulty": "medium",
                "tags": ["mbpp"],
                "source": "mbpp"
            }
            
            problems.append(problem)
            count += 1
            print(f"  ✓ {problem['id']}: {function_name}")
            
        except Exception as e:
            print(f"  ✗ skipped item {item.get('task_id', '?')}: {e}")
            continue
    
    return problems


def load_humaneval(num_problems=15):
    print("\nLoading HumanEval...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    problems = []
    count = 0
    
    for item in dataset:
        if count >= num_problems:
            break
        
        try:
            function_name = item["entry_point"]
            
            # HumanEval tests are assert-based inside a check() function
            # store them as raw assert strings and handle them 
            # separately in the execution scorer
            raw_test = item["test"]
            
            # parse assert lines from the check function
            test_cases = []
            for line in raw_test.split("\n"):
                line = line.strip()
                if not line.startswith("assert"):
                    continue
                    
                line = line.replace("assert ", "")
                
                # handle both "func(x) == y" and "func(x, y) == z"
                if "==" not in line:
                    continue
                    
                parts = line.split("==", 1)
                call = parts[0].strip()
                expected = parts[1].strip()
                
                inner = call[call.index("(")+1 : call.rindex(")")]
                
                try:
                    parsed_input = eval(inner)
                    parsed_expected = eval(expected)
                except:
                    continue
                
                if not isinstance(parsed_input, tuple):
                    parsed_input = [parsed_input]
                else:
                    parsed_input = list(parsed_input)
                
                test_cases.append({
                    "input": parsed_input,
                    "expected": parsed_expected
                })
            
            if len(test_cases) < 2:
                continue
            
            # HumanEval prompt includes the function signature
            # We clean it up into a plain English description
            docstring_start = item["prompt"].find('"""')
            docstring_end = item["prompt"].find('"""', docstring_start + 3)
            
            if docstring_start != -1 and docstring_end != -1:
                description = item["prompt"][docstring_start+3:docstring_end].strip()
            else:
                description = item["prompt"].strip()
            
            problem = {
                "id": f"humaneval_{item['task_id'].replace('/', '_')}",
                "function_name": function_name,
                "prompt": f"Write a Python function called `{function_name}`. {description}",
                "ground_truth": item["prompt"] + item["canonical_solution"],
                "test_cases": test_cases[:4],
                "difficulty": "medium",
                "tags": ["humaneval"],
                "source": "humaneval"
            }
            
            problems.append(problem)
            count += 1
            print(f"  ✓ {problem['id']}: {function_name}")
            
        except Exception as e:
            print(f"  ✗ skipped {item.get('task_id', '?')}: {e}")
            continue
    
    return problems


def merge_with_existing(new_problems):
    existing_path = "data/problems/sample_problems.json"
    
    with open(existing_path, "r") as f:
        existing = json.load(f)
    
    print(f"\nExisting problems: {len(existing)}")
    print(f"New problems: {len(new_problems)}")
    
    merged = existing + new_problems
    print(f"Total after merge: {len(merged)}")
    
    return merged

if __name__ == "__main__":
    import sys
    
    if "--merge" in sys.argv:
        review_path = "data/problems/external_problems.json"
        with open(review_path, "r") as f:
            new_problems = json.load(f)
        
        merged = merge_with_existing(new_problems)
        
        output_path = "data/problems/sample_problems.json"
        with open(output_path, "w") as f:
            json.dump(merged, f, indent=2)
        
        print(f"✓ Merged successfully. Total problems: {len(merged)}")
        print(f"Saved to {output_path}")
    
    else:
        mbpp_problems = load_mbpp(num_problems=20)
        humaneval_problems = load_humaneval(num_problems=15)
        
        all_new = mbpp_problems + humaneval_problems
        
        review_path = "data/problems/external_problems.json"
        with open(review_path, "w") as f:
            json.dump(all_new, f, indent=2)
        
        print(f"\n✓ Saved {len(all_new)} problems to {review_path}")
        print("Review this file before merging into the main dataset.")
        print("\nTo merge, run:")
        print("  python data/load_datasets.py --merge")