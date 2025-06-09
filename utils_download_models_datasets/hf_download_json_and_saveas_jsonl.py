import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from datasets import load_dataset


hf_home = "/root/autodl-tmp/.cache/huggingface"   # autodl
dataset_cache_dir = f"{hf_home}/datasets"
dataset_local_dir = f"/root/autodl-tmp/data"

repo_split_map = {
    # training
    # "sft-data-selection/3x-corpus-llama": "train",
    # "sft-data-selection/3x-corpus-mistral": "train",
    # "sft-data-selection/3x-corpus-qwen2.5-7b": "train",
    "TsinghuaC3I/UltraMedical": "train",
    
    
    # validation/eval
    # "evoeval/EvoEval_combine": "test",
    # "BAAI/TACO": "test",
    # "KbsdJames/Omni-MATH": "test",
    # "Post-training-Data-Flywheel/AutoIF-instruct-61k": "train",
    # "GBaker/MedQA-USMLE-4-options": "train",
    # "GBaker/MedQA-USMLE-4-options": "test",
}


def convert_solutions_type(jsonl_path):
    # Step 1: Read the existing data
    with open(jsonl_path, 'r') as fin:
        lines = [json.loads(line) for line in fin]
    print(f"len(lines): {len(lines)}\n")

    modified_lines = []

    # Step 2: Process each line
    for idx, line in enumerate(lines):
        try:
            assert "solutions" in line.keys() and type(line["solutions"]) == str
            line['solutions'] = json.loads(line['solutions'])[0]    # only use the first ground-truth solution for each problem
            # assert type(line["solutions"]) == list
            modified_lines.append(line)
        except Exception as e:
            print(f"For index: {idx}, Caught an exception: {e}")

    # Step 3: Write the modified data back to the same file
    print(f"len(modified_lines): {len(modified_lines)}\n")
    with open(jsonl_path, 'w') as fout:
        for line in modified_lines:
            fout.write(json.dumps(line) + '\n')


for repo, split in repo_split_map.items():
    dataset = load_dataset(
        path=repo, 
        split=split,
        cache_dir=dataset_cache_dir, 
        token=True,
        trust_remote_code=True
    )
    
    # 1. Basic inspection
    print(type(dataset))    # , type(dataset[split])
    print(dataset[:1])   # dataset[split][:1]
    
    basename = os.path.basename(repo)
    output_dir = os.path.join(dataset_local_dir, basename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Solution 2: directly use Dataset.to_json
    jsonl_path = os.path.join(output_dir, f'{basename}-{split}.jsonl')
    dataset.to_json(jsonl_path)    # save as .jsonl instead of .json by default; dataset[split].to_json
    print(f"Successfully downloaded and saved `{basename}-{split}.jsonl`!\n")
    
    
    # # 2. TACO-specific: "question" and "solutions"
    # dataset = dataset[:1]
    # for idx, (question, solutions) in enumerate(zip(dataset["question"], dataset["solutions"])):
    #     print(f"type(solutions): {type(solutions)}")
    #     solutions = json.loads(solutions)
    #     print(f"type(solutions): {type(solutions)}")
    #     print(f"INDEX: {idx}")
    #     print(f"QUESTION: {question}")
    #     print(f"len(solutions): {len(solutions)}")
    #     print(f"solutions[0]: {solutions[0]}")
    #     print(f"SOLUTIONS: {solutions}")
    #     print('\n')
    
    
    # # 3. TACO-specific: convert "solutions" in each entry of TACO.jsonl from str to list
    # convert_solutions_type(jsonl_path)