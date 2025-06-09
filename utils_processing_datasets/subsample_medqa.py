import os
import json
import random


if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "MedQA-USMLE-4-options")
    jsonname_modelname_mapping = {
        "gpt-4-1106-preview_subset=0-1272_seed=24_trylim=5_testset.json": "gpt-4-1106-preview",
        "IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106_subset=0-1272_seed=24_trylim=5_testset.json": "IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106",
    }
    budget = 5
    seed = 42

    random.seed(seed)

    for jsonname, modelname in jsonname_modelname_mapping.items():
        json_path = os.path.join(data_dir, jsonname)

        # Load the JSON data
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Sample data if the budget is less than or equal to available examples
        if len(data) >= budget:
            sampled_data = sorted(random.sample(data, budget), key=lambda x: int(x.get('question_idx', 0)))
        else:
            sampled_data = data

        # Save sampled data
        output_filename = f"{modelname}-size{budget}-seed{seed}.json"
        output_path = os.path.join(data_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(sampled_data, file, indent=4)

        print(f"Saved sampled data to {output_path}")