import os
import json


def reorder_medqa(data_dir, jsonname_list):
    """
    Load each JSON file from data_dir with names in jsonname_list,
    sort its list of dict entries by 'question_idx' (as integer),
    and overwrite the original file with the sorted list.
    """
    for jsonname in jsonname_list:
        json_path = os.path.join(data_dir, jsonname)
        # Check if file exists
        if not os.path.isfile(json_path):
            print(f"File not found: {json_path}")
            continue

        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {json_path}: {e}")
                continue

        if not isinstance(data, list):
            print(f"Expected list in {json_path}, got {type(data)}. Skipping.")
            continue

        # Sort entries by 'question_idx'
        try:
            sorted_data = sorted(data, key=lambda x: int(x.get('question_idx', 0)))
        except (ValueError, TypeError) as e:
            print(f"Error sorting data in {json_path}: {e}")
            continue

        # Overwrite original file with sorted data
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_data, f, indent=4, ensure_ascii=False)

        print(f"Reordered and saved: {json_path}")


if __name__ == "__main__":
    # Directory containing MedQA JSON files
    data_dir = os.path.join("..", "data", "MedQA-USMLE-4-options")
    # List of JSON filenames to reorder
    jsonname_list = [
        "gpt-4-1106-preview_subset=0-1272_seed=24_trylim=5_testset.json",
        "IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106_subset=0-1272_seed=24_trylim=5_testset.json",
    ]

    reorder_medqa(data_dir, jsonname_list)
