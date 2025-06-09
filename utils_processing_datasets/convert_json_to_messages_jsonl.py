import os
import json


if __name__ == "__main__":
    basedir = "../context-cite/qualitative_results"
    path_list = [
        os.path.join(basedir, "size10-seed42-ablations256-nonrag.json"),
        os.path.join(basedir, "size10-seed42-ablations256-rag.json"),
    ]
    
    for path in path_list:
        with open(path, 'r') as fin:
            all_examples = json.load(fin)
        
        outpath = path.replace(".json", ".jsonl")
        with open(outpath, 'w') as fout:
            for example in all_examples:
                stripped_response = example["response"].split("<|end_of_text|>")[0]
                new_example = {
                    "question_idx": example["question_idx"],
                    "answer_letter": example["answer_letter"],
                    "messages": [
                        {"role": "user", "content": f"Context: {example['context']}\n\nQuery: {example['query']}"},
                        {"role": "assistant", "content": stripped_response},
                    ]
                }
                fout.write(json.dumps(new_example) + '\n')