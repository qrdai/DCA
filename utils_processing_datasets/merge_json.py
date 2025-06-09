import os
import json


if __name__ == "__main__":
    basedir = "../context-cite/qualitative_results"
    paths_list = [
        [
            os.path.join(basedir, "gpt-4-1106-preview-size5-seed42-ablations256-nonrag.json"),
            os.path.join(basedir, "IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106-size5-seed42-ablations256-nonrag.json"),
        ],
        [
            os.path.join(basedir, "gpt-4-1106-preview-size5-seed42-ablations256-rag.json"),
            os.path.join(basedir, "IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106-size5-seed42-ablations256-rag.json"),
        ],
    ]
    
    for paths in paths_list:
        all_examples = []
        rag_flag = True
        for path in paths:
            with open(path, 'r') as fin:
                all_examples.extend(json.load(fin))
        if "-nonrag" in path:
            rag_flag = False
        assert len(all_examples) == 10
        
        with open(os.path.join(
            basedir, 
            f"size{len(all_examples)}-seed42-ablations256-rag.json" if rag_flag else f"size{len(all_examples)}-seed42-ablations256-nonrag.json",
        ), 'w') as fout:
            json.dump(all_examples, fout, indent=4)
        