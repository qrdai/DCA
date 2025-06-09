import os
import json
import torch


if __name__ == "__main__":
    rag_flag_list = [
        "nonrag",
        "rag",
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for rag_flag in rag_flag_list:
        ## 1. load AM tensor
        AM_path = os.path.join(
            "../attribution_matrix/LESS_UM-llama3.1",
            "step2_adam_valsize10_Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3",
            f"medqa_{rag_flag}",
            "UltraMedical-train-Exam-50k_instance_AM.pt",
        ) # 50K * 10
        instance_AM = torch.load(AM_path, map_location=device)
        
        
        ## 2. load test dataset
        json_path = os.path.join(
            "../context-cite/qualitative_results",
            f"size10-seed42-ablations256-{rag_flag}.json",
        )
        with open(json_path, 'r') as fin:
            test_json = json.load(fin)
        assert len(test_json) == 10
        
        
        ## 3. load training dataset
        jsonl_path = os.path.join(
            "../data/UltraMedical",
            "UltraMedical-train-Exam-50k.jsonl",
        )
        with open(jsonl_path, 'r') as fin:
            training_jsonls = [json.loads(line) for line in fin]
        assert len(training_jsonls) == 50_000
        
        
        ## 4. iterate through columns of AM
        # to find highest-influence training examples 
        # for each test sample
        budget = 5
        outdir = os.path.join(
            "./qualitative_results",
            f"medqa_{rag_flag}",
        )
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # ------------------------------------------------------------------
        # NEW CODE: select top-k influential training examples for each test
        # ------------------------------------------------------------------
        # Work on CPU for convenience (avoids GPU->CPU copy inside the loop)
        instance_AM_cpu = instance_AM.to("cpu") if instance_AM.is_cuda else instance_AM

        for col_idx, test_example in enumerate(test_json):
            # 1. Get influence scores for current test example (column)
            col_scores = instance_AM_cpu[:, col_idx]

            # 2. Identify top-`budget` training examples (largest scores)
            top_vals, top_indices = torch.topk(col_scores, budget, largest=True, sorted=True)

            # 3. Collect the corresponding training examples with influence values
            top_examples = []
            for infl_val, train_idx in zip(top_vals.tolist(), top_indices.tolist()):
                train_example = training_jsonls[train_idx].copy()  # do not mutate original
                train_example["influence"] = infl_val
                top_examples.append(train_example)

            # 4. Merge with the test example and write to disk
            output_record = test_example.copy()  # include all original test fields
            output_record[f"top_{budget}_da_scores"] = top_examples

            outfile = os.path.join(outdir, f"idx{test_example['question_idx']}.json")
            with open(outfile, "w", encoding="utf-8") as fout:
                json.dump(output_record, fout, indent=4)

        print(f"[✓] Saved qualitative attribution results for '{rag_flag}' to {outdir}")
