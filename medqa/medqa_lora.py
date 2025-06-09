import os
import json
import argparse
import re
import nltk
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from context_cite import ContextCiter
model_cache_dir = "/root/autodl-tmp/.cache/huggingface/transformers"

dtype_dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": "auto", # default to model.config.torch_dtype, which is recorded in config.json in its huggingface repo
    None: None, # default to torch.float32 for all pytorch-based packages, including huggingface transformers
}


def load_model(model_name_or_path: str,
               torch_dtype: Any) -> Any:
    """
    Load a model from a given model name or path.

    Args:
        model_name_or_path (str): The name or path of the model.
        torch_dtype (Any, optional): The torch data type: torch.float32 or torch.bfloat16

    Returns:
        Any: The loaded model.
    """

    is_peft = os.path.exists(os.path.join(
        model_name_or_path, "adapter_config.json"))
    if is_peft:
        # load this way to make sure that optimizer states match the model structure
        config = LoraConfig.from_pretrained(model_name_or_path) # no need for `cache_dir=model_cache_dir` here, since directly load from `model_name_or_path` (though adding `cache_dir` won't affect the loading of `model_name_or_path` from /out, since the former will be overruled by the latter)
        # also, `LoraConfig.from_pretrained` requires `adapter_config.json` to be present in `model_name_or_path` or the huggingface repo, but the cached `meta-llama/Llama-2-7b-hf` doesn't contain that
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            torch_dtype=torch_dtype, 
            device_map="auto",
            cache_dir=model_cache_dir   # only needs to be added here since `config.base_model_name_or_path` is the cached pretrained model downloaded from hf repo
        )
        model = PeftModel.from_pretrained(
            base_model, 
            model_name_or_path, # still load `adapter_config.json` from this path 
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            torch_dtype=torch_dtype, 
            device_map="auto",
            cache_dir=model_cache_dir,  # when directly loading pretrained models from huggingface
        )

    # for name, param in model.named_parameters():
    #     if 'lora' in name or 'Lora' in name:
    #         param.requires_grad = True
    
    return model


def extract_question_sentence(full_question: str) -> str:
    """
    Extracts the final question sentence from a full_question string in two steps:
    
    1. Sentence-splits with nltk.tokenize.sent_tokenize and takes the last sentence.
    2. Within that sentence, if there's any capitalized interrogative (What/Which/How/Where/When/Why/Who),
       extract from the *last* such word through the final question mark.
       Otherwise return the whole last sentence.
    """
    # 1) Sentence-split and grab the last one
    sentences = nltk.tokenize.sent_tokenize(full_question)
    question_sent = sentences[-1]
    
    # 2) Look for capital interrogatives and extract the last-match → '?'
    pattern = re.compile(
        r'\b(?:What|Which|How|Where|When|Why|Who)\b.*?\?', 
        flags=re.DOTALL
    )
    all_matches = pattern.findall(question_sent)
    if all_matches:
        # return the last match (in case multiple interrogatives appear)
        return all_matches[-1]
    
    # fallback: nothing to trim, return whole last sentence
    return question_sent


def parse_args():
    parser = argparse.ArgumentParser(description='Script for getting context attribution scores')
    parser.add_argument('--test_file', type=str, default="../data/MedQA-USMLE-4-options/gpt-4-1106-preview-size5-seed42.json")
    parser.add_argument('--model_path', type=str, default="../out/LESS_UM-llama3.1/Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3/checkpoint-388")
    parser.add_argument('--output_dir', type=str, default="./qualitative_results")
    parser.add_argument("--torch_dtype", type=str, choices=["float32", "bfloat16"])
    parser.add_argument("--rag", default=False, action="store_true")
    parser.add_argument('--num_ablations', type=int, default=64)
    
    args = parser.parse_args()
    assert ".json" in args.test_file
    print(args, '\n')
    return args


if __name__ == "__main__":
    ## 0. load args
    args = parse_args()
    rag_flag = "rag" if args.rag else "nonrag"
    torch_dtype = dtype_dict[args.torch_dtype]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    
    ## 1. load data
    with open(args.test_file, 'r') as fin:
        testset = json.load(fin)
    
    
    ## 2. load model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=model_cache_dir,  # when directly loading pretrained models from huggingface
    )
    print(f"tokenizer.pad_token: {tokenizer.pad_token}")
    print(f"tokenizer.chat_template:\n{tokenizer.chat_template}")
    model = load_model(args.model_path, torch_dtype)
    
    
    ## 3. iterative context-cite
    new_json_list = []
    for test_ques in testset:
        full_question = test_ques["question"].strip()
        
        # # old approach; sometimes may extract more than the last question
        # sentences = nltk.tokenize.sent_tokenize(full_question)
        # question_sent = sentences[-1]
        
        question_sent = extract_question_sentence(full_question)
        context = full_question[:-len(question_sent)].strip()
        
        if args.rag:
            kps = test_ques["identified_kps"].strip()
            context = f"""
{context}

Here are some knowledge points that might be relevant:
{kps}
""".strip()

        query = f"""
{question_sent}

A. {test_ques["answer_choices"]["A"]}
B. {test_ques["answer_choices"]["B"]}
C. {test_ques["answer_choices"]["C"]}
D. {test_ques["answer_choices"]["D"]}
""".strip()
        
        cc = ContextCiter(
            model, 
            tokenizer, 
            context=context, 
            query=query,
            num_ablations=args.num_ablations,  # increase to improve the faithfulness of context ablations
        )
        
        response = cc.response
        ca_scores_np = cc.get_attributions(as_dataframe=False, verbose=False)   # numpy array format
        sent_score_mapping = list(zip(cc.sources, ca_scores_np))
        
        new_json_dict = {
            "context": context,
            "query": query,
            "answer_letter": test_ques["answer_letter"],
            "question_idx": test_ques["question_idx"],
            "response": response,
            "ca_scores": sent_score_mapping,
        }
        print(f"For test question idx = {new_json_dict['question_idx']}:")
        print(f"Context:\n{new_json_dict['context']}")
        print(f"Query:\n{new_json_dict['query']}")
        print(f"Ground-Truth Answer: {new_json_dict['answer_letter']}")
        print(f"Response:\n{new_json_dict['response']}")
        print(f"CA Scores:\n")
        for sent, score in sent_score_mapping:
            print(f"{sent} --- {score}")
        print('\n')
        new_json_list.append(new_json_dict)
    
    
    ## 4. save new json list
    new_filename = os.path.basename(args.test_file).replace(".json", f"-ablations{args.num_ablations}-{rag_flag}.json")
    new_json_path = os.path.join(
        args.output_dir,
        new_filename,
    )
    with open(new_json_path, 'w') as fout:
        json.dump(new_json_list, fout, indent=4)
