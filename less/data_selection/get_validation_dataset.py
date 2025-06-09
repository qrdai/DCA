import json
from functools import partial
from typing import List, Tuple, Union
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# hf_home = "/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface"   # CC at UIUC
hf_home = "/root/autodl-tmp/.cache/huggingface" # autodl
dataset_cache_dir = f"{hf_home}/datasets" # `cache_dir` for load_dataset()


def get_messages_dataset(
        messages_files: Union[List[str], str], 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int,
        overwrite_cache: bool = False,
        **kwargs,   # capture unused arguments
    ):
    """ get validation dataset from .jsonl files with messages format"""

    # 1. load `messages` data from .jsonl files
    raw_datasets = load_raw_dataset(messages_files=messages_files)

    # 2. format each message into the chat template; then encode into input_ids & mask user prompt labels
    lm_datasets = encode_data(
                    raw_datasets, 
                    tokenizer, 
                    max_length,
                    processing_num_workers=2,
                    overwrite_cache=overwrite_cache,
                )    # if overwrite_cache == False, then even if we make changes to `encode_with_messages_format`, it will not be reflected in the final dataset, since we're still using the old cache

    return lm_datasets


def load_raw_dataset(
        messages_files: Union[List[str], str],
    ):
    """ load raw dataset from jsonl files"""
    if isinstance(messages_files, str):
        messages_files = [messages_files]

    processed_datasets = load_dataset(
        "json",
        data_files=messages_files, # can be a list of paths (in step 1) or a str of one path (in step 2&3)
        cache_dir=dataset_cache_dir
    )["train"]  # By default, loading local files creates a DatasetDict object with a train split

    print(f"Sample order inside {messages_files} is always maintained, since it's a Validation Dataset.")

    return processed_datasets


def encode_data(
        raw_datasets, 
        tokenizer, 
        max_length, 
        processing_num_workers=2, # changed from 10 -> 2 to debug `OSError: [Errno 12] Cannot allocate memory`
        overwrite_cache=False,  # True,
    ):
    """ encode data with the specified tokenizer and the chat format. """
    assert "messages" in raw_datasets.column_names
    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos=False,
    )

    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Chat-Formatting, Tokenizing, and Masking Validation Dataset",
    )
    lm_datasets.set_format(type="pt")

    # remove redundant columns
    columns = deepcopy(lm_datasets.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    lm_datasets = lm_datasets.remove_columns(columns)

    return lm_datasets


def encode_with_messages_format(example, tokenizer, max_length, add_bos=False):
    """
    1. (08/12/2024) Sourced from: https://github.com/allenai/open-instruct/blob/f5cd4286dd9fbe2f56e22bfc458a0e40f9a2d89d/open_instruct/finetune.py 

    2. Here we assume each example has a 'messages' field. Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.

    3. `tokenizer` should already have attribute `chat_template`. default to that of tulu_v2 with additional whitespace trimming
    
    4. This encoding method should only work with the following dataset condition: 
        - all the content in "assistant" msg should be treated as labels; i.e., all texts after "<|assistant|>\n" should be labels and calculated loss against
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    # def _concat_messages(messages):
    #     message_text = ""
    #     for message in messages:
    #         if message["role"] == "system":
    #             message_text += "<|system|>\n" + message["content"].strip() + "\n"
    #         elif message["role"] == "user":
    #             message_text += "<|user|>\n" + message["content"].strip() + "\n"
    #         elif message["role"] == "assistant":
    #             message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
    #         else:
    #             raise ValueError("Invalid role: {}".format(message["role"]))
    #     return message_text

    # 1. no need for `add_generation_prompt` during training or validation
    # 2. need to strip the trailing `\n` after `eos_token` in the final assistant message
    example_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text   # since llama-2 and llama-3 both add bos_token by default (but not eos_token), so `example_text` should not contain `bos_token`

    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    # 1. Should strip the trailing '\n' here, since it's also something after tokenizer.eos_token in assistant message content, and should be included in the masking scope (should also re-check if this is true in train.log)
                    # 2. But for UI training set, it contains only single-turn instruction data. So this branch actually won't be accessed in training
                    tokenizer.apply_chat_template(messages[:message_idx], tokenize=False, add_generation_prompt=False).strip(),
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).input_ids.shape[1]

            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the assistant header "<|assistant|>\n"
                # Note that we shouldn't include strip() here, since the trailing '\n' after each user message content should also be masked
                messages_so_far = tokenizer.apply_chat_template(messages[: message_idx + 1], tokenize=False, add_generation_prompt=False) + "<|assistant|>\n"   # equivalent to `add_generation_prompt=True`
            else:
                # messages_so_far = tokenizer.apply_chat_template(messages[: message_idx + 1], tokenize=False, add_generation_prompt=False)
                raise NotImplementedError

            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def tokenize_prompt_completion(
    tokenizer: PreTrainedTokenizerBase,    # base class for both slow and fast PreTrainedTokenizer
    prompt: str,
    completion: str,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        prompt (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    # NOTE: completion should already contain tokenizer.eos_token (but not tokenizer.bos_token)
    assert tokenizer.eos_token in completion # and tokenizer.bos_token not in prompt -> Qwen-2.5 doesn't have bos_token by design
    full_prompt = prompt + completion

    # add_special_tokens default to True: for llama-2 and llama-3, by default add <bos_token>, but not <eos_token>
    prompt_input_ids = torch.tensor(tokenizer.encode(prompt, max_length=max_length, truncation=True))
    full_input_ids = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length, truncation=True))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length, truncation=True)) # originally `truncation` unset in tokenizer.encode; but doesn't matter as long as `max_length` is set: `truncation` will be default to 'longest_first'

    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask


def get_human_eval_dataset(
    data_file: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096, # default to 4096 for UI
    **kwargs,   # capture unused arguments
):
    """_summary_

    Args:
        data_file (str): _description_
        tokenizer (PreTrainedTokenizerBase): _description_
        max_length (int, optional): _description_. Defaults to 4096.
    """
    def apply_chat_format(tokenizer, inst, suffix):
        messages = [{"role": "user", "content": inst}]
        # prompt = chat_formatting_function(messages, tokenizer, add_bos=False) # the tulu chat_formatting_function in open-instruct adds a generation prompt "<|assistant|>\n" by default
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = "" if prompt[-1] in ["\n", " "] else " "
        return prompt + prefix + suffix

    def make_conv_humanevalpack(tokenizer, example):
        '''Directly use more realistic and cleaned instructions from humanevalpack'''
        instruction = example["instruction"]    # already cleaned in humanevalpack.jsonl

        answer = "Here is the function:\n\n```python\n"
        suffix = answer + example["prompt"]

        prompt = apply_chat_format(tokenizer, instruction, suffix)
        return prompt

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    with open(data_file, 'r') as fin:
        examples = [json.loads(line) for line in fin]

    for example in examples:
        prompt = make_conv_humanevalpack(tokenizer, example)    # `prompt` contains all that should be excluded from the labels
        completion = example["canonical_solution"].rstrip() + "\n```" + tokenizer.eos_token   # `completion` is equivalent to `labels`: exactly what the model should fill in

        full_input_ids, labels, attention_mask = tokenize_prompt_completion(
            tokenizer, 
            prompt, 
            completion, 
            max_length,
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type="pt")   # if unset, `__getitem__` returns python lists (instead of torch tensors) by default
    return dataset


def get_mbpp_dataset(
    data_file: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096, # default to 4096 for UI
    **kwargs,   # capture unused arguments
):
    """_summary_

    Args:
        data_file (str): _description_
        tokenizer (PreTrainedTokenizerBase): _description_
        max_length (int, optional): _description_. Defaults to 4096.
    """
    def make_signature(code):
        # 1. Modified Eurus code: the original signature.lstrip("def ") will strip leading 'd', 'e', 'f' in function names!
        signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
        signature = signature[4:].replace(" ", "").rstrip(":").strip().replace(",", ", ")
        assert ":" not in signature

        # # 2. The following simple logic provided by open-instruct should work for all mbpp examples?
        # signature = code.split(":")[0] + ':'  # e.g., `def find_lucas(n):`
        return signature

    def apply_chat_format(tokenizer, inst, suffix):
        messages = [{"role": "user", "content": inst}]
        # prompt = chat_formatting_function(messages, tokenizer, add_bos=False) # the tulu chat_formatting_function in open-instruct adds a generation prompt "<|assistant|>\n" by default
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix = "" if prompt[-1] in ["\n", " "] else " "
        return prompt + prefix + suffix

    def make_conv(example):
        description = example["prompt"].split(" https://www.")[0].strip()
        testcase = "\n>>> ".join(example["test_list"])
        # testcase = example["test_list"][0]

        signature = make_signature(example["code"])

        data_inst = (
            f"Write a Python function `{signature}` to solve the following problem:\n"
            f"{description}\n"
            f">>> {testcase}\n"
        )

        suffix = "Below is a Python function that solves the problem and passes corresponding tests:"
        suffix_inst = (
            f"{suffix}\n"   # no need for a leading '\n'
            f"```python\n"
            f"{example['code'].split(':')[0]}:\n"
        )

        prompt = apply_chat_format(tokenizer, data_inst, suffix_inst)
        return prompt

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    with open(data_file, 'r') as fin:
        examples = [json.loads(line) for line in fin]
    
    for example in examples:
        prompt = make_conv(example)
        raw_completion = example["code"].split(f"{example['code'].split(':')[0]}:")[1]
        completion = '\n'.join(raw_completion.split('\n')[1:]).rstrip() + "\n```" + tokenizer.eos_token
        
        full_input_ids, labels, attention_mask = tokenize_prompt_completion(
            tokenizer, 
            prompt, 
            completion, 
            max_length,
        )
        
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type="pt")
    return dataset


def get_mmlu_and_bbh_dataset(
    data_file: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4096, # default to 4096 for UI
    **kwargs,   # capture unused arguments
):
    """_summary_

    Args:
        data_file (str): _description_
        tokenizer (PreTrainedTokenizerBase): _description_
        max_length (int, optional): _description_. Defaults to 4096.
    """
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    with open(data_file, 'r') as fin:
        examples = [json.loads(line) for line in fin]
    
    for example in examples:
        prompt = example["query"]
        completion = example["completion"].rstrip() + tokenizer.eos_token
        
        full_input_ids, labels, attention_mask = tokenize_prompt_completion(
            tokenizer, 
            prompt, 
            completion, 
            max_length,
        )

        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type="pt")
    return dataset


def get_validation_dataset(
    task: str,
    tokenizer: PreTrainedTokenizerBase,
    print_ex: bool,
    **kwargs,
):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    messages_task_file_mapping = {
        "TACO": f"../data/TACO/validation_50_messages.jsonl",   # competitive coding
        
        "gsm_plus": f"../Eurus/eval/Math/subset/data/validation_50_with_boxed_messages.jsonl",
        "MATH": f"../Eurus/eval/Math/math/validation_50_messages.jsonl",
        # "omni_math": f"../data/Omni-MATH/validation_100_messages.jsonl",    # competition-level math: 100-shot in D1 - D3 (for GSM, MATH, TheoremQA)
        "omni_math": f"../data/Omni-MATH/validation_50_messages.jsonl",    # competition-level math: 50-shot in D1.5 - D2.5 (for MATH only)
        
        "if_eval": f"../Eurus/eval/Ins-Following/if_eval/validation_50_messages.jsonl",
        "auto_if": f"../data/AutoIF-instruct-61k/validation_50_messages.jsonl", # general instruction following
        
        "medqa_nonrag": f"../context-cite/qualitative_results/size10-seed42-ablations256-nonrag.jsonl",
        "medqa_rag": f"../context-cite/qualitative_results/size10-seed42-ablations256-rag.jsonl",
    }
    prompt_completion_task_file_mapping = {
        "human_eval": f"../Eurus/eval/Coding/human_eval/data/validation_50_humanevalpack.jsonl",
        "mbpp": f"../Eurus/eval/Coding/mbpp/validation_50.jsonl",
        "evoeval_combine": f"../data/EvoEval_combine/validation_50_humanevalpack.jsonl",    # general coding
        
        "mmlu": f"../Eurus/eval/mmlu/validation_50.jsonl",
        "bbh": f"../Eurus/eval/Reasoning/bbh/validation_50.jsonl",
    }

    # 1. load validation sets
    if task in messages_task_file_mapping.keys():
        messages_files = messages_task_file_mapping[task]
        validation_set = get_messages_dataset(
            messages_files=messages_files, 
            tokenizer=tokenizer,
            **kwargs,
        )
    elif task in prompt_completion_task_file_mapping.keys():
        data_file = prompt_completion_task_file_mapping[task]
        if task == "mmlu" or task == "bbh":
            validation_set = get_mmlu_and_bbh_dataset(
                data_file=data_file,
                tokenizer=tokenizer,
                **kwargs,
            )
        elif task == "human_eval" or "evoeval" in task:
            validation_set = get_human_eval_dataset(
                data_file=data_file,
                tokenizer=tokenizer,
                **kwargs,
            )
        elif task == "mbpp":
            validation_set = get_mbpp_dataset(
                data_file=data_file,
                tokenizer=tokenizer,
                **kwargs,
            )
        else:
            return ValueError(f"Prompt-Completion Task <{task}> not Implemented yet")
    else:
        raise ValueError("Invalid task name: Neither messages nor prompt-completion")

    # 2. inspect if there're overlong samples in validation sets
    # after encoding, we should filter out overlong samples whose labels are all -100, since in step2 gradient calculation, these samples will cause vectorized_grads & projected_grads to be both all-zeros
    # NOTE: ideally, no validation data point is overlong; so len(validation_set) should remain unchanged after filtering
    print(f"Before Filtering: len(validation_set) = {len(validation_set)}")
    validation_set = validation_set.filter(lambda example: not torch.all(example["labels"] == -100))
    print(f"After Filtering: len(validation_set) = {len(validation_set)}")

    # 3. print all the decoded validation input_ids & labels, to see if they fully recover the prompt-completion format
    # NOTE: we place `print_ex` at the end of `get_validation_dataset`, so that any datasets cache-related problems can be reflected in the decoded text (e.g., some newly updated transforms do not take effect due to `overwrite_cache=False`)
    if print_ex:
        for idx in range(len(validation_set)):
            example = validation_set[idx]
            input_ids = example["input_ids"]
            labels = example["labels"]

            prompt_length = torch.eq(labels, -100).sum().item()
            prompt_input_ids = input_ids[:prompt_length]

            print("******** Example starts ********")
            print(f"Prompt:")
            print(tokenizer.decode(prompt_input_ids))
            print(f"Completion (labels):")
            print(tokenizer.decode(labels[len(prompt_input_ids):]))
            print("******** Example ends ********")

    return validation_set


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") # model=model
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)   # `shuffle` default to False, so preserve the original order of `dataset`
    print("There are {} examples in the dataset".format(len(dataset)))  # batch_size always default to 1, so len(dataset) == # examples
    return dataloader
