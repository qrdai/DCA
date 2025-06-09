import contextlib
from functools import partial
from typing import List, Union

import numpy as np
import torch
from datasets import load_dataset
from copy import deepcopy

# hf_home = "/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface"   # CC at UIUC
hf_home = "/root/autodl-tmp/.cache/huggingface" # autodl
dataset_cache_dir = f"{hf_home}/datasets" # `cache_dir` for load_dataset()


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(
        train_files: List[str], 
        tokenizer, 
        max_seq_length,
        maintain_sample_order,
        sample_percentage=1.0, 
        seed=0,
    ):
    """ get training dataset with a specified seed """

    # 1. load `messages` data from .jsonl files
    raw_datasets = load_raw_dataset(
                    train_files, 
                    sample_percentage=sample_percentage, 
                    seed=seed, 
                    maintain_sample_order=maintain_sample_order
                )

    # 2. format each message into the chat template; then encode into input_ids & mask user prompt labels
    lm_datasets = encode_data(
                    raw_datasets, 
                    tokenizer, 
                    max_seq_length
                )    # overwrite_cache default to False, which means even if we make changes to `encode_with_messages_format`, it will not be reflected in the final dataset, since we still use the old cache

    return lm_datasets


def load_raw_dataset(
        train_files: Union[List[str], str], 
        maintain_sample_order,
        sample_size=None, 
        sample_percentage=1.0, 
        seed=0,
    ):
    """ load raw dataset """
    if isinstance(train_files, str):
        train_files = [train_files]

    processed_datasets = load_dataset(
        "json",
        data_files=train_files, # can be a list of paths (in step 1) or a str of one path (in step 2&3)
        cache_dir=dataset_cache_dir
    )["train"]  # By default, loading local files creates a DatasetDict object with a train split

    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    assert not (maintain_sample_order and sample_size < len(processed_datasets)), f"maintain_sample_order should only be considered when sample_size == len(processed_datasets)"
    if maintain_sample_order:
        # not shuffle (by mengzhou) -> maintain the order of data points for step2: grad calculation
        assert sample_size == len(processed_datasets)
        print(f"Sample order inside {train_files} is maintained.")
        return processed_datasets
    else:
        # shuffle anyway even when sample_percentage=1.0, for step1: warmup training
        with temp_seed(seed):
            index = np.random.permutation(len(processed_datasets))[:sample_size]

    sampled_dataset = processed_datasets.select(index)

    return sampled_dataset


def encode_data(
        raw_datasets, 
        tokenizer, 
        max_seq_length, 
        processing_num_workers=8,  # to speed up dataset formatting in step 2
        overwrite_cache=False, 
        func_name="encode_with_messages_format"
    ):
    # `processing_num_workers` changed from 10 -> 2 to debug `OSError: [Errno 12] Cannot allocate memory`
    """ encode data with the specified tokenizer and the chat format. """
    assert func_name == "encode_with_messages_format", f"func_name = {func_name}!!!"

    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets

    encode_function = get_encode_function(
        raw_datasets, tokenizer, max_seq_length, func_name) # `func_name` default to encode_with_messages_format

    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Chat-Formatting, Tokenizing, and Masking instruction data",
    )
    lm_datasets.set_format(type="pt")

    # print(f"Before Filtering: len(lm_datasets) = {len(lm_datasets)}")
    # # after encoding, filter out overlong samples whose labels are all -100
    # # in step1 training, these samples will cause output.loss = nan, and thus loss.backward = [0.0, 0.0, ..., 0.0]
    # # in step2 gradient calculation, these samples will cause vectorized_grads & projected_grads to be both all-zeros
    # lm_datasets = lm_datasets.filter(lambda example: not torch.all(example["labels"] == -100))
    # print(f"After Filtering: len(lm_datasets) = {len(lm_datasets)}")

    # remove redundant columns
    columns = deepcopy(lm_datasets.column_names)
    columns.remove("input_ids")
    columns.remove("labels")
    columns.remove("attention_mask")
    lm_datasets = lm_datasets.remove_columns(columns)

    return lm_datasets


def get_encode_function(raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"):
    """ get encode function based on the dataset. """
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        # encode_function = partial(
        #     encode_with_prompt_completion_format,
        #     tokenizer=tokenizer,
        #     max_seq_length=max_seq_length,  # raw_datasets and tokenizer are preconfigured fixed arguments,
        #     # so `encode_function` can be directly passed into Dataset.map() since the two args no need to be set
        # )
        raise NotImplementedError
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
            encode_function = partial(
                encode_func,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                add_bos=False,
            )
        else:
            # encode_func = encode_with_messages_format_with_llama2_chat
            raise NotImplementedError
    else:
        raise ValueError("You need to have 'messages' in your column names.")
    return encode_function


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    """
    (08/12/2024) Sourced from: https://github.com/allenai/open-instruct/blob/f5cd4286dd9fbe2f56e22bfc458a0e40f9a2d89d/open_instruct/finetune.py 

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    
    `tokenizer` was already assigned `chat_template` in train.py
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
    # 2. need to strip the trailing `\n` after the final assistant message
    example_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text   # since llama-2 and llama-3 both add bos_token by default (but not eos_token), so `example_text` should not contain `bos_token`

    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
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
                    max_length=max_seq_length,
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
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages, ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""
        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + \
                    f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"])
                )
        formatted_text = formatted_text[len(bos):]
        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if messages[message_idx+1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }
