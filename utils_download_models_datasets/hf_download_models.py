import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast


def add_reserved_padding_to_tokenizer(model_name_or_path, tokenizer):
    '''For llama-3, there're already 256 reserved tokens with all-zero initialization that can be used as pad_token.
    Except for `bos_token` and `eos_token` (which are inited with normal distribution), all the others are initialized with all-zero distribution'''
    # pad_token_mapping = {
    #     "meta-llama/Meta-Llama-3-8B": "<|reserved_special_token_0|>",   # tokenizer.pad_token_id: 128002
    #     "mistralai/Mistral-7B-v0.3": "[control_8]",                     # tokenizer.pad_token_id: 10
    # }

    llama_pad_token = "<|reserved_special_token_0|>"
    mistral_pad_token = "[control_8]"
    nemo_pad_token = "<pad>"
    # for qwen-2.5, pad_token is initialized to be the same as eos_token
    # to prevent potential infinite generation issue, we follow generation config of qwen2.5-instruct and change eos_token to "<|im_end|>"

    if tokenizer.pad_token is None:
        if "Llama" in model_name_or_path:
            pad_token = llama_pad_token
        elif "Mistral-7B-v" in model_name_or_path:
            pad_token = mistral_pad_token
        elif "Mistral-Nemo" in model_name_or_path:
            pad_token = nemo_pad_token
        else:
            raise NotImplementedError

        tokenizer.add_special_tokens({"pad_token": pad_token})
        print(f"Special token {pad_token} is added to the tokenizer")
    else:
        if "Qwen" in model_name_or_path:
            assert tokenizer.pad_token == "<|endoftext|>"
            if tokenizer.eos_token == tokenizer.pad_token:
                tokenizer.eos_token = "<|im_end|>"
        print(f"tokenizer already has pre-defined pad_token: {tokenizer.pad_token}")
        print(f"eos_token is: {tokenizer.eos_token}, which should be different from pad_token")
    # print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    # print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")



if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    model_name_or_path_list = [
        # 'meta-llama/Llama-2-7b-hf'
        # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
        # 'meta-llama/Meta-Llama-3-8B'
        # 'TinyLlama/TinyLlama_v1.1' # tinyllama v1.1 updated in 06/2024
        # 'mistralai/Mistral-7B-v0.1'

        # 'meta-llama/Llama-3.1-8B',
        # 'mistralai/Mistral-7B-v0.3',
        # 'Qwen/Qwen2.5-7B',
        
        # 'meta-llama/Llama-3.2-1B',
        # 'Qwen/Qwen2.5-0.5B',
        
        # 'mistralai/Mistral-Nemo-Base-2407',   # larger models
        # 'meta-llama/Llama-3.2-3B',            # smaller models
        
        'meta-llama/Llama-3.2-3B-Instruct',
    ]

    cache_dir = '/root/autodl-tmp/.cache/huggingface/transformers'   # for autodl
    # cache_dir = '/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface/transformers'   # for CC at UIUC
    os.makedirs(cache_dir, exist_ok=True)


    for model_name_or_path in model_name_or_path_list:
        if "Mistral-Nemo-Base-2407" in model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir,
                model_input_names=[
                    "input_ids",
                    "attention_mask",
                ]
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                cache_dir=cache_dir,
            )
        print(f"Successfully downloaded and loaded tokenizer for: {model_name_or_path}")
        
        if 'Instruct' not in model_name_or_path:
            add_reserved_padding_to_tokenizer(model_name_or_path, tokenizer)
            print(f"tokenizer.chat_template:\n{tokenizer.chat_template}")
            tulu_chat_template_trim_whitespace = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] | trim }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] | trim + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
            tokenizer.chat_template = tulu_chat_template_trim_whitespace
        
        print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
        print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
        print(f"tokenizer.chat_template:\n{tokenizer.chat_template}")


        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            cache_dir=cache_dir
        )
        print(model)
        print(f"Successfully downloaded and loaded model for: {model_name_or_path}")
        
        # test simple inputs and outputs
        inputs = tokenizer(
            "What is your name?\n", 
            return_tensors="pt", 
            # return_token_type_ids=False,
        )
        print(inputs)
        outputs = model.generate(**inputs, max_new_tokens=20)
        print(tokenizer.decode(outputs[0], skip_special_tokens=False))
        
        
        print(type(tokenizer))
        print(type(tokenizer) == PreTrainedTokenizerFast)
        print(tokenizer.model_input_names)
        print("\n\n")
