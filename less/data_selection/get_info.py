"""
    This script is used for getting gradients or representations of a pre-trained model, a lora model, or a peft-initialized model for a given task.
"""

import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # place the model on only one GPU; otherwise error: tensors on different devices
from typing import Any

import torch
# num_threads = 14
# torch.set_num_threads(num_threads)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

from less.data_selection.collect_grad_reps import (collect_grads, collect_reps,
                                                   get_loss, collect_losses)
from less.data_selection.get_training_dataset import get_training_dataset
from less.data_selection.get_validation_dataset import (get_dataloader,
                                                        get_validation_dataset)
from less.train.model_arguments import add_reserved_padding_to_tokenizer

# hf_home = "/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface"   # CC at UIUC
hf_home = "/root/autodl-tmp/.cache/huggingface"
os.environ['HF_HOME'] = hf_home
model_cache_dir = f"{hf_home}/transformers" # `cache_dir` arg for .from_pretrained()

# torch_dtype argument for AutoModelForCausalLM.from_pretrained()
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

    for name, param in model.named_parameters():
        if 'lora' in name or 'Lora' in name:
            param.requires_grad = True
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script for getting validation gradients')
    parser.add_argument('--task', type=str, default=None,
                        help='Specify the name of validation task. One of variables of task and train_file must be specified')
    parser.add_argument("--train_file", type=str,
                        default=None, help="The path to the training data file we'd like to obtain the gradients/representations for. One of variables of task and train_file must be specified")
    parser.add_argument(
        "--info_type", choices=["grads", "reps", "losses"], help="The type of information")
    parser.add_argument("--model_path", type=str,
                        default=None, help="The path to the model")
    parser.add_argument("--max_samples", type=int,
                        default=None, help="The maximum number of samples")
    parser.add_argument("--torch_dtype", type=str,
                        choices=["float32", "bfloat16"], help="The torch data type")
    parser.add_argument("--output_path", type=str,
                        default=None, help="The path to the output")
    parser.add_argument("--gradient_projection_dimension", nargs='+',
                        help="The dimension of the projection, can be a list", type=int, default=[8192])
    parser.add_argument("--gradient_type", type=str, default="sgd",
                        choices=["adam", "sign", "sgd"], help="The type of gradient")
    parser.add_argument("--max_length", type=int, choices=[4096, 2048, 1024], 
                        help="The maximum length")  # default=4096,
    parser.add_argument("--batch_size", type=int, default=1, # choices=[1, 4, 8, 12, 16, 24, 32]
                        help="The batch size for dataloader; default to 1 for collect_grads")

    # the following 2 args are only used in step3.1: get info for validation points
    parser.add_argument("--print_ex", default=False, action="store_true",
                        help="Whether to print out & inspect all the decoded input_ids and labels of Validation Sets")
    parser.add_argument("--overwrite_cache", default=False, action="store_true",
                        help="Whether to overwrite saved datasets with all previous transforms under .cache/huggingface/datasets")

    # the following args are only used when `--initialize_lora=True`
    # by default, is_peft=True, thus the following args are not used,
    # that's the reason why they are not aligned with `base_training_args.sh`
    parser.add_argument("--initialize_lora", default=False, action="store_true",
                        help="Whether to initialize the base model with lora, only works when is_peft is False")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="The value of lora_r hyperparameter")
    parser.add_argument("--lora_alpha", type=float, default=32,
                        help="The value of lora_alpha hyperparameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="The value of lora_dropout hyperparameter")
    parser.add_argument("--lora_target_modules", nargs='+', default=[
                        "q_proj", "k_proj", "v_proj", "o_proj"],  help="The list of lora_target_modules")

    args = parser.parse_args()
    assert args.task is not None or args.train_file is not None
    print(args, '\n')


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        cache_dir=model_cache_dir,  # when directly loading pretrained models from huggingface
    )
    if tokenizer.chat_template == None:
        # sourced from https://huggingface.co/allenai/tulu-2-7b/blob/main/tokenizer_config.json
        # additionally trim the whitespaces of each `message['content']` to make it exactly the same format as open-instruct `concat_messages`
        tulu_chat_template_trim_whitespace = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] | trim }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] | trim + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        tokenizer.chat_template = tulu_chat_template_trim_whitespace
        print(f"tokenizer.chat_template is None. By default we add tulu_chat_template_trim_whitespace.")
    else:
        print(f"Tokenizer already has a pre-defined chat_template saved with model checkpoints.")

    dtype = dtype_dict[args.torch_dtype]    # bfloat16 or float32
    model = load_model(args.model_path, dtype)

    # pad token is not added by default for pretrained models
    add_reserved_padding_to_tokenizer(args.model_path, tokenizer)
    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        # this is only intended for the case where you directly load a PEFT adapter model
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False


    # if `is_peft=False` in `load_model`
    if args.initialize_lora:
        raise NotImplementedError
        assert not isinstance(model, PeftModel)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)

    if isinstance(model, PeftModel):
        model.print_trainable_parameters()


    adam_optimizer_state = None
    if args.info_type == "grads" and args.gradient_type == "adam":
        optimizer_path = os.path.join(args.model_path, "optimizer.bin") # original by mengzhou
        # optimizer_path = os.path.join(args.model_path, "optimizer.pt")  # ckpt name modified
        adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]


    if args.task is not None:
        dataset = get_validation_dataset(
            task=args.task,
            tokenizer=tokenizer,
            print_ex=args.print_ex,
            max_length=args.max_length,
            overwrite_cache=args.overwrite_cache,   # only used in `get_messages_dataset`
        )
        dataloader = get_dataloader(
            dataset, 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )   # batch_size default to 1
    else:
        assert args.train_file is not None
        # # see if the max length can go through
        # max_length = 2048
        # dataset_size = 30000
        # input_ids = [torch.randint(0, 128000, (max_length, )) for _ in range(dataset_size)]
        # attention_mask = [torch.ones(max_length, ) for _ in range(dataset_size)]
        # labels = input_ids
        # dataset = Dataset.from_dict({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
        
        dataset = get_training_dataset(
            train_files=args.train_file, 
            tokenizer=tokenizer, 
            max_seq_length=args.max_length, 
            sample_percentage=1.0, # set `sample_percentage=1.0` to calculate grad features for the whole training set
            seed=0,
            maintain_sample_order=True, # maintain_sample_order must be True in order to select data based on info scores
        )

        dataloader = get_dataloader(
            dataset, 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )   # batch_size default to 1

    if args.max_samples is None:
        args.max_samples = len(dataset)
        print(f"args.max_samples: {args.max_samples}")
    print(f"args.batch_size: {args.batch_size}\n")

    if args.info_type == "reps":
        collect_reps(
            dataloader, 
            model, 
            args.output_path,
            max_samples=args.max_samples
        )
    elif args.info_type == "grads":
        collect_grads(
            dataloader,
            model,
            args.output_path,
            proj_dim=args.gradient_projection_dimension,
            gradient_type=args.gradient_type,
            adam_optimizer_state=adam_optimizer_state,
            max_samples=args.max_samples
        )
    elif args.info_type == "losses":
        # get_loss(dataloader, model, args.output_path)
        collect_losses(
            dataloader,
            model,
            args.output_path,
            max_samples=args.max_samples
        )
    else:
        raise NotImplementedError

    print("\n\n\n") # separate logs of different datasets/checkpoints
