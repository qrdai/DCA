#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time
import subprocess
import signal
# import wandb

import datasets
import torch
import torch.distributed as dist
import transformers
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DataCollatorForSeq2Seq, 
    HfArgumentParser, 
    Trainer, 
    TrainerCallback,
    PreTrainedTokenizerFast,
    set_seed
)
from safetensors.torch import load_file, save_file
from copy import deepcopy

from less.data_selection.get_training_dataset import get_training_dataset
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer, add_reserved_padding_to_tokenizer
from less.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hf_home = "/projects/illinois/eng/cs/haopeng/qirundai/.cache/huggingface"   # CC at UIUC
hf_home = "/root/autodl-tmp/.cache/huggingface"
os.environ['HF_HOME'] = hf_home
model_cache_dir = f"{hf_home}/transformers" # `cache_dir` arg for .from_pretrained()
# os.environ['WANDB_PROJECT'] is now set in training_args
# os.environ['WANDB_CACHE_DIR'] = "/projects/illinois/eng/cs/haopeng/qirundai/.cache/wandb"
# os.environ['WANDB_CONFIG_DIR'] = "/projects/illinois/eng/cs/haopeng/qirundai/.config/wandb"
os.environ['WANDB_CACHE_DIR'] = "/root/autodl-tmp/.cache/wandb"
os.environ['WANDB_CONFIG_DIR'] = "/root/autodl-tmp/.config/wandb"


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


# torch_dtype argument for AutoModelForCausalLM.from_pretrained()
dtype_dict = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": "auto", # default to model.config.torch_dtype, which is recorded in config.json in its huggingface repo
    None: None, # default to torch.float32 for all pytorch-based packages, including huggingface transformers
}


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # wandb.init(entity="raidriar_dai", project=training_args.wandb_project)
    os.environ['WANDB_PROJECT'] = training_args.wandb_project

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. init and set chat template for tokenizer
    if "Mistral-Nemo-Base-2407" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=model_cache_dir,
            model_input_names=[
                "input_ids",
                "attention_mask",
            ]   # discard "token_type_ids"; this will be saved in ckpt: tokenizer_config.json
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, 
            cache_dir=model_cache_dir,
        )
    # sourced from https://huggingface.co/allenai/tulu-2-7b/blob/main/tokenizer_config.json
    # additionally trim the whitespaces of each `message['content']` to make it exactly the same format as open-instruct `concat_messages`
    tulu_chat_template_trim_whitespace = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] | trim }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] | trim + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    tokenizer.chat_template = tulu_chat_template_trim_whitespace    # this will also be saved in ckpts

    # Load training dataset
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed,
                                         maintain_sample_order=False)

    # Load evaluation dataset here, before `add_padding_to_tokenizer`, to handle the case where "<pad>" is among the normal text of `analysis_dataset` (though this possibility is really low)
    analysis_dataset = None
    if training_args.analysis_mode:
        raise  NotImplementedError
        # from less.data_selection.get_validation_dataset import get_dataset
        # analysis_dataset = get_dataset(training_args.analysis_dataset,
        #                                data_dir=data_args.data_dir,
        #                                tokenizer=tokenizer,
        #                                max_length=data_args.max_seq_length)
        analysis_dataset = get_training_dataset(data_args.validation_files,
                                                tokenizer=tokenizer,
                                                max_seq_length=data_args.max_seq_length,
                                                sample_percentage=1.0,
                                                seed=0) # for validation dataset, sample_percentage and seed are fixed
        analysis_dataset = analysis_dataset.remove_columns(["dataset", "id", "messages"])

        # columns = deepcopy(analysis_dataset.column_names)
        # columns.remove("input_ids")
        # columns.remove("labels")
        # columns.remove("attention_mask")
        # analysis_dataset = analysis_dataset.remove_columns(columns)
        # logger.info(f"\nThe following columns are removed from the analysis dataset:\n{columns}\n")

    # 2. add pad_token to tokenizer
    # (1): add an additional <pad> token and resize embeddings
    # pad_token_id: 128256
    # add_padding_to_tokenizer(tokenizer)
    # (2) directly use reserved_special_token in llama-3 tokenizer
    # pad_token_id: 128002 (llama-3) or 10 (mistral-v0.3 / mistral-nemo)
    add_reserved_padding_to_tokenizer(model_args.model_name_or_path, tokenizer)

    # 3. update model.config.pad_token_id (also model.get_input_embeddings().padding_idx) = tokenizer.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=dtype_dict[model_args.torch_dtype], # if None, model.dtype will be default to torch.float32
        cache_dir=model_cache_dir,
        pad_token_id=tokenizer.pad_token_id # set model.config.pad_token_id = tokenizer.pad_token_id
    )
    logger.info(f"\nmodel.config.torch_dtype: {model.config.torch_dtype}")
    logger.info(f"model.dtype: {model.dtype}\n")
    
    # resize embeddings if needed (e.g. for LlamaTokenizer) -> what about PretrainedTokenizer that llama-3 uses?
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        # this is only intended for the case where you directly load a PEFT adapter model
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False

    # if first load a pretrained model and later add adapter to it
    if not isinstance(model, PeftModel) and model_args.lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(
            f"Applied LoRA to model."
        )
        model.print_trainable_parameters()

        # for checkpointing
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate the gradients to make sure the gradient flows
        # We do this operation on both PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334 and gradient checkpointing: https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/modeling_utils.py#L2273-L2278
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
            print(f"\nCalled the model's attribute `enable_input_require_grads`.\n")
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print(f"\nModel doesn't have `enable_input_require_grads`. Copy its source code implementation here.\n")

    get_data_statistics(train_dataset)

    # redundant columns already removed in `get_training_dataset`
    # if "dataset" in train_dataset.features: # "features" are just columns of a dataset
    #     train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])

    for index in random.sample(range(len(train_dataset)), 1):
        sample = train_dataset[index]
        logger.info(f"Sample {index} of the training set: {sample}")

        input_ids = sample["input_ids"]
        logger.info(f"Decoded Input IDs:\n{tokenizer.decode(input_ids)}")

        labels = [label for label in sample["labels"] if label > -1]
        logger.info(f"Decoded Labels:\n{tokenizer.decode(labels)}")

    model_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    # # Testing if the model can go through full length
    # # Test Results: 4096 works, but 8192 will even make `new_transformers_env` (new FSDP implementation that's more mem-efficient) 
    # # + `bfloat16_base` unable to work. So use 4096 with `old_env` + `bfloat16_base`.
    # from datasets import Dataset
    # max_length = 2048
    # logger.info(f"\nNow Testing if the model can go through max_seq_length: {max_length}\n")
    # input_ids = [torch.randint(0, 128000, (max_length, )) for _ in range(10000)]
    # attention_mask = [torch.ones(max_length, ) for _ in range(10000)]
    # train_dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids, "attention_mask": attention_mask})

    if dist.is_initialized() and dist.get_rank() == 0:
        logger.info(f"dist.is_initialized() and dist.get_rank() == 0")
        logger.info(f"current master port: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        print(model)
    elif not dist.is_initialized():
        logger.info(f"not dist.is_initialized()")
        print(model)

    # we don't add `compute_metrics` here, then during evaluate() we'll only get loss (model.forward -> outputs.loss), but won't get `logits` or `labels` 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest")
    )

    # evaluate before the first training step
    # sol 2: but still evaluate after training for 1 step
    if training_args.analysis_mode:
        trainer.add_callback(EvaluateFirstStepCallback())

    # Training
    train_result = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint
    )
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


    if dist.get_rank() == 0 and isinstance(model, PeftModel):
        # post-processing (only for the main process to do)
        # 1. remove the full model in the end to save space, only adapter is needed
        # 2. for transformers=4.42.4, `get_peft_model_state_dict` will set `save_embedding_layers = True` if vocab is expanded, even if the weights of embedding layers are frozen during LoRA training, so we have to remove them in `adapter_model.safetensors`
        keys_to_remove = [
            "base_model.model.lm_head.weight",
            "base_model.model.model.embed_tokens.weight"
        ]
        pytorch_model_path = os.path.join(training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None
        time.sleep(60)

        safetensors_path = os.path.join(training_args.output_dir, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            safetensors_saved = False
            retry_count = 0
            while not safetensors_saved and retry_count < 20:
                try:
                    adapter_model = load_file(safetensors_path)
                    safetensors_saved = True
                except Exception as e:
                    print(f"Caught an exception when loading safetensors file:\n{e}\n")
                    retry_count += 1
                    time.sleep(30)
            is_deleted = False
            for key in keys_to_remove:
                if key in adapter_model:
                    del adapter_model[key]
                    is_deleted = True
            save_file(adapter_model, safetensors_path) if is_deleted else None

        # also remove "pytorch_model_fsdp.bin" in each ckpt dir
        subdirs = [d for d in os.listdir(training_args.output_dir) if os.path.isdir(os.path.join(training_args.output_dir, d))]
        for subdir in subdirs:
            pytorch_model_path = os.path.join(training_args.output_dir, subdir, "pytorch_model_fsdp.bin")
            os.remove(pytorch_model_path) if os.path.exists(pytorch_model_path) else None
            time.sleep(60)

            safetensors_path = os.path.join(training_args.output_dir, subdir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                adapter_model = load_file(safetensors_path)
                is_deleted = False
                for key in keys_to_remove:
                    if key in adapter_model:
                        del adapter_model[key]
                        is_deleted = True
                save_file(adapter_model, safetensors_path) if is_deleted else None

    # No need for CC at UIUC
    # # Check and Kill active wandb-service processes, to avoid eternal upload
    # try:
    #     initial_check = subprocess.check_output(['pgrep', '-u', 'root', '-f', '^wandb-service'])
    #     print("wandb-service processes found, will check again in 60 seconds.")

    #     time.sleep(60)  # Wait for 60 seconds

    #     try:
    #         processes = subprocess.check_output(['pgrep', '-u', 'root', '-f', '^wandb-service'])
    #         pids = processes.decode('utf-8').strip().split()
    #         for pid in pids:
    #             # directly send SIGKILL
    #             print(f"Killing process ID {pid}")
    #             os.kill(int(pid), signal.SIGKILL)  # Ensure PID is an integer and use SIGKILL to force kill

    #             # "Graceful Shutdown" by sending SIGTERM first and then SIGKILL
    #             # pid = int(pid)
    #             # print(f"Sending SIGTERM to process ID {pid}")
    #             # os.kill(pid, signal.SIGTERM)
    #             # # Wait and check if the process is still running, then force kill
    #             # time.sleep(5)  # Wait for 5 seconds
    #             # try:
    #             #     subprocess.check_output(['ps', '-p', str(pid)])
    #             #     print(f"Process {pid} did not terminate, sending SIGKILL.")
    #             #     os.kill(pid, signal.SIGKILL)
    #             # except subprocess.CalledProcessError:
    #             #     print(f"Process {pid} has terminated.")
    #     except subprocess.CalledProcessError:
    #         print("No wandb-service processes found after waiting, or already terminated.")
    # except subprocess.CalledProcessError:
    #     print("No initial wandb-service processes found, no need to wait.")


if __name__ == "__main__":
    main()
