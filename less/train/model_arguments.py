import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import GPT2Tokenizer, GPTNeoXTokenizerFast, LlamaTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    ### added ####
    lora: Optional[bool] = field(default=False, metadata={
                                 "help": "whether to use lora"})
    lora_r: Optional[int] = field(default=8, metadata={"help": ("r for lora")})
    lora_alpha: Optional[float]=field(default=32, metadata={"help": ("alpha for lora")})
    lora_dropout: Optional[float]=field(default=0.1, metadata={"help": ("dropout for lora")})
    lora_target_modules: List[str]=field(
        default_factory=list, metadata={"help": ("target modules for lora")})


def add_padding_to_tokenizer(tokenizer):
    """add an additional <pad> token to the tokenizer: will be initialized with normal distribution """
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})    # "additional_special_tokens" won't be split by the tokenization process


def add_reserved_padding_to_tokenizer(model_name_or_path, tokenizer):
    '''For llama-3, there're already 256 reserved tokens with all-zero initialization that can be used as pad_token.
    Except for `bos_token` and `eos_token` (which are inited with normal distribution), all the others are initialized with all-zero distribution'''
    # pad_token_mapping = {
    #     "meta-llama/Meta-Llama-3-8B":         "<|reserved_special_token_0|>",     # tokenizer.pad_token_id: 128002
    #     "mistralai/Mistral-7B-v0.3":          "[control_8]",                      # tokenizer.pad_token_id: 10
    #     "mistralai/Mistral-Nemo-Base-2407":   "<pad>"                             # tokenizer.pad_token_id: 10
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
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")
    print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
    