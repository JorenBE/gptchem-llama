import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import transformers
from datasets import Dataset
import pandas as pd
from functools import partial
from typing import List 


# models have different conventions for naming the attention modules
LORA_TARGET_MODULES_MAPPING = {
    "alpaca_native": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bart": ["q_proj", "v_proj"],
    "bert-base-uncased": ["query", "value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "bloom": ["query_key_value"],
    "chatglm": ["query_key_value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "electra": ["query", "value"],
    "EleutherAI/gpt-neo-125m": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neo-1.3B": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neo-2.7B": ["q_proj", "v_proj"],
    "EleutherAI/gpt-neox-20b": ["query_key_value"],
    "EleutherAI/pythia-12b": ["query_key_value"],
    "EleutherAI/pythia-70m-deduped": ["query_key_value"],
    "gpt2": ["c_attn"],
    "EleutherAI/gpt-j-6b": ["q_proj", "v_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "mt5": ["q", "v"],
    "opt": ["q_proj", "v_proj"],
    "roberta": ["query", "value"],
    "t5": ["q", "v"],
    "xlm-roberta": ["query", "value"],
}


def freeze_and_cast(model):
    """Freeze the model and cast small parameters to fp32 for stability."""
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(base_model: str = "gptj", load_in_8bit: bool = True, lora_kwargs: dict = {}):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_in_8bit,
        torch_dtype=torch.float16,
        device_map='sequential',
    )

    if load_in_8bit:
        freeze_and_cast(model)

    lora_default_kwargs = {
        "r": 16,  # lora attention size
        "lora_alpha": 32,  # "When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. As a result, we simply set α to the first r we try and do not tune it."
        "target_modules": LORA_TARGET_MODULES_MAPPING[base_model],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    lora_kwargs = {**lora_default_kwargs, **lora_kwargs}

    # now, apply LoRA to the model
    config = LoraConfig(
        **lora_kwargs,
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    return model, tokenizer


def tokenize(prompt, tokenizer, cutoff_len=1024):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    return result


def tokenize_prompt(data_point,  tokenizer, cutoff_len=1024, add_completion=True):
    if add_completion:
        full_prompt = data_point["prompt"] + data_point["completion"]
    else:
        full_prompt = data_point["prompt"]
    tokenized_full_prompt = tokenize(full_prompt, tokenizer=tokenizer, cutoff_len=cutoff_len)
    return tokenized_full_prompt


def train_model(
    model,
    tokenizer,
    train_data: pd.DataFrame,
    train_kwargs: dict = {},
    hub_model_name: str = None,
    report_to: str = None,
):
    default_train_kwargs = {
        "per_device_train_batch_size": 128,
        "warmup_steps": 100,
        "num_train_epochs": 20,
        "learning_rate": 3e-4,
        "fp16": True,
        "optim": "adamw_torch",
        "output_dir": './output',
          "report_to": report_to,  # can be used for wandb tracking
    }
    train_kwargs = {**default_train_kwargs, **train_kwargs}
    tokenize_partial = partial(tokenize_prompt, tokenizer=tokenizer)
    train_data = Dataset.from_pandas(train_data).shuffle().map(tokenize_partial)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(**train_kwargs),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    if hub_model_name is not None:
        model.push_to_hub(hub_model_name)


def complete(
    model,
    tokenizer: AutoTokenizer,
    prompt_text: List[str],
    max_length: int = 1024,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    do_sample: bool = False,
    repetition_penalty: float = 1.2,
    num_beams: int = 1,
):
    model.eval()
    device = model.device
    with torch.no_grad():
        prompt = tokenizer(
            prompt_text, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        prompt = {key: value.to(device) for key, value in prompt.items()}
        out = model.generate(
            **prompt,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )
        return {"out": out, "decoded": tokenizer.decode(out)}
