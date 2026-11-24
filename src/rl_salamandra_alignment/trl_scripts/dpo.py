# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
from accelerate import logging
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForMultimodalLM, AutoTokenizer
from dataclasses import dataclass, field
from typing import Optional

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

logger = logging.get_logger(__name__)

@dataclass
class DPOScriptArguments(ScriptArguments):
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Use only the first N training examples (smoke tests)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Use only the first N eval examples (smoke tests)."}
    )

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

# Gemma 4 is multimodal: "all-linear" would also hit the vision/audio towers.
# Restrict LoRA to the language model's linear layers (check `print(model)` if the path differs).
GEMMA4_LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def main(script_args, training_args, model_args, dataset_args):
    if training_args.fsdp_config:
        print("printing fsdp_config")
        print(type(training_args.fsdp_config))
        print(training_args.fsdp_config)
    else:
        print("No fsdp_config detected!")

    ################
    # Model Setup
    ###################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForMultimodalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is not None and model_args.lora_target_modules in (None, ["all-linear"], "all-linear"):
        peft_config.target_modules = GEMMA4_LORA_TARGET_REGEX

    # Reference Model Logic
    if (not hasattr(training_args, "ref_model_path") or
        (training_args.ref_model_path is None and isinstance(model_args.model_name_or_path, str))):
        training_args.ref_model_path = model_args.model_name_or_path

    if peft_config is None:
        ref_model = AutoModelForMultimodalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs
        )
    else:
        # For LoRA, the reference model is handled internally by DPOTrainer
        ref_model = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer.chat_template_kwargs = {"enable_thinking": False}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            f"No chat template found for {model_args.model_name_or_path}. "
            "Refusing to fall back to a generic template; check the tokenizer/processor files."
        )
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning("Both `datasets` and `dataset_name` are provided. Using `datasets`.")
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_from_disk(script_args.dataset_name)
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the DPO trainer
    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(script_args.max_train_samples, len(train_dataset))))

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_test_split]
        if script_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(script_args.max_eval_samples, len(eval_dataset))))

    # Workaround for PEFT + DeepSpeed ZeRO-3 + gradient checkpointing
    if peft_config is not None and training_args.gradient_checkpointing:
        gc_kwargs = (
            getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
        )
        training_args.gradient_checkpointing_kwargs = {
            **gc_kwargs,
            "use_reentrant": True,
        }

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # if training_args.eval_strategy != "no":
    #     metrics = trainer.evaluate()
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (DPOScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
