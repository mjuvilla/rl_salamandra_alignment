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

# /// script
# dependencies = [
#     "trl @ git+https://github.com/huggingface/trl.git",
#     "peft",
#     "trackio",
# ]
# ///

"""
# Full training
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

# LoRA:
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --dataset_streaming \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""

import argparse
import os

import torch
from accelerate import logging
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.models.auto import modeling_auto

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

from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

logger = logging.get_logger(__name__)

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args, dataset_args):
    if training_args.fsdp_config:
        print("printing fsdp_config")
        print(type(training_args.fsdp_config))
        print(training_args.fsdp_config)
    else:
        print("No fsdp_config detected!")

    model_id = str(model_args.model_name_or_path).lower()
    is_mistral_small_3 = (
        "mistral-small-3.1" in model_id
        or "mistral3" in model_id
    )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    attn_impl = model_args.attn_implementation

    # Minimal Mistral-specific adjustment:
    # use eager only when the configuration requests auto.
    if is_mistral_small_3 and attn_impl == "auto":
        attn_impl = "eager"

    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=attn_impl,
        torch_dtype=torch_dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # Model loading
    if is_mistral_small_3:
        logger.info(
            "Mistral Small 3.1 detected; loading "
            "Mistral3ForConditionalGeneration."
        )

        try:
            from transformers.models.mistral3.modeling_mistral3 import (
                Mistral3ForConditionalGeneration,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import Mistral3ForConditionalGeneration. "
                "Check that your Transformers version supports Mistral Small 3.1."
            ) from e

        model_class = Mistral3ForConditionalGeneration
    else:
        logger.info("Using AutoModelForCausalLM.")
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )

    # PEFT / LoRA configuration
    peft_config = get_peft_config(model_args)

    if is_mistral_small_3 and peft_config is not None:
        peft_config.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]

    # Reference model
    if (
        not hasattr(training_args, "ref_model_path")
        or (
            training_args.ref_model_path is None
            and isinstance(model_args.model_name_or_path, str)
        )
    ):
        # If the reference model is not specified, assume it is the same as the initial modelAdd commentMore actions
        training_args.ref_model_path = model_args.model_name_or_path

    if peft_config is None:
        ref_model = model_class.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        # With PEFT, DPOTrainer handles reference-policy behavior.
        ref_model = None

    # Tokenizer
    tokenizer_kwargs = {
        "trust_remote_code": model_args.trust_remote_code,
    }

    if is_mistral_small_3:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
        except TypeError as e:
            if "fix_mistral_regex" not in str(e):
                raise

            logger.warning(
                "fix_mistral_regex is unsupported by this Transformers version."
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                **tokenizer_kwargs,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            **tokenizer_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        # dataset = load_dataset(
        #     script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        # )
        dataset = load_from_disk(script_args.dataset_name)
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    if peft_config is not None and training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # Initialize the DPO trainer
    # --- FIX: Vision Registry Hack for Mistral 3.1 ---
    if is_mistral_small_3:
        logger.info("Mistral Small 3.1 detected: Applying TRL Vision Registry hack.")
        mapping = modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
        original_mistral3_entry = mapping.pop("mistral3", None)
        try:
            trainer = DPOTrainer(
                model, ref_model, args=training_args,
                train_dataset=dataset[script_args.dataset_train_split],
                eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
                processing_class=tokenizer, peft_config=peft_config,
            )
        finally:
            if original_mistral3_entry is not None:
                mapping["mistral3"] = original_mistral3_entry

        print("Model class:", type(model).__name__)
        print("Vision model path:", trainer.is_vision_model)

        if is_mistral_small_3 and trainer.is_vision_model:
            raise RuntimeError(
                "Mistral Small 3.1 is still using TRL's vision processing path. "
                "Check the installed TRL model detection logic."
            )

    else:
        # Normal path for all other models
        trainer = DPOTrainer(
            model, ref_model, args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
            processing_class=tokenizer, peft_config=peft_config,
        )

    # Train the model
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, dataset_args)
