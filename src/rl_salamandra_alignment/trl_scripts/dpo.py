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

"""
DPO training script for TRL 0.29 (model-agnostic; written for a Gemma 4 run).

Differences from the stock TRL script:
  * the dataset and the chat template are checked BEFORE the model is loaded, so a bad template fails in seconds;
  * there is no silent chat-template fallback: the tokenizer must have its own template;
  * --max_train_samples / --max_eval_samples shrink the data for smoke tests;
  * the `datasets` + `dataset_name` branch actually loads the data.

Smoke test example (in the yaml, under rl_script_args):
    max_train_samples: 320
    max_eval_samples: 32
"""

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import logging
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class DPOScriptArguments(ScriptArguments):
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Use only the first N training examples (smoke tests)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "Use only the first N eval examples (smoke tests)."}
    )


def load_local_dataset(path, train_split, test_split):
    """Load `dataset_name` from disk. Accepts:
      * a folder written with save_to_disk (a DatasetDict with train/eval splits),
      * a folder of .jsonl/.json files: every file goes into the train split, except files whose name is exactly
        a split name (e.g. eval.jsonl for the eval split, train.jsonl is added to train),
      * a single .jsonl/.json file (becomes the train split)."""
    if os.path.isfile(path):
        return load_dataset("json", data_files={train_split: path})
    if os.path.isfile(os.path.join(path, "dataset_dict.json")) or os.path.isfile(os.path.join(path, "state.json")):
        return load_from_disk(path)
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")) + glob.glob(os.path.join(path, "*.json")))
    if not files:
        raise ValueError(f"{path} is not a save_to_disk dataset and contains no .jsonl/.json files.")
    data_files = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        split = stem if stem in (train_split, test_split) else train_split
        data_files.setdefault(split, []).append(f)
    logger.info(f"Loading jsonl files from {path}: {data_files}")
    return load_dataset("json", data_files=data_files)


def check_chat_template(tokenizer, dataset, n=3):
    """TRL splits every conversational example into prompt and answer by rendering the template twice
    (prompt + generation prompt, and prompt + answer). The first must be a prefix of the second, otherwise
    chosen/rejected tokens are silently misaligned (e.g. a thinking-channel block that appears in only one)."""
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        prompt, chosen, rejected = ex.get("prompt"), ex["chosen"], ex["rejected"]
        if prompt is None or isinstance(chosen, str):
            logger.warning("Data without explicit conversational prompt: chat template check skipped.")
            return
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for name, answer in (("chosen", chosen), ("rejected", rejected)):
            full_text = tokenizer.apply_chat_template(prompt + answer, tokenize=False)
            if i == 0:
                print(f"[template check] prompt  tail: {prompt_text[-160:]!r}")
                print(f"[template check] {name:8s} tail: {full_text[-160:]!r}")
            if not full_text.startswith(prompt_text):
                raise RuntimeError(
                    f"Chat template check failed on example {i} ({name}): the rendered prompt is not a prefix of "
                    f"prompt+answer, so TRL would misalign the completion tokens.\n"
                    f"prompt     : {prompt_text[-300:]!r}\nprompt+answer: {full_text[:len(prompt_text) + 100][-400:]!r}"
                )
    print("[template check] prompt is a prefix of prompt+answer: OK")


def main(script_args, training_args, model_args, dataset_args):
    if training_args.fsdp_config:
        print("printing fsdp_config")
        print(type(training_args.fsdp_config))
        print(training_args.fsdp_config)
    else:
        print("No fsdp_config detected!")

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    logger.info(f"Model type: {config.model_type}")

    # ------------------------------------------------------------------ tokenizer and data first (fail fast)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        raise ValueError(
            f"The tokenizer of {model_args.model_name_or_path} has no chat template. Refusing to train with an "
            "implicit fallback; set the model's own template explicitly."
        )

    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_local_dataset(script_args.dataset_name, script_args.dataset_train_split, script_args.dataset_test_split)
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    needed = [script_args.dataset_train_split]
    if training_args.eval_strategy != "no":
        needed.append(script_args.dataset_test_split)
    missing = [split for split in needed if split not in dataset]
    if missing:
        raise ValueError(
            f"Dataset splits {missing} not found; available: {list(dataset.keys())}. A jsonl folder only gets an "
            "eval split from a file named after it (e.g. eval.jsonl); otherwise set eval_strategy: 'no'."
        )

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    if script_args.max_train_samples:
        train_dataset = train_dataset.select(range(min(script_args.max_train_samples, len(train_dataset))))
    if eval_dataset is not None and script_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(script_args.max_eval_samples, len(eval_dataset))))
    logger.info(f"Train examples: {len(train_dataset)} | eval examples: {len(eval_dataset) if eval_dataset else 0}")

    check_chat_template(tokenizer, train_dataset)

    # ------------------------------------------------------------------ model
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

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    logger.info(f"Loaded model class: {type(model).__name__}")

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        # With PEFT, DPOTrainer handles the reference policy (adapter disabled).
        ref_model = None

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if peft_config is not None and training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ------------------------------------------------------------------ trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    logger.info(f"TRL treats the model as a vision-language model: {getattr(trainer, '_is_vlm', 'unknown')}")

    if peft_config is not None:
        from collections import Counter

        from peft.tuners.lora import LoraLayer

        names = [n for n, m in trainer.model.named_modules() if isinstance(m, LoraLayer)]
        breakdown = Counter(n.split(".")[-1] for n in names)
        print(f"[LoRA sanity] modules={len(names)}, breakdown={dict(breakdown)}")
        non_text = [n for n in names if any(k in n for k in ("vision", "audio", "multi_modal", "image"))]
        if non_text:
            logger.warning(
                f"{len(non_text)} LoRA modules sit in vision/audio parts of the model, e.g. {non_text[:3]}. "
                "Restrict `lora_target_modules` to the language model if you only train on text."
            )
        n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        n_total = sum(getattr(p, "ds_numel", p.numel()) for p in trainer.model.parameters())
        print(f"[LoRA sanity] trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / max(n_total, 1):.3f}%)")

    # ------------------------------------------------------------------ train
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
    dataclass_types = (DPOScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
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
