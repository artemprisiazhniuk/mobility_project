from functools import partial
import os
import json
import re
from itertools import chain
import argparse
from collections import Counter
import tempfile

from pathlib import Path
import sys
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForTokenClassification, DebertaV2Tokenizer
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType

import mlflow


def prepare_dataset(data, tokenizer, 
                    label2id=None, task_prompt=None, 
                    entity2contraction=None,
                    synonyms=None, replace_with_synonyms=False,
                    max_length=512, stride=8):
    token_data = []

    for item in tqdm(data, desc="Processing items"):
        text = item["text"]
        
        if task_prompt:
            text = f"{task_prompt}: {text}"
        
        encodings = tokenizer(text, 
                                return_offsets_mapping=True, 
                                truncation=True, 
                                padding=False,
                                max_length=max_length,
                                return_overflowing_tokens=True,
                                stride=stride)
        
        for i in range(len(encodings["input_ids"])):
            encoding = encodings[i]
            
            tokens_ = tokenizer.convert_ids_to_tokens(encoding.ids)
            offsets_ = encoding.offsets
            offset_offset = offsets_[1][0]

            labels_ = ["O"] * len(tokens_)
                
            text_chunk = tokenizer.decode(encoding.ids, skip_special_tokens=True)
            
            for entity_type, entity_list in item.get("entities", {}).items():
                if entity_type not in entity2contraction: continue
                for entity in entity_list:
                    matches = list(re.finditer(re.escape(entity), text_chunk, flags=re.IGNORECASE))
                    if not matches:
                        continue
                                    
                    # Find all matches for the entity
                    # non-overlapping entities in one-ish loop
                    matches = sorted(matches, key=lambda m: m.start())
                    idx = 0
                    
                    for match in matches:
                        start_char, end_char = match.span()
                        
                        entity_started = False
                        for i, (start, end) in enumerate(offsets_[idx:]):
                            start -= offset_offset
                            end -= offset_offset
                            
                            if start >= end_char:
                                idx += i+1
                                break
                            elif start >= start_char and end <= end_char:
                                if not entity_started:
                                    labels_[i+idx] = f"B-{entity2contraction[entity_type]}"
                                    entity_started = True
                                else:
                                    labels_[i+idx] = f"I-{entity2contraction[entity_type]}"

            if len(Counter(labels_)) == 1 and "O" in labels_:
                # If only "O" label is present, skip this item
                continue
            
            data_item = {
                "input_ids": encoding.ids,
                "attention_mask": encoding.attention_mask,
                "labels": [label2id[label] for label in labels_] if label2id else labels_
            }
            
            token_data.append(data_item)

    return Dataset.from_list(token_data)


def compute_metrics(eval_preds, LABELS):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    metric = evaluate.load("seqeval")

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[LABELS[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [LABELS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    metrics_dict = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
    
    return metrics_dict


def compute_weighted_loss(outputs, labels, return_outputs=False, num_items_in_batch=None, label_weights=None):
    logits = outputs.logits

    # Weighted CrossEntropyLoss
    loss_fct = nn.CrossEntropyLoss(
        weight=label_weights.to(logits.device),
        ignore_index=-100
    )

    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return (loss, outputs) if return_outputs else loss


def process_item(item):
    text = item["context"]
    entities = item.get("entities", 
                        {"human-powered": [], 
                         "animal-powered": [], 
                         "railways": [], 
                         "roadways": [], 
                         "water_transport": [], 
                         "air_transport": []})
    
    return {
        "text": text,
        "entities": entities
    }
    

def train(args):
    ENTITY2CONTRACTION = json.loads(args.entity2contraction)
    
    # Model
    LABELS = ["O"]
    for entity in ENTITY2CONTRACTION.values():
        LABELS.extend([f"B-{entity}", f"I-{entity}"])
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    
    with open(args.synonyms_file) as f:
        SYNONYMS = json.load(f)
    
    if not args.tokenizer_name:
        args.tokenizer_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, 
                                                add_prefix_space=args.add_prefix_space, use_fast=args.use_fast_tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(LABELS))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model.config.label2id = LABEL2ID
    model.config.id2label = {i: label for i, label in enumerate(LABELS)}

    if args.freeze_except_classifier:
        for param in model.base_model.parameters():
            param.requires_grad = False
        if hasattr(model, "classifier"):
            model.classifier.requires_grad_(True)
        elif hasattr(model, "score"):
            model.score.requires_grad_(True)
        elif hasattr(model, "lm_head"):
            model.lm_head.requires_grad_(True)
        else:
            raise ValueError("Model does not have a classifier or score layer to train.")
    elif args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="all",
        )
        model = get_peft_model(model, peft_config)

        if args.train_classifier:
            if hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, "score"):
                for param in model.score.parameters():
                    param.requires_grad = True
            elif hasattr(model, "lm_head"):
                for param in model.lm_head.parameters():
                    param.requires_grad = True
            else:
                raise ValueError("Model does not have a classifier or score layer to train.")
            
    # Data
    train_set = None
    dev_set = None
    test_set = None

    data_path = args.data_path
    print(f"Loading data from {data_path}", file=sys.stderr)
    
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            task_prompt = f.read().strip()
    elif args.task_prompt:
        task_prompt = args.task_prompt
    else:
        task_prompt = None
        
    raw_data = []
    datasets = []
        
    split_filename_dict = {
        "train": "train.json",
        "dev": "dev.json",
        "test": "test.json"
    }
    for split, filename in split_filename_dict.items():
        if os.path.exists(os.path.join(data_path, filename)):
            print(f"Loading {split} data from {os.path.join(data_path, filename)}", file=sys.stderr)
        
            with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Number of {split} items: {len(data)}", file=sys.stderr)
                
            for item in data:
                raw_data.append(process_item(item))
                
            dataset = prepare_dataset(raw_data, tokenizer, LABEL2ID, 
                                        entity2contraction=ENTITY2CONTRACTION,
                                        synonyms=SYNONYMS,
                                        replace_with_synonyms=args.replace_with_synonyms,
                                        task_prompt=task_prompt,
                                        max_length=args.model_max_length)

            datasets.append(dataset)

            print(f"{split} set size: {len(dataset)}", file=sys.stderr)
            
    train_set, dev_set, test_set = (datasets + [None, None, None])[:3]

    # Training
    os.makedirs(args.save_path, exist_ok=True)
        
    training_args = TrainingArguments(
        output_dir=args.save_path,
        save_total_limit=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        logging_strategy=args.logging_strategy,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"factor": args.lr_scheduler_factor, 
                             "patience": args.lr_scheduler_patience, 
                             "threshold": args.lr_scheduler_threshold, 
                             "min_lr": args.lr_scheduler_min_lr} if args.lr_scheduler_type == "reduce_lr_on_plateau" else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        bf16=args.bf16,
        auto_find_batch_size=True,
        report_to="mlflow" if args.use_mlflow else None,
    )

    num_labels = len(LABEL2ID)
    label_weights = torch.ones(num_labels)
    label_weights[LABEL2ID["O"]] = 0.05  # downweight "O" class
    
    compute_metrics_partial = partial(compute_metrics, LABELS=LABELS)
    compute_loss_partial = partial(compute_weighted_loss, label_weights=label_weights)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial,
        compute_loss_func=compute_loss_partial,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience else None
    )

    if args.use_mlflow:
        mlflow.set_tracking_uri(f"{args.mlruns_path}")
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.autolog()
        
        if not args.mlflow_run_name:
            args.mlflow_run_name = args.model_name.split("/")[-1]
        with mlflow.start_run(run_name=args.mlflow_run_name):
            trainer.train()
    else:
        trainer.train()

    model_save_path = Path(args.save_path) / "best_model"
    model_save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Evaluate on test set
    if args.use_mlflow:
        mlflow.set_experiment(f"{args.mlflow_experiment_name}-eval")
        with mlflow.start_run(run_name=args.mlflow_run_name):
            test_results = trainer.evaluate(test_set)
    else:
        test_results = trainer.evaluate(test_set)
    print(f"Test results: {test_results}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for organization extraction")
    # model
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-large", help="Model name or path")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name or path (if different from model)")
    parser.add_argument("--model-max-length", type=int, default=512, help="Maximum sequence length for the model")
    parser.add_argument("--add-prefix-space", action="store_true", help="Add prefix space to tokenizer")
    parser.add_argument("--use-fast-tokenizer", action="store_true", help="Use fast tokenizer")
    
    # data
    parser.add_argument("--data-path", type=str, default="../data/extraction/corrected", help="Path to data files")
    parser.add_argument("--save-path", type=str, default="../models/", help="Path to save the model")
    
    # task
    parser.add_argument("--task-prompt", type=str, default=None, help="Task prompt to prepend to text")
    parser.add_argument("--instruction-file", type=str, default=None, help="File containing task instructions to prepend to text")
    parser.add_argument("--synonyms-file", type=str, default="../data/synonyms.json")
    parser.add_argument("--replace-with-synonyms", "-rws", action="store_true")
    parser.add_argument("--entity2contraction", default='{"human-powered": "HUMAN", "animal-powered": "ANIMAL", "railways": "RAIL", "roadways": "ROAD", "water_transport": "WATER", "air_transport": "AIR"}')
    
    # training
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--metric-for-best-model", type=str, default=None, choices=["f1", "precision", "recall"], help="Metric to use for selecting the best model")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Patience for early stopping")
    ## scheduler
    parser.add_argument("--lr-scheduler-type", type=str, default="reduce_lr_on_plateau", choices=["linear", "cosine", "reduce_lr_on_plateau"], help="Learning rate scheduler type")
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help="Patience for learning rate scheduler")
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1e-3, help="Threshold for learning rate scheduler")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="Factor for learning rate scheduler")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-5, help="Minimum learning rate for scheduler")
    
    # eval
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--eval-strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--eval-steps", type=int, default=100, help="Number of steps between evaluations")
    parser.add_argument("--logging-strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Logging strategy")
    parser.add_argument("--logging-steps", type=int, default=20, help="Number of steps between logging")
    
    # peft
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training")
    ## lora
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None, help="Target modules for LoRA, e.g. ['query', 'key', 'value']")
    ## freeze
    parser.add_argument("--freeze-except-classifier", action="store_true", help="Freeze all layers except the classifier")
    parser.add_argument("--train-classifier", action="store_true", help="Train only the classifier layer")
    
    # mlflow
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow for tracking")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Name of the MLflow run")
    parser.add_argument("--mlflow-experiment-name", type=str, default="Mobility_Extraction", help="Name of the MLflow experiment")
    parser.add_argument("--mlruns-path", type=str, default="../mlruns", help="Path to MLflow runs directory")
    args = parser.parse_args()
    
    train(args)