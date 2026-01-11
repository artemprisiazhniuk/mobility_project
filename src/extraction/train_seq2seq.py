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
import re
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments, Seq2SeqTrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType

import mlflow


def preprocess_function(examples, tokenizer, delimiter="###", task_prompt=None, metadata=None,
                        source_max_length=512, target_max_length=128, stride=16):    
    inputs = []
    
    for text, entities in zip(examples["text"], examples["entities"]):    
        for entity_type, entity_list in entities.items():    
            text = f"{task_prompt}\n\nText:{text}\n\nAnswer:" if task_prompt else text
            text = text.replace('{{{entity}}}', entity_type)
            if '{{{definition}}}' in text:
                text = text.replace('{{{definition}}}', metadata[entity_type]["definition"] if metadata and entity_type in metadata else "")
            if '{{{guidelines}}}' in text:
                text = text.replace('{{{guidelines}}}', metadata[entity_type]["guidelines"] if metadata and entity_type in metadata else "")
            
            encodings = tokenizer(text, 
                                truncation=True, max_length=source_max_length, padding=False,
                                return_overflowing_tokens=True, stride=stride
                                )
            
            for input_ids, attn in zip(encodings["input_ids"], encodings["attention_mask"]):                
                text_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
            
                labels_per_type = [e for e in set(entity_list) if re.findall(re.escape(e), text_chunk)]
                label_ = delimiter.join(labels_per_type)
                
                label = tokenizer(label_, truncation=True, max_length=target_max_length, padding=False)            
            
                inputs.append({
                    "input_ids": input_ids,
                    "attention_mask": attn,
                    "labels": label["input_ids"]
                })

    final_inputs = {
        "input_ids": [i["input_ids"] for i in inputs],
        "attention_mask": [i["attention_mask"] for i in inputs],
        "labels": [i["labels"] for i in inputs],
    }

    return final_inputs


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


def compute_metrics(eval_preds, tokenizer=None, delimiter="###"):
    preds, labels = eval_preds
    
    # Decode predictions and labels
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    pred_entities_bow = [set(text.split(delimiter)) for text in pred_texts]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    label_entities_bow = [set(text.split(delimiter)) for text in label_texts]

    # Calculate metrics
    exact, jaccs, tp, fp, fn, count_errs = [], [], 0, 0, 0, []
    for P, Y in zip(pred_entities_bow, label_entities_bow):
        inter = len(P & Y)
        union = len(P | Y)
        exact.append(1.0 if P == Y else 0.0)
        jaccs.append(0.0 if union == 0 else inter / union)
        tp += inter
        fp += len(P - Y)
        fn += len(Y - P)
        count_errs.append(abs(len(P) - len(Y)))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 0.0 if (precision + recall) == 0 else 2*precision*recall/(precision+recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_match": float(np.mean(exact))
    }


def preprocess_logits_for_metrics_fn(logits, labels):
    # logits can be a tuple for some models
    if isinstance(logits, tuple):
        logits = logits[0]
    # return token IDs instead of full logits to save memory
    return logits.argmax(dim=-1)


def train(args):
    with open(args.entity_metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    if args.freeze_except_classifier:
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.lm_head.requires_grad_(True)
    elif args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias="all",
        )
        model = get_peft_model(model, peft_config)

        if args.train_classifier:
            for param in model.lm_head.parameters():
                param.requires_grad = True

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

    partial_preprocess_function = partial(preprocess_function, 
                                          tokenizer=tokenizer, task_prompt=task_prompt, metadata=metadata,
                                          delimiter=args.delimiter, stride=args.stride,
                                          source_max_length=args.source_max_length, target_max_length=args.target_max_length)
        
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
                item_repr = process_item(item)
                raw_data.append(item_repr)
                
            dataset = Dataset.from_list(raw_data)
            dataset = dataset.map(partial_preprocess_function, batched=True, num_proc=4, remove_columns=["text", "entities"])
            datasets.append(dataset)
            
            print(f"{split} set size: {len(dataset)}", file=sys.stderr)
            
    train_set, dev_set, test_set = (datasets + [None, None, None])[:3]

    # Training
    os.makedirs(args.save_path, exist_ok=True)
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.save_path,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        num_train_epochs=args.num_epochs,
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
        group_by_length=args.group_by_length,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        bf16=args.bf16,
        bf16_full_eval=args.bf16_full_eval,
        auto_find_batch_size=True,
        report_to="mlflow" if args.use_mlflow else None
    )

    compute_metrics_partial = partial(compute_metrics, tokenizer=tokenizer, delimiter=args.delimiter)

    if args.disable_cache:
        model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if args.early_stopping_patience else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics_fn
    )

    if args.use_mlflow:
        mlflow.set_tracking_uri(f"{args.mlruns_path}")
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.autolog()
        
        if not args.mlflow_run_name:
            args.mlflow_run_name = args.model_name.split("/")[-1]
        with mlflow.start_run(run_name=args.mlflow_run_name):
            
            print("Model name:", args.model_name, file=sys.stderr)
            
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
    parser.add_argument("--model-name", type=str, default="FacebookAI/xlm-roberta-large-finetuned-conll03-english", help="Model name or path")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name or path (if different from model)")
    parser.add_argument("--data-path", type=str, default="../data/extraction/", help="Path to data files")
    parser.add_argument("--save-path", type=str, default="../models/", help="Path to save the model")
    
    parser.add_argument("--delimiter", type=str, default="###", help="Delimiter for entity sets in labels")
    parser.add_argument("--type-delimiter", type=str, default="===", help="Delimiter for entity types in labels")
    parser.add_argument("--instruction-file", type=str, default=None, help="File containing task instructions to prepend to text")
    parser.add_argument("--task-prompt", type=str, default=None, help="Task prompt to prepend to text")
    parser.add_argument("--entity-metadata-file", type=str, default="../data/extraction/entity_metadata.json", help="File containing entity definitions and guidelines")
    
    parser.add_argument("--source-max-length", type=int, default=512, help="Maximum sequence length for the model (if different from tokenizer's max length)")
    parser.add_argument("--target-max-length", type=int, default=64, help="Maximum sequence length for the model (if different from tokenizer's max length)")
    parser.add_argument("--stride", type=int, default=64, help="Stride for sliding window tokenization")
    parser.add_argument("--eval-strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Evaluation strategy")
    parser.add_argument("--eval-steps", type=int, default=50, help="Number of steps between evaluations")
    parser.add_argument("--logging-strategy", type=str, default="steps", choices=["no", "steps", "epoch"], help="Logging strategy")
    parser.add_argument("--logging-steps", type=int, default=20, help="Number of steps between logging")
    
    parser.add_argument("--disable-cache", action="store_true", help="Disable the use of cache in the model")

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--eval-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients during evaluation")
    parser.add_argument("--bf16-full-eval", action="store_true", help="Use bfloat16 for full evaluation")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--metric-for-best-model", type=str, default=None, choices=["f1", "precision", "recall"], help="Metric to use for selecting the best model")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Patience for early stopping")
    parser.add_argument("--generation-num-beams", type=int, default=3, help="Number of beams for generation during evaluation")
    parser.add_argument("--group_by_length", action="store_true", help="Group sequences by length for more efficient training")
    parser.add_argument("--predict-with-generate", action="store_true", help="Use generate to predict during evaluation")
    
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora-target-modules", type=str, nargs="+", default=None, help="Target modules for LoRA, e.g. ['query', 'key', 'value']")
    
    parser.add_argument("--lr-scheduler-type", type=str, default="reduce_lr_on_plateau", choices=["linear", "cosine", "reduce_lr_on_plateau"], help="Learning rate scheduler type")
    parser.add_argument("--lr-scheduler-patience", type=int, default=5, help="Patience for learning rate scheduler")
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1e-3, help="Threshold for learning rate scheduler")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5, help="Factor for learning rate scheduler")
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-5, help="Minimum learning rate for scheduler")
    
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training")
    parser.add_argument("--freeze-except-classifier", action="store_true", help="Freeze all layers except the classifier")
    parser.add_argument("--train-classifier", action="store_true", help="Train only the classifier layer")
    
    parser.add_argument("--add-prefix-space", action="store_true", help="Add prefix space to tokenizer")
    parser.add_argument("--use-fast-tokenizer", action="store_true", help="Use fast tokenizer")
    
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow for tracking")
    parser.add_argument("--mlflow-run-name", type=str, default=None, help="Name of the MLflow run")
    parser.add_argument("--mlflow-experiment-name", type=str, default="NER_Organization_Extraction", help="Name of the MLflow experiment")
    parser.add_argument("--mlruns-path", type=str, default="../mlruns", help="Path to MLflow runs directory")
    args = parser.parse_args()
    
    train(args)