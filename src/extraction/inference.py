from functools import partial
import os
import json
import re
import argparse
from collections import Counter
import sys

from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import Dataset
from transformers import DataCollatorForTokenClassification
import numpy as np
import evaluate
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType


def prepare_dataset_inference(data, tokenizer,
                              label2id=None, task_prompt=None,
                              entity2contraction=None, eval_empty=False,
                              max_length=512, stride=64):
    token_data = []
    chunk_ids = []

    for id_, item in enumerate(tqdm(data, desc="Processing items")):
        if "id" in item and item["id"]:
            id_ = item["id"]
        
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
            
            data_item = {
                "input_ids": encoding.ids,
                "attention_mask": encoding.attention_mask
            }
            
            token_data.append(data_item)
            chunk_ids.append(id_)

    return Dataset.from_list(token_data), chunk_ids


def prepare_dataset(data, tokenizer, 
                    label2id=None, task_prompt=None, 
                    entity2contraction=None, eval_empty=False,
                    max_length=512, stride=64):
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

            if not eval_empty and (len(Counter(labels_)) == 1) and ("O" in labels_):
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


def inference(args):
    ENTITY2CONTRACTION = json.loads(args.entity2contraction)
    
    # Model
    LABELS = ["O"]
    for entity in ENTITY2CONTRACTION.values():
        LABELS.extend([f"B-{entity}", f"I-{entity}"])
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    num_labels = len(LABEL2ID)
    
    if not args.tokenizer_name:
        args.tokenizer_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, 
                                              add_prefix_space=args.add_prefix_space, use_fast=args.use_fast_tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=num_labels)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    model.config.label2id = LABEL2ID
    model.config.id2label = {i: label for i, label in enumerate(LABELS)}
    
    label_weights = torch.ones(num_labels)
    label_weights[LABEL2ID["O"]] = 0.05  # downweight "O" class

    # Data
    train_set = None
    dev_set = None
    test_set = None
    
    train_chunk_ids = []
    dev_chunk_ids = []
    test_chunk_ids = []

    data_path = args.data_path
    print(f"Loading data from {data_path}", file=sys.stderr)
    
    train_raw_data = []
    dev_raw_data = []
    test_raw_data = []
    full_raw_data = []
    
    if args.eval_full:
        if os.path.exists(os.path.join(data_path, "train.json")):
            print(f"Loading training data from {os.path.join(data_path, 'train.json')}", file=sys.stderr)
            
            with open(os.path.join(data_path, "train.json"), "r", encoding="utf-8") as f:
                train_data = json.load(f)
                
            print(f"Number of training items: {len(train_data)}", file=sys.stderr)
                        
            for item in train_data:
                text = item["context"]
                entities = item.get("entities", {})
                id_ = item["id"] if "id" in item else None

                train_raw_data.append({
                    "text": text,
                    "entities": entities,
                    "id": id_
                })

            if args.eval_full:
                full_raw_data.extend(train_raw_data)

            if args.run_type.lower() == "eval":
                train_set = prepare_dataset(train_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt, eval_empty=args.eval_empty)
            else:
                train_set, train_chunk_ids = prepare_dataset_inference(train_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt)
            print(f"Training set size: {len(train_set)}", file=sys.stderr)
        if os.path.exists(os.path.join(data_path, "dev.json")):
            print(f"Loading development data from {os.path.join(data_path, 'dev.json')}", file=sys.stderr)
            
            with open(os.path.join(data_path, "dev.json"), "r", encoding="utf-8") as f:
                dev_data = json.load(f)
                
            print(f"Number of development items: {len(dev_data)}", file=sys.stderr)
                        
            for item in dev_data:
                text = item["context"]
                entities = item.get("entities", {} )
                id_ = item["id"] if "id" in item else None
                
                dev_raw_data.append({
                    "text": text,
                    "entities": entities,
                    "id": id_
                })
                
            if args.eval_full:
                full_raw_data.extend(dev_raw_data)
                
            if args.run_type.lower() == "eval":
                dev_set = prepare_dataset(dev_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt, eval_empty=args.eval_empty)
            else:
                dev_set, dev_chunk_ids = prepare_dataset_inference(dev_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt)
            print(f"Development set size: {len(dev_set)}", file=sys.stderr)
    if os.path.exists(os.path.join(data_path, "test.json")):
        print(f"Loading test data from {os.path.join(data_path, 'test.json')}", file=sys.stderr)
        
        with open(os.path.join(data_path, "test.json"), "r", encoding="utf-8") as f:
            test_data = json.load(f)
            
        print(f"Number of test items: {len(test_data)}", file=sys.stderr)
                    
        for item in test_data:
            text = item["context"]
            entities = item.get("entities", {})
            id_ = item["id"] if "id" in item else None
            
            test_raw_data.append({
                "text": text,
                "entities": entities,
                "id": id_
            })
            
        if args.eval_full:
            full_raw_data.extend(test_raw_data)
            
        if args.run_type.lower() == "eval":
            test_set = prepare_dataset(test_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt, eval_empty=args.eval_empty)
        else:
            test_set, test_chunk_ids = prepare_dataset_inference(test_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt)
        print(f"Test set size: {len(test_set)}", file=sys.stderr)

    if test_set is None:
        raise ValueError("No dev or test set provided")
    if args.eval_full and full_raw_data:
        if args.run_type.lower() == "eval":
            full_set = prepare_dataset(full_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt, eval_empty=args.eval_empty)
        else:
            full_set, full_chunk_ids = prepare_dataset_inference(full_raw_data, tokenizer, LABEL2ID, task_prompt=args.task_prompt)
        print(f"Full set size: {len(full_set)}", file=sys.stderr)

    # Training
    os.makedirs(args.save_path, exist_ok=True)
        
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=args.bf16,
        bf16_full_eval=args.bf16_full_eval,
        auto_find_batch_size=True,
        remove_unused_columns=False
    )

    compute_metrics_partial = partial(compute_metrics, LABELS=LABELS)
    compute_loss_partial = partial(compute_weighted_loss, label_weights=label_weights)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_partial,
        compute_loss_func=compute_loss_partial,
    )
    
    # Evaluate on test set
    if args.run_type.lower() == "eval":
        test_results = trainer.evaluate(test_set)
        print(f"Test results: {test_results}")
        
        if args.eval_full:
            train_results = trainer.evaluate(train_set)
            print(f"Train results: {train_results}")
            
            dev_results = trainer.evaluate(dev_set)
            print(f"Dev results: {dev_results}")

            full_results = trainer.evaluate(full_set)
            print(f"Full results: {full_results}")

        print("Evaluation finished")
    elif args.run_type.lower() == "inference":
        if args.eval_full:
            set_ = full_set
            chunk_ids_ = full_chunk_ids
            
        else:
            set_ = test_set
            chunk_ids_ = test_chunk_ids
            
        predictions = trainer.predict(set_)

        predictions_path = os.path.join(args.save_path, "predictions.jsonl")

        pred_list = predictions.predictions.tolist()

        # Get predictions for chunks
        preds = []
        for i, example in enumerate(set_):
            pred_ = pred_list[i]

            # Convert predicted label ids to label names
            pred_labels = [LABELS[np.argmax(p, axis=-1)] for p in pred_]

            # Get tokens for this example
            tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])

            # Helper to extract entities from BIO labels
            def extract_entities(tokens, labels, entity_type="ORG"):
                entities = []
                entity = []
                for token, label in zip(tokens, labels):
                    if label == f"B-{entity_type}":
                        if entity:
                            entities.append(" ".join(entity))
                            entity = []
                        entity = [token]
                    elif label == f"I-{entity_type}" and entity:
                        entity.append(token)
                    else:
                        if entity:
                            entities.append(" ".join(entity))
                            entity = []
                if entity:
                    entities.append(" ".join(entity))
                # Clean up tokenization artifacts
                entities = [tokenizer.convert_tokens_to_string(e.split()) for e in entities]
                return entities

            preds_ = dict.fromkeys(ENTITY2CONTRACTION.keys(), [])
            for entity_type, entity_short_type in ENTITY2CONTRACTION.items():
                preds_entities = extract_entities(tokens, pred_labels, entity_type=entity_short_type)
                preds_entities = list(set(preds_entities))
                preds_[entity_type] = preds_entities
                
            preds.append(preds_)
            
        # Combine predictions for chunks to predictions for whole documents
        full_preds = []
        cur_id = None
        cur_pred = dict.fromkeys(ENTITY2CONTRACTION.keys(), [])
        for id_, pred_ in zip(chunk_ids_, preds):
            if cur_id is None:
                cur_id = id_
            if id_ != cur_id:
                full_preds.append((cur_id, cur_pred))
                cur_pred = dict.fromkeys(ENTITY2CONTRACTION.keys(), [])
                cur_id = id_
            # cur_pred.extend(pred_)
            for entity_type, preds_entities in pred_.items():
                cur_pred[entity_type] = list(set(cur_pred[entity_type] + preds_entities))
        if cur_pred:
            full_preds.append((cur_id, cur_pred))

        # Save predictions for full documents
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as fp:
            for i in range(len(full_preds)):
                id_, preds_ = full_preds[i]
                
                json.dump({"id": id_, "entities": preds_}, fp, ensure_ascii=False)
                print(file=fp)

        print("Inference finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for organization extraction")
    parser.add_argument("--model-name", type=str, default="../../../models/extraction/conll03/ner-org-finetuned", help="Model name or path")
    parser.add_argument("--data-path", type=str, default="../../../data/ocr/TISS/tesseract_ocr_text", help="Path to data files")
    parser.add_argument("--save-path", type=str, default="../../../data/extraction/predictions/", help="Path to predictions")
    parser.add_argument("--run-type", type=str, default="eval", help="Run type (eval, inference)")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument("--add-prefix-space", action="store_true", help="Add prefix space to tokenizer")
    parser.add_argument("--use-fast-tokenizer", action="store_true", help="Use fast tokenizer")
    parser.add_argument("--task-prompt", type=str, default=None, help="Task prompt to prepend to text")
    parser.add_argument("--entity2contraction", default='{"human-powered": "HUMAN", "animal-powered": "ANIMAL", "railways": "RAIL", "roadways": "ROAD", "water_transport": "WATER", "air_transport": "AIR"}')
    parser.add_argument("--eval-empty", action="store_true")
    
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training")
    parser.add_argument("--bf16-full-eval", action="store_true", help="Use bfloat16 precision for full evaluation")

    parser.add_argument("--eval-full", action="store_true", help="Run full evaluation")
    
    args = parser.parse_args()
    
    inference(args)