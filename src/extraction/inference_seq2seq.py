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
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, EarlyStoppingCallback
from peft import get_peft_model, LoraConfig, TaskType

import mlflow


ENTITY2CONTRACTION = {
                        "human-powered": "HUMAN",
                        "animal-powered": "ANIMAL",
                        "railways": "RAIL",
                        "roadways": "ROAD",
                        "water_transport": "WATER",
                        "air_transport": "AIR"
                    }


def preprocess_function_inference(examples, tokenizer, 
                        delimiter="###", task_prompt=None, metadata=None,
                        source_max_length=512, target_max_length=128, stride=64):    
    inputs = []
    chunk_ids = []
    etypes = []
    
    for id_, item in enumerate(tqdm(examples, desc="Processing items")):    
        if "id" in item and item["id"]:
            id_ = item["id"]    
            
        original_text = item["text"]
        entities = item["entities"]
        original_text = re.sub(r"\s+", " ", original_text)
        
        for entity_type, entity_list in entities.items():    
            text = f"{task_prompt}\n\nText:{original_text}\n\nAnswer:" if task_prompt else text
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
                chunk_ids.append(id_)
                etypes.append(entity_type)

    return Dataset.from_list(inputs), chunk_ids, etypes


def preprocess_function(examples, tokenizer, 
                        delimiter="###", task_prompt=None, 
                        source_max_length=512, target_max_length=64, stride=64):    
    inputs = []
    
    for id_, item in enumerate(tqdm(examples, desc="Processing items")):    
        if "id" in item and item["id"]:
            id_ = item["id"]    
            
        text = item["text"]
        entities = item["entities"]
        
        encodings = tokenizer(text, 
                             truncation=True, max_length=source_max_length, padding=False,
                             return_overflowing_tokens=True, stride=stride
                             )
        
        for input_ids, attn in zip(encodings["input_ids"], encodings["attention_mask"]):
            labels_ = []
                
            text_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
            
            labels_ = [e for e in set(entities) if re.findall(re.escape(e), text_chunk)]
            
            label_ = delimiter.join(labels_)
            label = tokenizer(label_, truncation=True, max_length=target_max_length, padding=False)            
            
            inputs.append({
                "input_ids": input_ids,
                "attention_mask": attn,
                "labels": label["input_ids"]
            })

    return Dataset.from_list(inputs)


def compute_metrics(eval_preds, tokenizer=None, delimiter="###"):
    preds, labels = eval_preds
    
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    pred_entities_bow = [set(text.split(delimiter)) for text in pred_texts]

    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
    label_entities_bow = [set(text.split(delimiter)) for text in label_texts]

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
        "exact_set_match": float(np.mean(exact)),
        "jaccard": float(np.mean(jaccs)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "count_mae": float(np.mean(count_errs)),
    }
    

def preprocess_logits_for_metrics_fn(logits, labels):
    # logits can be a tuple for some models
    if isinstance(logits, tuple):
        logits = logits[0]
    # return token IDs instead of full logits to save memory
    return logits.argmax(dim=-1)


def inference(args):
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
        
    train_raw_data = []
    dev_raw_data = []
    test_raw_data = []
    full_raw_data = []
    
    train_chunk_ids = []
    dev_chunk_ids = []
    test_chunk_ids = []
    full_chunk_ids = []
    
    if args.instruction_file:
        with open(args.instruction_file, "r", encoding="utf-8") as f:
            task_prompt = f.read().strip()
    elif args.task_prompt:
        task_prompt = args.task_prompt
    else:
        task_prompt = None

    partial_preprocess_function = partial(preprocess_function, 
                                          tokenizer=tokenizer, task_prompt=task_prompt, 
                                          delimiter=args.delimiter, stride=args.stride,
                                          source_max_length=args.source_max_length, target_max_length=args.target_max_length)
    partial_preprocess_function_inference = partial(preprocess_function_inference, 
                                          tokenizer=tokenizer, task_prompt=task_prompt, metadata=metadata,
                                          delimiter=args.delimiter, stride=args.stride,
                                          source_max_length=args.source_max_length, target_max_length=args.target_max_length)
        
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
                train_set = partial_preprocess_function(train_raw_data)
            else:
                train_set, train_chunk_ids = partial_preprocess_function_inference(train_raw_data)
            
            print(f"Training set size: {len(train_set)}", file=sys.stderr)
        if os.path.exists(os.path.join(data_path, "dev.json")):
            print(f"Loading development data from {os.path.join(data_path, 'dev.json')}", file=sys.stderr)
            
            with open(os.path.join(data_path, "dev.json"), "r", encoding="utf-8") as f:
                dev_data = json.load(f)
                
            print(f"Number of development items: {len(dev_data)}", file=sys.stderr)
                        
            for item in dev_data:
                text = item["context"]
                entities = item.get("entities", {})
                id_ = item["id"] if "id" in item else None
                
                dev_raw_data.append({
                    "text": text,
                    "entities": entities,
                    "id": id_
                })
            if args.eval_full:
                full_raw_data.extend(dev_raw_data)
                
            if args.run_type.lower() == "eval":
                dev_set = partial_preprocess_function(dev_raw_data)
            else:
                dev_set, dev_chunk_ids = partial_preprocess_function_inference(dev_raw_data)
            
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
            test_set = partial_preprocess_function(test_raw_data)
        else:
            test_set, test_chunk_ids, test_etypes = partial_preprocess_function_inference(test_raw_data)
        
        print(f"Test set size: {len(test_set)}", file=sys.stderr)
        
    if test_set is None:
        raise ValueError("No dev or test set provided")
    if args.eval_full and full_raw_data:
        if args.run_type.lower() == "eval":
            full_set = partial_preprocess_function(full_raw_data)
        else:
            full_set, full_chunk_ids, full_etypes = partial_preprocess_function_inference(full_raw_data)
        print(f"Full set size: {len(full_set)}", file=sys.stderr)

    # Training
    os.makedirs(args.save_path, exist_ok=True)
        
    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        bf16=args.bf16,
        bf16_full_eval=args.bf16_full_eval,
        auto_find_batch_size=True,
        remove_unused_columns=False,
        predict_with_generate=True,
        generation_num_beams=args.generation_num_beams,
        generation_max_length=args.target_max_length
    )

    compute_metrics_partial = partial(compute_metrics, tokenizer=tokenizer, delimiter=args.delimiter)
    
    # Evaluate on test set
    if args.run_type.lower() == "eval":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_partial,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics_fn
        )
        
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
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        if args.eval_full:
            set_ = full_set
            chunk_ids_ = full_chunk_ids
            etypes_ = full_etypes
        else:
            set_ = test_set
            chunk_ids_ = test_chunk_ids
            etypes_ = test_etypes

        # Get predictions for chunks
        predictions, _, _ = trainer.predict(set_)
        predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        if args.debug:
            print(decoded_preds, file=sys.stderr)
        
        preds = []
        for pred_ in decoded_preds:
            prediction_ = pred_.split(args.delimiter)
            prediction_ = list(filter(lambda x: x.strip(), prediction_))
            preds.append(prediction_)

        # Combine predictions for chunks to predictions for whole documents
        full_preds = []
        cur_id = None
        cur_pred = dict.fromkeys(ENTITY2CONTRACTION.keys(), [])
        for id_, etype_, pred_ in zip(chunk_ids_, etypes_, preds):
            if cur_id is None:
                cur_id = id_
            if id_ != cur_id:
                full_preds.append((cur_id, cur_pred))
                cur_pred = dict.fromkeys(ENTITY2CONTRACTION.keys(), [])
                cur_id = id_

            cur_pred[etype_] = list(set(cur_pred[etype_] + pred_))
            cur_pred[etype_] = list(filter(lambda x: x.strip(), cur_pred[etype_]))
        if cur_pred:
            full_preds.append((cur_id, cur_pred))

        # Save predictions for full documents
        predictions_path = os.path.join(args.save_path, "predictions.jsonl")
            
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as fp:
            for i in range(len(full_preds)):
                id_, preds_ = full_preds[i]
                    
                json.dump({"id": id_, "prediction": preds_}, fp, ensure_ascii=False)
                print(file=fp)

        print("Inference finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for organization extraction")
    
    parser.add_argument("--run-type", type=str, default="eval", help="Run type (eval, inference)")

    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer name or path")
    parser.add_argument("--model-name", type=str, default="../../../models/extraction/conll03/ner-org-finetuned", help="Model name or path")
    parser.add_argument("--data-path", type=str, default="../../../data/ocr/TISS/tesseract_ocr_text", help="Path to data files")
    parser.add_argument("--save-path", type=str, default="../../../data/extraction/predictions/", help="Path to predictions")
    
    parser.add_argument("--instruction-file", type=str, default=None, help="Path to instruction file")
    parser.add_argument("--task-prompt", type=str, default=None, help="Task prompt to prepend to text")
    parser.add_argument("--entity-metadata-file", type=str, default="../../../data/entity_metadata.json", help="Path to entity metadata JSON file")
    
    parser.add_argument("--add-prefix-space", action="store_true", help="Add prefix space to tokenizer")
    parser.add_argument("--use-fast-tokenizer", action="store_true", help="Use fast tokenizer")
    parser.add_argument("--delimiter", type=str, default="###", help="Delimiter for entity predictions")
    parser.add_argument("--type-delimiter", type=str, default="===", help="Delimiter for entity type")
    parser.add_argument("--source-max-length", type=int, default=512, help="Maximum source sequence length")
    parser.add_argument("--target-max-length", type=int, default=128, help="Maximum target sequence length")
    parser.add_argument("--stride", type=int, default=64, help="Stride for chunking input sequences")
    
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision for training")
    parser.add_argument("--bf16-full-eval", action="store_true", help="Use bfloat16 precision for full evaluation")
    
    parser.add_argument("--eval-full", action="store_true", help="Run full evaluation")
    
    parser.add_argument("--freeze-except-classifier", action="store_true", help="Freeze all layers except the classifier")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for training")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora-target-modules", type=str, default=None, help="LoRA target modules (comma-separated)")

    parser.add_argument("--generation-num-beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--debug", action="store_true", help="Debug mode with smaller data subset")

    args = parser.parse_args()
    
    inference(args)