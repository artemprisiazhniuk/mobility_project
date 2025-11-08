import argparse
import os
import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, \
                        TrainingArguments, Trainer, pipeline
from datasets import create_dataset
import evaluate
import numpy as np


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def get_tokenized_dataset(dataset, tokenizer):
    pass

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def main(args):
    #region setup
    model = AutoModelForTokenClassification.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    tokenized_dataset = get_tokenized_dataset(args.dataset, tokenizer)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    metric = evaluate.load("seqeval")
    
    training_args = TrainingArguments(
        "bert-finetuned-ner",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )
    #endregion
    
    #region training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    
    trainer.push_to_hub(commit_message="Training complete")
    #endregion

    #region eval
    # Replace this with your own checkpoint
    # model_checkpoint = "huggingface-course/bert-finetuned-ner"
    # token_classifier = pipeline(
    #     "token-classification", model=model_checkpoint, aggregation_strategy="simple"
    # )
    # token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    #endregion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence classification model.")
    parser.add_argument("--model-name", type=str, defualt="bert-base-uncased")
    args = parser.parse_args()
    
    main(args)