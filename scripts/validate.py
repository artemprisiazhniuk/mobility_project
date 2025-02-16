from collections import Counter, defaultdict
import json
import os
import argparse
import re

from transformers import AutoTokenizer
import evaluate
from tqdm import tqdm


def evaluate_extraction(true_entities_dict, predicted_entities_dict, tokenizer=None):
    true_entities_list = true_entities_dict.values()
    predicted_entities_list = predicted_entities_dict.values()
    
    true_entities_keys = list(true_entities_dict.keys())
    predicted_entities_keys = list(predicted_entities_dict.keys())
    
    assert len(true_entities_list) == len(predicted_entities_list), "The number of examples must match."

    # Collect all entity types appearing in any example
    entity_types = set()
    for true_dict in true_entities_list:
        entity_types.update(true_dict.keys())
    for pred_dict in predicted_entities_list:
        entity_types.update(pred_dict.keys())

    # Dictionary to store aggregated counts per entity type
    per_entity_counts = {etype: {"TP": 0, "FP": 0, "FN": 0} for etype in entity_types}
    fp = defaultdict(list)
    fn = defaultdict(list)

    # Iterate over all examples and aggregate counts for each entity type
    for true_dict, pred_dict, true_key, pred_key in zip(true_entities_list, predicted_entities_list, true_entities_keys, predicted_entities_keys):
        for etype in entity_types:
            # Get the list of entities for this type; default to empty list if missing.
            true_list = true_dict.get(etype, [])
            pred_list = pred_dict.get(etype, [])
            tmp = []
            for entity in pred_list:
                if isinstance(entity, str):
                    tmp.append(entity)
                elif "text" in entity:
                    tmp.append(entity["text"])
                elif "name" in entity:
                    tmp.append(entity["name"])
            pred_list = tmp
            pred_list = list(set(pred_list))
            
            # Use Counter to account for duplicates
            true_counter = Counter(true_list)
            pred_counter = Counter(pred_list)
            
            # Count true positives: for each entity present in both, add the minimum count
            common_entities = set(true_counter.keys()) & set(pred_counter.keys())
            TP = sum(min(true_counter[ent], pred_counter[ent]) for ent in common_entities)
            
            # Count false positives: predicted count minus the matched count for every predicted entity
            FP = sum(pred_counter[ent] - min(true_counter.get(ent, 0), pred_counter[ent]) for ent in pred_counter)
            
            # Count false negatives: true count minus the matched count for every true entity
            FN = sum(true_counter[ent] - min(true_counter[ent], pred_counter.get(ent, 0)) for ent in true_counter)
            
            # Aggregate counts
            per_entity_counts[etype]["TP"] += TP
            per_entity_counts[etype]["FP"] += FP
            per_entity_counts[etype]["FN"] += FN
            
            if FP > 0:
                fp[true_key].append({"entity": etype, "true": true_list, "pred": pred_list})
            if FN > 0:
                fn[true_key].append({"entity": etype, "true": true_list, "pred": pred_list})

    # Compute precision, recall, and F1 for each entity type
    per_entity_results = {}
    for etype, counts in per_entity_counts.items():
        TP = counts["TP"]
        FP = counts["FP"]
        FN = counts["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        per_entity_results[etype] = {"Precision": precision, "Recall": recall, "F1": f1}

    # Macro-average: average the metric scores over all entity types
    macro_precision = sum(result["Precision"] for result in per_entity_results.values()) / len(entity_types)
    macro_recall    = sum(result["Recall"] for result in per_entity_results.values()) / len(entity_types)
    macro_f1        = sum(result["F1"] for result in per_entity_results.values()) / len(entity_types)
    
    # Micro-average: aggregate counts over all entity types
    total_TP = sum(counts["TP"] for counts in per_entity_counts.values())
    total_FP = sum(counts["FP"] for counts in per_entity_counts.values())
    total_FN = sum(counts["FN"] for counts in per_entity_counts.values())

    micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    results = {
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Micro Precision": micro_precision,
        "Micro Recall": micro_recall,
        "Micro F1": micro_f1
    }
    results = {metric: [round(value, 3)] for metric, value in results.items()}
    
    return per_entity_results, results, fp, fn


def create_bio_tags(text, labels_dict, tokenizer):
    """
    Given a text string and a labels dictionary of the form:
       {"class1": [entity1, entity2, ...], "class2": [entity3, ...], ...}
    this function returns the tokenized text and a list of BIO tags.
    """
    # Collect all entity spans with their class label
    # Each span is represented as a dictionary with start/end offsets.
    entity_spans = []
    for label, entities in labels_dict.items():
        tmp = []
        for entity in entities:
            if isinstance(entity, str):
                tmp.append(entity)
            elif "text" in entity:
                tmp.append(entity["text"])
            elif "name" in entity:
                tmp.append(entity["name"])
        entities = tmp
        entities = list(set(entities))
        
        for entity in entities:
            # Use re.finditer to locate all occurrences of the entity in the text.
            # re.escape is used to avoid issues with special characters.
            for match in re.finditer(re.escape(entity), text):
                span = {"label": label, "start": match.start(), "end": match.end()}
                entity_spans.append(span)
    # Sort spans by start position (this helps if there are multiple entities)
    entity_spans = sorted(entity_spans, key=lambda x: x["start"])

    # Tokenize the text using the fast tokenizer to get offset mappings
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=False, padding=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    offset_mapping = encoding["offset_mapping"]
    
    # Initialize all tags as "O" (outside)
    bio_tags = ["O" for token in tokens if token not in tokenizer.all_special_tokens] * len(tokens)

    # For each entity, label the tokens whose offsets overlap with the entity span.
    for entity in entity_spans:
        ent_start = entity["start"]
        ent_end = entity["end"]
        entity_label = entity["label"]
        token_indices = []

        # Loop over the tokens’ offset mappings.
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            # Some fast tokenizers return (0, 0) for special tokens (e.g., [CLS], [SEP])
            if tok_start == tok_end == 0:
                continue
            # If the token lies completely after the entity span, we can break early.
            if tok_start >= ent_end:
                break
            # If the token overlaps with the entity span, record its index.
            if tok_end > ent_start and tok_start < ent_end:
                token_indices.append(i)
        
        # Assign BIO tags: first token gets "B-<label>", subsequent tokens get "I-<label>".
        if token_indices:
            bio_tags[token_indices[0]] = f"B-{entity_label}"
            for idx in token_indices[1:]:
                bio_tags[idx] = f"I-{entity_label}"
    
    return bio_tags


def evaluate_bio(true_entities_dict, predicted_entities_dict, tokenizer=None, texts_path=None):    
    texts = {}
    for filename in os.listdir(texts_path):
        if filename.endswith(".txt"):
            with open(os.path.join(texts_path, filename)) as f:
                texts[filename.split('.')[0]] = f.read()
                
    true_labels = []
    predicted_labels = []
                
    for id_ in tqdm(list(predicted_entities_dict.keys())):
        if id_ not in texts:
            raise ValueError(f"Text not found for ID: {id_}")
        true_entities = true_entities_dict[id_]
        predicted_entities = predicted_entities_dict[id_]
        
        true_bio = create_bio_tags(texts[id_], true_entities, tokenizer)
        pred_bio = create_bio_tags(texts[id_], predicted_entities, tokenizer)
        
        # Prune consecutive "O" tags into one
        pruned_true_bio = []
        pruned_pred_bio = []
        
        for i in range(len(true_bio)):
            if true_bio[i] == "O" and true_bio[i] == pred_bio[i] and (i == 0 or true_bio[i-1] != "O" or pred_bio[i-1] != "O"):
                pruned_true_bio.append("O")
                pruned_pred_bio.append("O")
            else:
                pruned_true_bio.append(true_bio[i])
                pruned_pred_bio.append(pred_bio[i])
        
        true_labels.append(pruned_true_bio)
        predicted_labels.append(pruned_pred_bio)
        
    # Compute metrics
    metric = evaluate.load("seqeval")
    
    metrics = metric.compute(predictions=predicted_labels, references=true_labels)
    
    metrics = {metric: [round(value, 3)] for metric, value in metrics.items() if "overall" in metric}
    
    return metrics


def main(args):
    with open(os.path.join(args.automatic_folder, args.automatic_file)) as f:
        labels_automatic = [json.loads(line) for line in f]
        
    args.automatic_file = re.sub(r"_\d+?(?=.jsonl)", "", args.automatic_file)
        
    labels_automatic_dict = {label["id"]: label["entities"] for label in labels_automatic} 
    
    with open(args.manual_file) as f:
        labels_manual = [json.loads(line) for line in f]
    
    labels_manual_dict = {label["id"]: label["entities"] for label in labels_manual if label["id"] in labels_automatic_dict}    
    
    labels_manual_dict = dict(sorted(labels_manual_dict.items()))
    labels_automatic_dict = dict(sorted(labels_automatic_dict.items()))
    
    if args.eval_type == "extraction":
        per_entity_results, results, fp_array, fn_array = evaluate_extraction(labels_manual_dict, labels_automatic_dict)

        print("Per Entity Type Metrics:")
        for etype, metrics in per_entity_results.items():
            print(f"  {etype}: {metrics}")

        print("\nAveraged Metrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
    
        metrics = dict()
        if os.path.exists(os.path.join(args.output_folder, "metrics_extr.json")):
            metrics = json.load(open(os.path.join(args.output_folder, "metrics_extr.json")))
            
        with open(os.path.join(args.output_folder, "metrics_extr.json"), "w") as f:
            if args.automatic_file.split('.')[0] in metrics:
                for key in metrics[args.automatic_file.split('.')[0]]:
                    value = metrics[args.automatic_file.split('.')[0]][key]
                    other_value = results[key]
                    
                    metrics[args.automatic_file.split('.')[0]][key] = value + other_value
            else:
                metrics[args.automatic_file.split('.')[0]] = results
            json.dump(metrics, f, indent=4)
            
        with open(os.path.join(args.output_folder, "errors", f"{args.automatic_file.split('.')[0]}_fp.json"), "w") as f:
            json.dump(fp_array, f, indent=4)
        with open(os.path.join(args.output_folder, "errors", f"{args.automatic_file.split('.')[0]}_fn.json"), "w") as f:
            json.dump(fn_array, f, indent=4)
            
    elif args.eval_type == "bio":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        texts_path = args.texts_folder
        results = evaluate_bio(labels_manual_dict, labels_automatic_dict, tokenizer, texts_path)
        
        print("\nMetrics:")
        for metric, value in results.items():
            print(f"  {metric}: {value}")
        
        metrics = dict()
        if os.path.exists(os.path.join(args.output_folder, "metrics_bio.json")):
            metrics = json.load(open(os.path.join(args.output_folder, "metrics_bio.json")))
            
        with open(os.path.join(args.output_folder, "metrics_bio.json"), "w") as f:
            if args.automatic_file.split('.')[0] in metrics:
                for key in metrics[args.automatic_file.split('.')[0]]:
                    value = metrics[args.automatic_file.split('.')[0]][key]
                    other_value = results[key]
                    
                    metrics[args.automatic_file.split('.')[0]][key] = value + other_value
            else:
                metrics[args.automatic_file.split('.')[0]] = results
            json.dump(metrics, f, indent=4)       
    else:
        raise ValueError(f"Invalid evaluation type: {args.eval_type}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NER performance.")
    parser.add_argument("--automatic-folder", type=str, default="./data/labels_automatic")
    parser.add_argument("-af", "--automatic-file", type=str, required=True)
    parser.add_argument("--manual-file", type=str, default="./data/labels_manual.jsonl")
    parser.add_argument("--output-folder", type=str, default="./data/analysis")
    parser.add_argument("--eval-type", type=str, default="bio") # bio, extraction
    parser.add_argument("--model-name", type=str, default="bert-base-cased")
    parser.add_argument("--texts-folder", type=str, default="./data/lyrics")
    args = parser.parse_args()
    
    main(args)