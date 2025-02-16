import os
import json
import re
import random
import time
import argparse

import dotenv
from pydantic import BaseModel
from openai import OpenAI
import stanza
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm


dotenv.load_dotenv()
# stanza.download('en')

client = OpenAI()
nlp = stanza.Pipeline('en', processors='tokenize,pos')

class ResponseSchema(BaseModel):
    entities: dict[str, list[str]]
    
SLEEP_TIME = 1

def majority_vote(results):
    counts = {}
    for result in results:
        result_str = json.dumps(result)
        if result_str not in counts:
            counts[result_str] = 1
        else:
            counts[result_str] += 1
    
    max_count = max(counts.values())
    majority_results = [json.loads(result_str) for result_str in counts if counts[result_str] == max_count]
    
    return random.choice(majority_results)

def run_query(system_prompt, user_prompt, 
              self_consistency_num=1,
              response_type="json_object", json_schema=None):
    format_ = {"type": response_type} if response_type == "json_object" else {"type": response_type, "json_schema": json_schema}
    
    if response_type == "json_object":
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=format_,
            n=self_consistency_num
        )
        if self_consistency_num > 1:
            result = dict()
            choices = [json.loads(c.message.content) for c in completion.choices]
            for entity_name in choices[0].keys():
                results = [c[entity_name] for c in choices]
                result_ = majority_vote(results)
                
                result[entity_name] = result_
        else:
            result = completion.choices[0].message.content
            if response_type == "json_object": result = json.loads(result)    
    elif response_type == "json_schema":
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=json_schema
        )
        result = completion.choices[0].message.parsed
    else:
        print("Invalid response type.")
    
    return result


def get_embedding(text):
    text = text.replace("\n", " ")
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    embedding = response.data[0].embedding
    
    return embedding


def pos_prompt(text, nlp=None):
    def get_tool_augmentation(sentence, engine=nlp):
        doc = engine(sentence)
        pos_tags = [f"{word.text}/{word.pos}" for sent in doc.sentences for word in sent.words]
        pos_text = ' '.join(pos_tags)
        
        return pos_text
    
    if not nlp:
        nlp = stanza.Pipeline('en', processors='tokenize,pos')
    pos_text = get_tool_augmentation(text, engine=nlp)
    
    ta_sp_user_prompt = f"""
    Let's infer named entities step by step from the text based on the given Part-of-Speech tags.
    Part-of-Speech tags: {pos_text}
    """
        
    return ta_sp_user_prompt


def negative_sampling_prompt(entity, recognized_values, entity_vocab):
    ns_user_prompt = '\n'.join([f"'{value}' is a '{entity}'." for value in recognized_values])
    
    values = list(set(entity_vocab[entity]) - set(recognized_values))
    if values:
        _value = random.choice(values)
        ns_user_prompt += f"\n'{_value}' is a '{entity}'."
    
    key_ = random.choice(list(set(entity_vocab.keys()) - {entity}))
    value_ = random.choice(entity_vocab[key_])

    ns_user_prompt += f"\n'{value_}' is not a '{entity}'."
    
    return ns_user_prompt


def annotate(text, entity_metadata, entity_vocab, 
             num_examples=3, few_shot_strategy="knn",
             pos_tooling=False, negative_sampling=False):
    prompt_ = ""
    
    if few_shot_strategy == "knn":
        def get_top_k_examples(query, k=3):
            query_vec = get_embedding(query)
            
            distances, indices = knn.kneighbors(np.array([query_vec]), n_neighbors=k)
            top_k_examples = [examples[idx] for idx in indices.flatten()]
            
            return top_k_examples
        
        # get embeddings
        examples_bank = entity_metadata # TODO: load from manually annotated data
        
        # enumerate data
        examples = []
        embeddings = []

        for entity, data in examples_bank.items():
            for example in data['examples']:
                examples.append(example)
                embeddings.append(example['embedding'])

        X = np.array(embeddings)
        
        # Fit the k-NN model
        knn = NearestNeighbors(n_neighbors=num_examples, metric='cosine')
        knn.fit(X)

        few_shot_demonstrations = get_top_k_examples(text, k=num_examples)
    else: # few_shot_strategy == "random"
        few_shot_demonstrations = []
        
        for _ in range(num_examples):
            entity = random.choice(list(entity_metadata.keys()))
            example = random.choice(entity_metadata[entity]["examples"])
            
            few_shot_demonstrations.append(example)
    
    # Preprocess few-shot demonstrations
    for example in few_shot_demonstrations:
        example_text = example['text']
        
        example_pos_prompt = ""
        if pos_tooling:
            example_pos_prompt = pos_prompt(example_text, nlp)
        
        ns_prompt = ""
        if negative_sampling:
            recognized_values = example['entities'][entity] if entity in example['entities'] else []
            if not isinstance(recognized_values, list):
                recognized_values = [recognized_values]
            ns_prompt = negative_sampling_prompt(entity, recognized_values, entity_vocab)
        
        answer = json.dumps({
            "entities": dict.fromkeys(entity_vocab.keys(), [])
        }, indent=2)
        
        prompt_ += f"""
        Input text: {example_text}
        {example_pos_prompt * pos_tooling}
        {ns_prompt * negative_sampling}
        {answer}
        """
    
    system_prompt = """
    You are good at annotating data for Named Entity Recognition. You are given a text and a list of named entities. Think step-by-step and annotate the named entities in the text.
    Given entity label list: {entity_set}.
    """.format(entity_set=list(entity_vocab.keys()))
    
    format_prompt = """
    Return response in the following JSON format:
    {{
        "entities": {{
        """ + \
    ',\n'.join([f"\"{entity}\": []" for entity in entity_vocab.keys()]) + \
        """
        }}
    }}
    """
    
    pos_prompt_ = ""
    if pos_tooling:
        pos_prompt_ = pos_prompt(text, nlp)
    few_shot_prompt = prompt_
    
    prompt = f"""
    {system_prompt}
    {format_prompt}
    
    {few_shot_prompt}
    Input text: {text}
    {pos_prompt_}
    """
    
    prompt = re.sub(r"\n\s*\n", "\n", prompt)
    result = run_query(system_prompt, prompt, response_type="json_object") #"json_schema", json_schema=ResponseSchema)
    
    return result


def main(args):
    def get_entity_vocab(entity_metadata):
        entity_vocab = {}
        for entity in entity_metadata:
            entity_vocab[entity] = entity_metadata[entity]["items"]
        
        return entity_vocab
    
    with open(args.metadata_path, "r") as f:
        entity_metadata = json.load(f)
    
    entity_vocab = get_entity_vocab(entity_metadata)
    
    results = []
    
    for filename in tqdm(os.listdir(args.input_folder), "Inference"):
        with open(os.path.join(args.input_folder, filename), "r") as f:
            text = f.read()
        result = annotate(text, entity_metadata, entity_vocab, 
                          num_examples=args.num_examples, few_shot_strategy=args.few_shot_strategy,
                          pos_tooling=args.pos_tooling, negative_sampling=args.negative_sampling)
        result["id"] = filename.split(".")[0]
        results.append(result)
            
        time.sleep(SLEEP_TIME)
    
    count = 0
    filename_ = args.output.split('/')[-1].split('.')[0]
    for filename in os.listdir(args.output_folder):
        if re.search(rf"{filename_}_\d+", filename):
            count += 1
    args.output = args.output.replace(".jsonl", f"_{count+1}.jsonl")
    
    with open(args.output, "w") as f:
        for result_ in results:
            f.write(json.dumps(result_) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate named entities in a given text.")
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, default="./data/labels_automatic")
    parser.add_argument("--metadata-path", type=str, default="./data/entity_metadata.json")
    parser.add_argument("--example-bank-path", type=str, default="./data/example_bank.json")
    
    parser.add_argument("-n", "--num-examples", type=int, default=3)
    parser.add_argument("-fss", "--few-shot-strategy", type=str, default="knn")
    parser.add_argument("-scn", "--self-consistency-num", type=int, default=1)
    
    parser.add_argument("-pt", "--pos-tooling", action="store_true")
    parser.add_argument("-ns", "--negative-sampling", action="store_true")
    
    args = parser.parse_args()
    
    exp_name = f"{args.few_shot_strategy}_n{args.num_examples}_scn{args.self_consistency_num}{'_pt'*args.pos_tooling}{'_ns'*args.negative_sampling}.jsonl"
    args.output = os.path.join(args.output_folder, exp_name)
    
    if not os.path.exists(os.path.dirname(args.output_folder)):
        os.makedirs(os.path.dirname(args.output_folder))
    
    main(args)