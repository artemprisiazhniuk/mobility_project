import os
import json
import random
import time
import argparse

from pydantic import BaseModel
from openai import OpenAI
import stanza


# stanza.download('en')

client = OpenAI()
nlp = stanza.Pipeline('en', processors='tokenize,pos')

class ResponseSchema(BaseModel):
    entities: dict[str, list[str]]
    
SLEEP_TIME = 1


def run_query(system_prompt, user_prompt, response_type="json_object", json_schema=None):
    format_ = {"type": response_type} if response_type == "json_object" else {"type": response_type, "json_schema": json_schema}
    
    if response_type == "json_object":
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=format_ 
        )
        result = completion.choices[0].message.content
        
        if response_type == "json_object":
            result = json.loads(result)    
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


def annotate(text, entity_metadata, entity_vocab):
    prompt_ = ""
    num_examples = 3
    for _ in range(num_examples):
        entity = random.choice(list(entity_metadata.keys()))
        example = random.choice(entity_metadata[entity]["examples"])
        
        example_text = example['sentence']
        example_pos_prompt = pos_prompt(example_text, nlp)
        recognized_values = example['entities'][entity]
        
        if not isinstance(recognized_values, list):
            recognized_values = [recognized_values]
        ns_prompt = negative_sampling_prompt(entity, recognized_values, entity_vocab)
        
        answer = json.dumps({
            "entities": dict.fromkeys(entity_vocab.keys(), [])
        }, indent=2)
        
        prompt_ += f"""
        Input text: {example_text}
        {example_pos_prompt}
        {ns_prompt}
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
    
    pos_prompt_ = pos_prompt(text, nlp)
    few_shot_prompt = prompt_
    
    prompt = f"""
    {system_prompt}
    {format_prompt}
    
    {few_shot_prompt}
    Input text: {text}
    {pos_prompt_}
    """
    
    result = run_query(system_prompt, prompt, response_type="json_object")#"json_schema", json_schema=ResponseSchema)
    
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
    
    for filename in os.listdir(args.input_folder):
        with open(os.path.join(args.input_folder, filename), "r") as f:
            text = f.read()
        result = annotate(text, entity_metadata, entity_vocab)
        
        with open(os.path.join(args.output_folder, filename.replace(".txt", ".json")), "w") as f:
            json.dump(result, f, indent=2)
            
        time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate named entities in a given text.")
    parser.add_argument("--input-folder", type=str)
    parser.add_argument("--output-folder", type=str)
    parser.add_argument("--metadata-path", type=str, default="data/entity_metadata.json")
    args = parser.parse_args()
    
    main(args)