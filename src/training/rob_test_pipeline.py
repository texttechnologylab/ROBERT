from db import db
from torchmetrics.text.rouge import ROUGEScore
import sys
import numpy as np
import json
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/robert'))
from src.robert.robert_lora import robert


db = db()
# Add here all models you want. Set the "test" property to false if
# you dont want that model to be tested in the next run.
test_models = [
    {
        'name': 'robert_1k',
        'desc': 'Trained on 1k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_5k',
        'desc': 'Trained on 5k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_5k_chat_only',
        'desc': 'Trained on 5k chatting chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_21k_chat_only_para',
        'desc': 'Trained on 5k chatting chatgpt ds + 16k chatting paraphrased',
        'test': True
    },
    {
        'name': 'robert_6k_para_chat',
        'desc': 'Trained on 6k base chatgpt ds + 12k paraphrased + 5k chatting ds',
        'test': True
    },
    {
        'name': 'robert_10k_gpt4all',
        'desc': 'Trained on 10k base gpt4all ds',
        'test': True
    },
    {
        'name': 'robert_10k',
        'desc': 'Trained on 10k base chatgpt ds',
        'test': True
    },
    {
        'name': 'robert_45k_chat_para',
        'desc': 'Trained on 12k base chatgpt ds + 12k paraphrased + 5 chatting ds + 16k paraphrased',
        'test': True
    }
]
base_datasets_count = 1000


def test_instruction_following_capabilities(model_name):
    '''Test a model for its instruction following capabilities'''
    # First step: calculate a rogue score. Use chatgpt datasets for that.
    my_robert = robert(finetuned_path=build_finetuned_path(model_name))

    db_name = "chatgpt"
    if("gpt4all" in model_name):
        db_name = "gpt4all"

    base_datasets = db.get_base_datasets("chatgpt", base_datasets_count)
    print("Going through " + str(base_datasets_count) + " datasets.")
    count = 1
    rouge = ROUGEScore()
    # In these tests, we dont want context or anything. Just instruction
    # following capabilities
    for data in base_datasets:
        target = data['output']
        prediction = my_robert.get_response(data['instruction'], use_context=False)
        progress = "Done with " + str(100/base_datasets_count*count) + "%"
        score = rouge(prediction, target)
        entry = {
            'rouge1_precision': json.dumps(score['rouge1_precision'].numpy().tolist()),
            'rouge1_fmeasure': json.dumps(score['rouge1_fmeasure'].numpy().tolist()),
            'rouge2_precision': json.dumps(score['rouge2_precision'].numpy().tolist()),
            'rouge2_fmeasure': json.dumps(score['rouge2_fmeasure'].numpy().tolist()),
            'rougeL_precision': json.dumps(score['rougeL_precision'].numpy().tolist()),
            'rougeL_fmeasure': json.dumps(score['rougeL_fmeasure'].numpy().tolist()),
            'rougeLsum_precision': json.dumps(score['rougeLsum_precision'].numpy().tolist()),
            'rougeLsum_fmeasure': json.dumps(score['rougeLsum_fmeasure'].numpy().tolist()),
            'instruction': data['instruction'],
            'target': target,
            'prediction': prediction
        }
        db.get_database()['rouge_scores'].insert_one(entry)
        count = count + 1
        sys.stdout.write('\r')
        sys.stdout.write('ROUGE on ' + str(base_datasets_count) + ' datasets. ' + progress)
        sys.stdout.flush()


def start_test_pipeline():
    '''
    Starts a test pipeline by testing the given robert models with
    various prompts, dialogs and questions.
    '''
    # We go through each model and test them
    for model in test_models:
        if(model['test'] is False):
            continue

        test_instruction_following_capabilities(model['name'])


if __name__ == "__main__":
    db.init()
    print("Database initiated.")

    start_test_pipeline()
    #preds = "My name is John"
    #target = "Is your name John"
    #rouge = ROUGEScore()
    #print(rouge(preds, target))
