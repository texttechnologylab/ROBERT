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
    }
]
base_datasets_count = 1000


def start_test_pipeline(model_name):
    '''
    Starts a test pipeline by testing the given robert models with
    various prompts, dialogs and questions.
    '''

    # First step: calculate a rogue score. Use chatgpt datasets for that.
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


if __name__ == "__main__":
    db.init()
    print("Database initiated.")
    my_robert = robert()

    start_test_pipeline("robert_10k")
    #preds = "My name is John"
    #target = "Is your name John"
    #rouge = ROUGEScore()
    #print(rouge(preds, target))
