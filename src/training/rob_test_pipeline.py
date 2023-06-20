from db import db
from torchmetrics.text.rouge import ROUGEScore
import sys
import numpy as np
import json
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/robert'))
from src.robert.robert_lora import robert
from src.robert.robert_lora import build_finetuned_path


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
chat_datasets_count = 1000


def test_instruction_following_capabilities(model_name, my_robert):
    '''Test a model for its instruction following capabilities'''
    # First step: calculate a rogue score. Use chatgpt datasets for that.
    print("\n")
    print("----- Testing instruction following capabilities of " + model_name)

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
        progress = "Done with " + str(round(100/base_datasets_count*count, 1)) + "%"
        score = rouge(prediction, target)
        db.insert_rogue_score(score, model_name, data['instruction'], target, prediction)

        count = count + 1
        sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(base_datasets_count) + ' datasets. ' + progress)
        sys.stdout.flush()


def test_dialog_capabilities(model_name, my_robert):
    print("\n")
    print("----- Testing instruction following capabilities of " + model_name)

    chat_datasets = db.get_chatting_datasets(chat_datasets_count, False)
    print("Going through " + str(chat_datasets_count) + " datasets.")
    count = 1
    rouge = ROUGEScore()
    for data in chat_datasets:
        # For here, we want to work with the input as context.
        target = data['output']
        my_robert.set_context(data['input'].split('\n'))
        prediction = my_robert.get_response(data['instruction'])
        score = rouge(prediction, target)
        db.insert_rogue_score(score, model_name, data['instruction'], target, prediction)

        count = count + 1
        sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(base_datasets_count) + ' datasets. ' + progress)
        sys.stdout.flush()


def start_test_pipeline():
    '''
    Starts a test pipeline by testing the given robert models with
    various prompts, dialogs and questions.
    '''
    # We go through each model and test them
    to_test = [m for m in test_models if m['test'] is True]
    print("===================== Starting a new pipeline =====================")
    print("For that, we have " + len(to_test) + " models to test.\n\n")
    for model in to_test:
        my_robert = robert(finetuned_path=build_finetuned_path(model_name))
        print("Doing " + model['name'] + " now:")
        # test_instruction_following_capabilities(model['name'], my_robert)
        test_dialog_capabilities(model['name'], my_robert)

        print("Done with " + model['name'] + "!\n")


if __name__ == "__main__":
    db.init()
    print("Database initiated.")

    start_test_pipeline()
