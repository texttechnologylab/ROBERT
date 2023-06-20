from db import db
from torchmetrics.text.rouge import ROUGEScore
from nltk.translate.bleu_score import sentence_bleu
import torch
import time
import sys
import numpy as np
import json
import os
import gc
from datetime import datetime

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
        'test': False
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
base_datasets_count = 2
chat_datasets_count = 2
tries = 3
done_models = []


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
        bleu_score = float(sentence_bleu([target], prediction))
        db.insert_rogue_score(score, model_name, data['instruction'],
                              target, prediction, data['input'], bleu_score)

        count = count + 1
        #sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(count) + ' datasets. ' + progress)
        #sys.stdout.flush()


def test_dialog_capabilities(model_name, my_robert):
    print("\n")
    print("----- Testing dialog capabilities of " + model_name)

    chat_datasets = db.get_chatting_datasets_with_input(chat_datasets_count, False)
    print("Going through " + str(chat_datasets_count) + " datasets.")
    count = 1
    rouge = ROUGEScore()
    for data in chat_datasets:
        # For here, we want to work with the input as context.
        target = data['output']
        my_robert.set_context(data['input'].split('\n'))
        prediction = my_robert.get_response(data['instruction'])
        progress = "Done with " + str(round(100/chat_datasets_count*count, 1)) + "%"
        score = rouge(prediction, target)
        bleu_score = float(sentence_bleu([target], prediction))
        db.insert_rogue_score(score, model_name, data['instruction'],
                              target, prediction, data['input'], bleu_score)

        count = count + 1
        # Decomment the two lines below if you dont want a new line in the console.
        #sys.stdout.write('\r')
        sys.stdout.write('Done with ' + str(count) + ' datasets. ' + progress)
        #sys.stdout.flush()


def start_test_pipeline():
    '''
    Starts a test pipeline by testing the given robert models with
    various prompts, dialogs and questions.
    '''
    # We go through each model and test them
    to_test = [m for m in test_models if m['test'] is True]
    print(str(datetime.now()))
    print("===================== Starting a new pipeline =====================")
    print("For that, we have " + str(len(to_test)) + " models to test.\n\n")
    for model in to_test:
        try:
            if(model['name'] in done_models):
                continue
            my_robert = robert(finetuned_path=build_finetuned_path(model['name']))
            print("Doing " + model['name'] + " now:")

            test_instruction_following_capabilities(model['name'], my_robert)
            test_dialog_capabilities(model['name'], my_robert)

            print("Done with " + model['name'] + "!\n")
            done_models.append(model['name'])
            # Free the gpu from the model
            my_robert = ""
            gc.collect()
            torch.cuda.empty_cache()
            # I hope this gives pytorch enough time to free the memory. Otherwise, we crash here.
            time.sleep(5)
        except Exception as ex:
            print("Caught en exception")
            # We want to try again if an error occured because it could be just
            # missing memory.
            if(tries > 0):
                print("Retrying again in 10 seconds.")
                time.sleep(10)
                tries = tries - 1
                start_test_pipeline()

    print(str(datetime.now()))
    print("===================== Done with the pipeline =====================")


if __name__ == "__main__":
    db.init()
    print("Database initiated.")

    start_test_pipeline()
