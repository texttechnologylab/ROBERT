from db import db
import json
import sys
from bson.objectid import ObjectId

db = db()
include_base = False
include_paraphrased = False
include_dialogs = False
include_student = True
model_name = "chatgpt"


def get_alpaca_datasets():
    datasets = []
    '''used to read in the 52k datasets used for alpaca'''
    with open('data_52k_alpaca.json', 'r') as f:
        datas = json.load(f)
        for data in datas:
            datasets.append(data)
    return datasets


def get_paraphrased_from(model, limit):
    res = list(db.get_database()['test_datasets_' + model].find().limit(limit))
    return res


if __name__ == "__main__":
    try:
        print("Initing db...")
        db.init()
        print("Done!\nExporting the datasets...")
        datasets = []
        total_base = 10000

        if(include_student):
            fetched = list(db.get_database()['test_datasets_chatgpt'].find())
            #fetched.extend(list(db.get_database()['test_datasets_gpt4all'].find()))
            print("Exporting " + str(len(fetched)) + " datasets")
            for data in fetched:
                datasets.append({
                    'instruction': "Formulate an instruction or a question towards Rob about the given input",
                    'input': "\n".join(data['context'].split('[ITEM]')),
                    'output': data['instruction'],
                })
            print("Student done")

        if(include_base):
            for data in db.get_base_datasets(model_name, total_base):
                datasets.append({
                    'instruction': data['instruction'],
                    'input': data['input'],
                    'output': data['output'],
                })
            print("Base datasets done")

        # Get paraphrased from them if we want
        if(include_paraphrased):
            print("Doing paraphrased now...")
            for para in get_paraphrased_from(model_name, total_base * 6):
                datasets.append({
                    'instruction': para['instruction'],
                    'input': para['input'],
                    'output': para['output'],
                })
            print("Done!")

        # Include dialogs if we want them
        if(include_dialogs):
            print("Adding the dialog datasets...")
            for data in db.get_chatting_datasets(30000, include_paraphrased):
                datasets.append({
                    'instruction': data['instruction'],
                    'input': data['input'],
                    'output': data['output'],
                })
            print("Done!")
        with open('datasets/data_Xk_para_student.json', 'w') as f:
            json.dump(datasets, f)
        print("Done exporting.")
    except Exception as e:
        print(str(e))
