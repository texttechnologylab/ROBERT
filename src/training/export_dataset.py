from db import db
import json

db = db()


def get_alpaca_datasets():
    datasets = []
    '''used to read in the 52k datasets used for alpaca'''
    with open('data_52k_alpaca.json', 'r') as f:
        datas = json.load(f)
        for data in datas:
            datasets.append(data)
    return datasets


if __name__ == "__main__":
    try:
        print("Initing db...")
        db.init()
        print("Done!\nExporting the datasets...")
        datasets = []
        for data in db.get_database()['test_datasets'].find():
            datasets.append({
                'instruction': data['instruction'],
                'input': data['input'],
                'output': data['output'],
            })
        with open('data_62k_alpaca_robert.json', 'w') as f:
            json.dump(datasets, f)
        print("Done exporting.")
    except Exception as e:
        print(str(e))