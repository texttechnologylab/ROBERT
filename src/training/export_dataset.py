from db import db
import json

db = db()


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
        with open('data.json', 'w') as f:
            json.dump(datasets, f)
        print("Done exporting.")
    except Exception as e:
        print(str(e))