from pymongo import MongoClient
import json


class db:

    def init(self):
        # Read db connection string
        f = open("mongodb_credentials.txt", "r", encoding='utf-8')
        self.cred = json.load(f)

        # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        self.client = MongoClient(host=self.cred['remote_host'],
                                  port=int(self.cred['remote_port']),
                                  username=self.cred['remote_user'],
                                  password=self.cred['remote_password'],
                                  authSource=self.cred['remote_database'])

    def get_database(self):
        return self.client[self.cred['remote_database']]

    def insert_rogue_score(self, score, model_name, inst, target, pred):
        entry = {
            'model': model_name,
            'rouge1_precision': float(json.dumps(score['rouge1_precision'].numpy().tolist())),
            'rouge1_fmeasure': float(json.dumps(score['rouge1_fmeasure'].numpy().tolist())),
            'rouge2_precision': float(json.dumps(score['rouge2_precision'].numpy().tolist())),
            'rouge2_fmeasure': float(json.dumps(score['rouge2_fmeasure'].numpy().tolist())),
            'rougeL_precision': float(json.dumps(score['rougeL_precision'].numpy().tolist())),
            'rougeL_fmeasure': float(json.dumps(score['rougeL_fmeasure'].numpy().tolist())),
            'rougeLsum_precision': float(json.dumps(score['rougeLsum_precision'].numpy().tolist())),
            'rougeLsum_fmeasure': float(json.dumps(score['rougeLsum_fmeasure'].numpy().tolist())),
            'instruction': inst,
            'target': target,
            'prediction': pred
        }
        self.get_database()['rouge_scores'].insert_one(entry)

    def get_base_datasets(self, model, amount):
        return list(self.get_database()['test_datasets_' + model].aggregate([
            {"$sort": {'_id': 1}},  # Make sure we always get the same results!
            {"$match": {"p_model": {"$ne": "pegasus_paraphrase"}}},
            {"$limit": amount}
            ]))

    def get_chatting_datasets(self, amount, include_paraphrased=True):
        # again, make sure to sort so we get the same results again as before.
        if(include_paraphrased):
            return list(self.get_database()['test_datasets_chatting']
                        .find().sort("_id", 1).limit(amount))
        return list(self.get_database()['test_datasets_chatting']
                    .find({"p_model": {"$ne": "pegasus_paraphrase"}}).sort("_id", 1).limit(amount))
