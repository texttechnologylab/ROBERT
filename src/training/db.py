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

    def insert_chatgpt_score(self, score, model_name, inst, output, inp, con, rouge_score):
        entry = {
          'score': score,
          'instruction': inst,
          'output': output,
          'context': con,
          'input': inp,
          'model': model_name,
          'rogue_score': rouge_score
        }
        self.get_database()['chatgpt_scores'].insert_one(entry)

    def insert_rogue_score(self, score, model_name, inst, target, pred, inp, bleu):
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
            'bleu': bleu,
            'instruction': inst,
            'target': target,
            'prediction': pred,
            'inp': inp
        }
        self.get_database()['rouge_scores'].insert_one(entry)

    def get_student_instructions(self, amount):
        return list(self.get_database())['student_instructions'].find().sort("_id", 1).limt(amount)

    def get_base_datasets(self, model, amount):
        return list(self.get_database()['test_datasets_' + model].aggregate([
            {"$sort": {'_id': 1}},  # Make sure we always get the same results!
            {"$match": {"p_model": {"$ne": "pegasus_paraphrase"}}},
            {"$limit": amount}
            ]))

    def get_rouge_scores(self, amount):
        return self.get_database()['rouge_scores'].find().sort("model", 1).limit(amount)

    def get_chatting_datasets_with_input(self, amount, include_paraphrased=True):
        datasets = self.get_chatting_datasets(99999999, include_paraphrased)
        return [d for d in datasets if d['input'] != ''][:amount]

    def get_chatting_datasets(self, amount, include_paraphrased=True):
        # again, make sure to sort so we get the same results again as before.
        if(include_paraphrased):
            return list(self.get_database()['test_datasets_chatting']
                        .find().sort("_id", 1).limit(amount))
        return list(self.get_database()['test_datasets_chatting']
                    .find({"p_model": {"$ne": "pegasus_paraphrase"}}).sort("_id", 1).limit(amount))
