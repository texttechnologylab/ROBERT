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
