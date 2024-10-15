import pymongo
import os
from dotenv import load_dotenv

load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_URI"))

db = client[os.getenv("MONGODB_DB")]
col = db[os.getenv("MONGODB_COLLECTION")]

class Connection():
    def __new__(cls, database):
        connection = client
        return connection[database]

# x = col.find_one()

# print(x)