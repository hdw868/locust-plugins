from pymongo import MongoClient
from datetime import datetime
import logging
import time
import os
from contextlib import contextmanager


class MongoReader:
    def __init__(self, filters, id_column, uri, database, collection):
        self.coll = MongoClient(uri)[database][collection]
        self.id_column = id_column
        self.delay_warning = 0.5
        self.query = {"$and": filters + [{"logged_in": 0}]}

    @contextmanager
    def user(self):
        start_at = time.time()
        user = self.coll.find_one_and_update(
            self.query, {"$set": {"last_login": datetime.now(), "logged_in": 1}}, sort=[("last_login", 1)]
        )
        if user is None:
            raise Exception(f"Didnt get any user from db ({self.coll}) using query {self.query}")
        if start_at + self.delay_warning < time.time():
            logging.warning(
                f"Getting a user took more than {self.delay_warning} seconds (doubling warning threshold for next time)"
            )
            self.delay_warning *= 2
        try:
            yield user
        finally:
            releasessn = self.coll.find_one_and_update(
                {"$and": [{self.id_column: user[self.id_column]}, {"logged_in": 1}]}, {"$set": {"logged_in": 0}}
            )
        if releasessn is None:
            raise Exception(f"Couldnt release lock for user in db. ")
