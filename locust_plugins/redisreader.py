import os.path
import logging

import redis
import csv
from typing import Optional, List, Dict, Any, Iterator

from locust.env import Environment
from locust.runners import WorkerRunner


def get_base_name_from_path(filepath) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]


class RedisReader:
    """Thread-safe Redis reader for distributed testing environments"""

    def __init__(
            self,
            environment: Environment,
            uri: str,
            csv_file: str,
            recycle_on_eof: bool = True,
    ):
        """
        Initialize Redis reader

        Args:
            environment: Locust Environment object
            uri: Redis uri, in format of redis://[[username]:[password]]@localhost:6379/0
            csv_file: CSV file to uploaded to Redis
            recycle_on_eof: Whether to start over when reaching end of data, only work when stop_on_eof is set to False
        """
        self.environment = environment
        self.runner = self.environment.runner
        self.redis_client = redis.Redis.from_url(uri, decode_responses=True)
        self.csv_file = csv_file
        self.key_prefix = f"fastperf_datasets_{get_base_name_from_path(csv_file)}"
        self.recycle_on_eof = recycle_on_eof
        self._data_key = f"{self.key_prefix}:data"
        self._index_key = f"{self.key_prefix}:index"
        self._lock_key = f"{self.key_prefix}:lock"
        self._counter_key = f"{self.key_prefix}:counter"

        self._on_init(environment)

    def _on_init(self, environment, **kwargs):
        if not isinstance(environment.runner, WorkerRunner):
            try:
                logging.info(f"Master node uploading records...will take some time")
                count = self.upload_csv()
                logging.info(
                    f"Master node uploaded {count} records to Redis from {self.csv_file}"
                )
            except Exception as e:
                logging.error(f"Failed to initialize data in master node: {str(e)}")
                raise

    def upload_csv(self, chunk_size: int = 1000) -> int:
        """
        Upload CSV data to Redis

        Args:
            chunk_size: Number of records to process at once

        Returns:
            Number of records uploaded
        """
        with open(self.csv_file, "r") as f:
            reader = csv.DictReader(f)
            records = []
            counter = 0
            pipe = self.redis_client.pipeline()

            # Reset counter and clear existing data
            pipe.delete(self._counter_key, self._data_key, self._index_key)
            pipe.set(self._counter_key, 0)
            pipe.execute()

            for record in reader:
                records.append(record)
                counter += 1

                if len(records) >= chunk_size:
                    self._upload_chunk(pipe, records, counter - len(records))
                    records = []
                    pipe.execute()
                    pipe = self.redis_client.pipeline()

            if records:
                self._upload_chunk(pipe, records, counter - len(records))
                pipe.execute()

            return counter

    def _upload_chunk(self, pipe, records: List[dict], start_index: int) -> None:
        """Upload a chunk of records to Redis"""
        for i, record in enumerate(records):
            index = start_index + i
            record_key = f"{self._data_key}:{index}"
            pipe.hset(record_key, mapping=record)
            pipe.sadd(self._index_key, index)
        pipe.incrby(self._counter_key, len(records))

    def reset(self):
        total = self.get_total_records()
        self.redis_client.sadd(self._index_key, *range(total))

    def __next__(self) -> Optional[Dict[str, Any]]:
        """
        Get next available record

        Args:

        Returns:
            Record as dictionary or None if no records available
        """
        try:
            index = self.redis_client.spop(self._index_key)
            if index is None:
                # raise
                if not self.recycle_on_eof:
                    raise StopIteration

                # Reload all indices
                self.reset()
                index = self.redis_client.spop(self._index_key)

            record_key = f"{self._data_key}:{index}"
            return self.redis_client.hgetall(record_key)
        except StopIteration:
            logging.info("No more test data available. Stopping the test run.")
            if self.environment.parsed_options and self.environment.parsed_options.headless:
                self.runner.quit()
            else:
                self.runner.stop()
            self.runner.shape_greenlet = None
            self.runner.shape_last_tick = None
        except Exception as e:
            logging.error(f"Error getting data from redis: {str(e)}")
            raise

    def get_total_records(self) -> int:
        """Get total number of records in Redis"""
        return int(self.redis_client.get(self._counter_key) or 0)
