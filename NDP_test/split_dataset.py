import os
import sys
import uuid
import time
import multiprocessing as mp
# import numpy as np
# import random

import pyarrow.parquet as pq

from concurrent.futures import ThreadPoolExecutor


def write_file(filename, table):
    open(filename, 'a').close()
    attribute = "ceph.file.layout.object_size"
    os.system(
        f"setfattr -n {attribute} -v 16777216 {filename}")
    pq.write_table(
        table, filename,
        row_group_size=table.num_rows, compression="snappy"
    )


class SplittedParquetWriter(object):
    def __init__(self, filename, destination, chunksize=128*1024*1024):
        self.filename = filename
        self.destination = destination
        self.chunksize = chunksize

    def round(self, num):
        num_str = str(int(num))
        result_str = ""
        result_str += num_str[0]
        for i in range(len(num_str) - 1):
            result_str += "0"
        return int(result_str)

    def write_file(self, filename, table):
        open(filename, 'a').close()
        attribute = "ceph.file.layout.object_size"
        os.system(
            f"setfattr -n {attribute} -v 16777216 {filename}")
        pq.write_table(
            table, filename,
            row_group_size=table.num_rows, compression="snappy"
        )

    def estimate_rows(self):
        self.table = pq.read_table(self.filename)
        disk_size = os.stat(self.filename).st_size
        inmemory_table_size = self.table.nbytes
        inmemory_row_size = inmemory_table_size/self.table.num_rows
        compression_ratio = inmemory_table_size/disk_size
        required_inmemory_table_size = self.chunksize * compression_ratio
        required_rows_per_file = required_inmemory_table_size/inmemory_row_size
        return self.table.num_rows, self.round(required_rows_per_file)

    def write(self):
        os.makedirs(self.destination, exist_ok=True)
        s_time = time.time()
        total_rows, rows_per_file = self.estimate_rows()
        i = 0
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            while i < total_rows:
                executor.submit(
                    self.write_file,
                    os.path.join(
                        self.destination, f"{uuid.uuid4().hex}.parquet"),
                    self.table.slice(i, rows_per_file)
                )
                i += rows_per_file
        e_time = time.time()
        print(f"Finished writing in {e_time - s_time} seconds")

if __name__ == "__main__":
    chunksize = 13 * 1024 * 1024 # 13MB
    dataset_dir = str(sys.argv[1])
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                file_size = os.stat(file_path).st_size/(1024*1024)
                print(f"{file_path} = {file_size} MB")

                if file_size > 16:
                    print(f"{os.path.join(root, file)} is larger than 16 MB, splitting")
                    writer = SplittedParquetWriter(file_path, os.path.dirname(file_path), chunksize)
                    writer.write()
                    print(f"{os.path.join(root, file)} is split")
                    os.remove(file_path)
                else:
                    table = pq.read_table(file_path)
                    os.remove(file_path)
                    file_path = os.path.join(os.path.dirname(file_path), f"{uuid.uuid4().hex}.parquet")
                    write_file(file_path, table)
