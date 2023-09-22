import os
import sys
import time
import json

import multiprocessing as mp
import numpy as np
import random
import duckdb
import pyarrow.dataset as ds



def drop_caches():
    os.system("sync")
    os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    os.system("sync")
    time.sleep(2)


if __name__ == "__main__":
    dataset_path = str(sys.argv[1])
    query_no = int(sys.argv[2])
    # format = str(sys.argv[3])
    format = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")

    data = list()
    lineitem = ds.dataset(os.path.join(dataset_path, "lineitem"), format=format)
    supplier = ds.dataset(os.path.join(dataset_path, "supplier"), format=format)
    customer = ds.dataset(os.path.join(dataset_path, "customer"), format=format)
    region   = ds.dataset(os.path.join(dataset_path, "region"), format=format)
    nation   = ds.dataset(os.path.join(dataset_path, "nation"), format=format)
    orders   = ds.dataset(os.path.join(dataset_path, "orders"), format=format)
    part     = ds.dataset(os.path.join(dataset_path, "part"), format=format)
    partsupp = ds.dataset(os.path.join(dataset_path, "partsupp"), format=format)

    # query = "SELECT * FROM lineitem LIMIT 20;" 

    # for command
    with open(f"project/NDP_test/queries/q{query_no}.sql", "r") as f:
        query = f.read()

    # query = f"PRAGMA disable_object_cache;\nPRAGMA threads={mp.cpu_count()};\n{query}"
    for _ in range(200):
        drop_caches()
        conn = duckdb.connect()
        s = time.time()
        result = conn.execute(query).fetchall()
        e = time.time()
        conn.close()

        log_str = f"{query_no}|{format}|{e - s}"
        print(log_str)

        data.append({
            "query": query_no,
            "format": format,
            "latency": e - s
        })
    print("Benchmark finished")
