import os
import sys
import time
import json

import multiprocessing as mp
import numpy as np
import random
import duckdb
import pyarrow.dataset as ds #pyarrow1 is the skyhook version pyarrow, pyarrow is lastest version pyarrow used for lancedb
import lancedb
import lance



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
    lineitem = ds.dataset(os.path.join(dataset_path, "lineitem"), format=format).to_table().to_pandas()
    supplier = ds.dataset(os.path.join(dataset_path, "supplier"), format=format)
    customer = ds.dataset(os.path.join(dataset_path, "customer"), format=format)
    region   = ds.dataset(os.path.join(dataset_path, "region"), format=format)
    nation   = ds.dataset(os.path.join(dataset_path, "nation"), format=format)
    orders   = ds.dataset(os.path.join(dataset_path, "orders"), format=format)
    part     = ds.dataset(os.path.join(dataset_path, "part"), format=format)
    partsupp = ds.dataset(os.path.join(dataset_path, "partsupp"), format=format)

    # db = lancedb.connect("/mnt/cephfs/lancedb")
    # db.create_table("lineitem", lineitem)

    lance.write_dataset(supplier, "/mnt/cephfs/lancedb")

    # query = "SELECT * FROM lineitem LIMIT 20;" 

    # # for command
    # with open(f"NDP_test/queries/q{query_no}.sql", "r") as f:
    #     query = f.read()

    # # query = f"PRAGMA disable_object_cache;\nPRAGMA threads={mp.cpu_count()};\n{query}"
    # for _ in range(2000):
    #     drop_caches()
    #     conn = duckdb.connect()
    #     s = time.time()
    #     result = conn.execute(query).fetchall()
    #     e = time.time()
    #     conn.close()

    #     log_str = f"{query_no}|{format}|{e - s}"
    #     print(log_str)

    #     data.append({
    #         "query": query_no,
    #         "format": format,
    #         "latency": e - s
    #     })
    # print("Benchmark finished")



        # Create LanceDB context, connection, and cursor
    # connection = context.connect()
    # cursor = connection.cursor()

    # Read and execute the query using LanceDB
    with open(f"NDP_test/queries/q{query_no}.sql", "r") as f:
        query = f.read()
        cursor.execute(query)
        results = cursor.fetchall()
        print(results)

    # Close the cursor and connection
    cursor.close()
    connection.close()
