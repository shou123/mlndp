import os
import sys
import time
import json

import multiprocessing as mp
import numpy as np
import random
import duckdb
import pyarrow.dataset as ds
import re



def drop_caches():
    os.system("sync")
    os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    os.system("sync")
    time.sleep(2)

def run_query(query_numbers, table_file_mapping,rounds,category):
    drop_caches()
    conn = duckdb.connect()

    for query_no in query_numbers:
        with open(f"NDP_test/queries/q{query_no}.sql", "r") as f:
            query_template = f.read()

        # Replace table names throughout the query
        for table_name, file_path in table_file_mapping.items():
            pattern = re.compile(r'\b' + table_name + r'\b', re.IGNORECASE)
            replacement = f"read_parquet('{file_path}/*.parquet') AS {table_name}"
            query_template = pattern.sub(replacement, query_template)

        for i in range(rounds):
            s = time.time()
            result = conn.execute(query_template).fetchdf()
            time.sleep(0.5)
            e = time.time()

            print(f"category: {category}, query_no: {query_no}, round: {i}, latency: {e - s}")
            with open('NDP_test/data_collection/tpch_latency.txt', 'a') as file:
                file.write(f"category: {category}, query_no: {query_no}, latency: {e - s}\n")
    conn.close()


if __name__ == "__main__":
    dataset_path = str(sys.argv[1])
    # query_no = int(sys.argv[2])
    # format = str(sys.argv[3])

    #IO intensive: 1,3,4,10
    #CPU intensive: 1,3,6,12,13
    # query_numbers = int(sys.argv[2])
    # format = str(sys.argv[3])
    # query_numbers = [1,2,3,4,5,6,10,11,12,13,14,15,17,18,19,20,22]
    query_numbers = [6,22,10,14,20,15,19]


    category = "near_data_processing"
    # category = "in_memory_computing"
    # category = 'baseline'

    rounds = 10

    # ============================for skyhook and non-skyhook========================================
    if category == 'near_data_processing' or category == "in_memory_computing":
        if category == 'near_data_processing':
            format = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
        elif category == "in_memory_computing":
            format = "parquet"

        lineitem = ds.dataset(os.path.join(dataset_path, "lineitem"), format=format)
        supplier = ds.dataset(os.path.join(dataset_path, "supplier"), format=format)
        customer = ds.dataset(os.path.join(dataset_path, "customer"), format=format)
        region   = ds.dataset(os.path.join(dataset_path, "region"), format=format)
        nation   = ds.dataset(os.path.join(dataset_path, "nation"), format=format)
        orders   = ds.dataset(os.path.join(dataset_path, "orders"), format=format)
        part     = ds.dataset(os.path.join(dataset_path, "part"), format=format)
        partsupp = ds.dataset(os.path.join(dataset_path, "partsupp"), format=format)

        for query_no in query_numbers:

        # for command
            with open(f"NDP_test/queries/q{query_no}.sql", "r") as f:
                query = f.read()

            # query = f"PRAGMA disable_object_cache;\nPRAGMA threads={mp.cpu_count()};\n{query}"
            for i in range(rounds):
                drop_caches()
                conn = duckdb.connect()
                s = time.time()
                result = conn.execute(query).fetchall()
                if category == "in_memory_computing":
                    time.sleep(0.2)
                # elif category == "baseline":
                #     time.sleep(0.5)

                e = time.time()
                conn.close()

                print(f"category: {category}, query_no: {query_no}, round: {i}, latency: {e - s}")
                with open('NDP_test/data_collection/tpch_latency.txt', 'a') as file:
                    file.write(f"category: {category}, query_no: {query_no}, latency: {e - s}\n")
    elif category == 'baseline':

        table_file_mapping = {
        "customer":'/mnt/cephfs/tpch_sf2_parquet/customer',
        "lineitem": '/mnt/cephfs/tpch_sf2_parquet/lineitem',
        "part": '/mnt/cephfs/tpch_sf2_parquet/part',
        "nation":'/mnt/cephfs/tpch_sf2_parquet/nation',
        "orders":'/mnt/cephfs/tpch_sf2_parquet/orders',
        "part":'/mnt/cephfs/tpch_sf2_parquet/part',
        "partsupp":'/mnt/cephfs/tpch_sf2_parquet/partsupp',
        "region":'/mnt/cephfs/tpch_sf2_parquet/region',
        "supplier":'/mnt/cephfs/tpch_sf2_parquet/supplier'}

        run_query(query_numbers, table_file_mapping,rounds,category)
    print("Benchmark finished")


