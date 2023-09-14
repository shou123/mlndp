from asyncio import futures
import os
import sys
from numpy import source

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq
import pyarrow.csv as csv

def convert(source_dir, destination_dir, filename):
    # Read the csv file and convert it to a pyarrow table 
    # applying the provided schema
    parse_options = csv.ParseOptions(delimiter='|')
    read_options = csv.ReadOptions(column_names=cols)
    table = csv.read_csv(os.path.join(source_dir, filename), read_options=read_options, parse_options=parse_options)
    print(table.schema)

    pq.write_table(table, os.path.join(destination_dir, f'{filename}.parquet'), compression='snappy')
    print(f'{filename}.parquet file created.')


if __name__ == "__main__":
    source_dir = str(sys.argv[1])
    destination_dir = str(sys.argv[2])

    tables = [
        "lineitem",
        "orders",
        "part",
        "partsupp",
        "supplier",
        "customer",
        "nation",
        "region"
    ]

    for table_name in tables:
        # Read the schema file and parse the column names
        with open(f'schemas/{table_name}', 'r') as f:
            lines = f.readlines()
        cols = list()
        for line in lines:
            key, type = line.split(',')
            key = key.strip()
            type = type.strip()
            cols.append(key)

        source_dir_m = os.path.join(source_dir, table_name)
        destination_dir_m = os.path.join(destination_dir, table_name)
        os.makedirs(os.path.join(destination_dir_m), exist_ok=True)

        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(convert, source_dir_m, destination_dir_m, filename) for filename in os.listdir(source_dir_m)]
            for future in as_completed(futures):
                print(f'{table_name} file converted.')


