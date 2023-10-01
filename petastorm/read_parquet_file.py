from pyspark.sql import SparkSession
from petastorm.etl.dataset_metadata import materialize_dataset
from petastorm.unischema import dict_to_spark_row
import numpy as np
from petastorm.unischema import Unischema, UnischemaField
from pyspark.sql.types import IntegerType,StringType,DoubleType,DateType
from petastorm.codecs import ScalarCodec

import pyarrow.dataset as ds
import os
import multiprocessing as mp
import duckdb
import time
import pyarrow.parquet as pq

def spark_wrap():
    spark = SparkSession.builder.appName("ParquetToDataset").getOrCreate()
    print(f"spark: {spark}")
    customer_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/customer")
    lineitem_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/lineitem")
    nation_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/nation")
    orders_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/orders")
    part_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/part")
    partsupp_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/partsupp")
    region_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/region")
    supplier_parquet_data = spark.read.parquet("/mnt/cephfs/tpch_sf2_parquet/supplier")
    return customer_parquet_data,lineitem_parquet_data,nation_parquet_data,orders_parquet_data,part_parquet_data,partsupp_parquet_data,region_parquet_data,supplier_parquet_data

def scheme_assemble():
    customer_schema = Unischema('CustomerSchema', [
    UnischemaField('c_custkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('c_name', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('c_address', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('c_nationkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('c_phone', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('c_acctbal', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('c_mktsegment', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('c_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    lineitem_schema = Unischema('YourSchema', [
    UnischemaField('l_orderkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('l_partkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('l_suppkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('l_linenumber', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('l_quantity', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('l_extendedprice', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('l_discount', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('l_tax', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('l_returnflag', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('l_linestatus', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('l_shipdate', np.int32, (), ScalarCodec(DateType()), False),
    UnischemaField('l_commitdate', np.int32, (), ScalarCodec(DateType()), False),
    UnischemaField('l_receiptdate', np.int32, (), ScalarCodec(DateType()), False),
    UnischemaField('l_shipinstruct', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('l_shipmode', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('l_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    nation_schema = Unischema('YourSchema', [
    UnischemaField('n_nationkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('n_name', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('n_regionkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('n_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    orders_schema = Unischema('YourSchema', [
    UnischemaField('o_orderkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('o_custkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('o_orderstatus', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('o_totalprice', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('o_orderdate', np.int32, (), ScalarCodec(DateType()), False),
    UnischemaField('o_orderpriority', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('o_clerk', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('o_shippriority', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('o_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    part_schema = Unischema('YourSchema', [
    UnischemaField('p_partkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('p_name', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('p_mfgr', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('p_brand', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('p_type', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('p_size', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('p_container', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('p_retailprice', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('p_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    partsupp_schema = Unischema('YourSchema', [
    UnischemaField('ps_partkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('ps_suppkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('ps_availqty', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('ps_supplycost', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('ps_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    region_schema = Unischema('YourSchema', [
    UnischemaField('r_regionkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('r_name', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('r_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])

    supplier_schema = Unischema('YourSchema', [
    UnischemaField('s_suppkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('s_name', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('s_address', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('s_nationkey', np.int32, (), ScalarCodec(IntegerType()), False),
    UnischemaField('s_phone', np.str_, (), ScalarCodec(StringType()), False),
    UnischemaField('s_acctbal', np.float64, (), ScalarCodec(DoubleType()), False),
    UnischemaField('s_comment', np.str_, (), ScalarCodec(StringType()), False)
    ])



    return customer_schema, lineitem_schema, nation_schema, orders_schema, part_schema, partsupp_schema, region_schema, supplier_schema

def drop_caches():
    os.system("sync")
    os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    os.system("sync")
    time.sleep(2)

def petastorm_dataset():
    customer_schema, lineitem_schema, nation_schema, orders_schema, part_schema, partsupp_schema, region_schema, supplier_schema = scheme_assemble()
    schemas = [customer_schema, lineitem_schema, nation_schema, orders_schema, part_schema, partsupp_schema, region_schema, supplier_schema]

    spark = SparkSession.builder.appName("ParquetToDataset").getOrCreate()
    table_names = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']
    # tables = [customer_df,lineitem_df, nation_df, orders_df, part_df, partsupp_df, region_df, supplier_df]
    # print(region_df.show())
    table_schema_mapping = dict(zip(table_names, schemas))

    for table_name  in table_names:
        parquet_dataset_path = f"file:///mnt/cephfs/tpch_sf2_parquet/{table_name}"
        petastorm_dataset_path = f"file:///mnt/cephfs/petastorm_dataset/{table_name}"

        df = spark.read.parquet(parquet_dataset_path)

        schema = table_schema_mapping.get(table_name)

        rowgroup_size_mb = 16

        # Set the desired block size (in bytes)
        desired_block_size_bytes = 10 * 1024 * 1024  # 100MB

        # Calculate the number of partitions needed to achieve the desired block size
        num_partitions = max(1,int(df.rdd.getNumPartitions() * (df.rdd.map(lambda x: len(str(x))).sum() / desired_block_size_bytes)))
        data = df.repartition(num_partitions)

        with materialize_dataset(spark, petastorm_dataset_path, schema, rowgroup_size_mb):

        # spark.createDataFrame(rows_rdd, schema.as_spark_schema()) \
            # df.write.option("compression", "uncompressed").mode('overwrite').parquet(petastorm_dataset_path)
            data.write.option("parquet.block.size", str(desired_block_size_bytes)).mode('overwrite').parquet(petastorm_dataset_path)
        print(table_name)
        print(df.show())

    spark.stop()


if __name__ == '__main__':
    # petastorm_dataset()
    drop_caches()

    query_no = 22
    format = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
    

    dataset_path = f"file:///mnt/cephfs/petastorm_dataset"
    data = list()
    lineitem = ds.dataset(os.path.join(dataset_path, "lineitem"), format=format)

    supplier = ds.dataset(os.path.join(dataset_path, "supplier"), format=format)
    customer = ds.dataset(os.path.join(dataset_path, "customer"), format=format)
    region = ds.dataset(os.path.join(dataset_path, "region"), format=format)
    nation   = ds.dataset(os.path.join(dataset_path, "nation"), format=format)
    # print(f"nation:  {nation.schema}")
    # print(nation.files)
    # print(f"nation arrow dataset: {nation.to_table(filter=ds.field('n_nationkey') < 10).to_pandas()}")

    orders   = ds.dataset(os.path.join(dataset_path, "orders"), format=format)
    part     = ds.dataset(os.path.join(dataset_path, "part"), format=format)
    partsupp = ds.dataset(os.path.join(dataset_path, "partsupp"), format=format)
    # print(f"arrow dataset: {partsupp.to_table().to_pandas()}")
    drop_caches()

    # for command
    with open(f"NDP_test/queries/q{query_no}.sql", "r") as f:
        query = f.read()

    query = f"PRAGMA disable_object_cache;\nPRAGMA threads={mp.cpu_count()};\n{query}"
    # query = "SELECT * FROM lineitem LIMIT 20;"
    for _ in range(2000):
        # drop_caches()
        conn = duckdb.connect()
        s = time.time()
        result = conn.execute(query).fetchall()
        print(result)
        e = time.time()
        conn.close()

        log_str = f"{query_no}|{format}|{e - s}"
        print(log_str)

        data.append({
            "query": query_no,
            "format": format,
            "latency": e - s
        })

    # with open(f"cse215/results/current_results_1/bench_result.{query_no}.{format}.json", "w") as f:
    #     f.write(json.dumps(data))
    print("Benchmark finished")
