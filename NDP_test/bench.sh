#!/bin/bash
set -ex

mkdir current_results/

for i in {1..22}; do
    echo "Running query $i"
    python3 bench_single.py /mnt/cephfs/tpch_sf100_parquet ${i} parquet
    sleep 2
    python3 bench_single.py /mnt/cephfs/tpch_sf100_parquet ${i} skyhook
done
