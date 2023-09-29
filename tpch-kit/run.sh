#!/bin/bash
set -e

cd dbgen/
make MACHINE=LINUX

mkdir -p /mnt/cephfs/tpch_sf100/lineitem
mkdir -p /mnt/cephfs/tpch_sf100/customer
mkdir -p /mnt/cephfs/tpch_sf100/orders
mkdir -p /mnt/cephfs/tpch_sf100/part
mkdir -p /mnt/cephfs/tpch_sf100/partsupp
mkdir -p /mnt/cephfs/tpch_sf100/supplier
mkdir -p /mnt/cephfs/tpch_sf100/nation
mkdir -p /mnt/cephfs/tpch_sf100/region

python3 gen.py L /mnt/cephfs/tpch_sf100/lineitem 128
python3 gen.py O /mnt/cephfs/tpch_sf100/orders 64
python3 gen.py S /mnt/cephfs/tpch_sf100/partsupp 64 
python3 gen.py P /mnt/cephfs/tpch_sf100/part 32 
python3 gen.py c /mnt/cephfs/tpch_sf100/customer 32 
python3 gen.py s /mnt/cephfs/tpch_sf100/supplier 10
python3 gen.py n /mnt/cephfs/tpch_sf100/nation 1 
python3 gen.py r /mnt/cephfs/tpch_sf100/region 1 
