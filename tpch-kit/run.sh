#!/bin/bash
set -e

cd dbgen/
make MACHINE=LINUX

mkdir -p /mnt/cephfs/tpch_sf2/lineitem
mkdir -p /mnt/cephfs/tpch_sf2/customer
mkdir -p /mnt/cephfs/tpch_sf2/orders
mkdir -p /mnt/cephfs/tpch_sf2/part
mkdir -p /mnt/cephfs/tpch_sf2/partsupp
mkdir -p /mnt/cephfs/tpch_sf2/supplier
mkdir -p /mnt/cephfs/tpch_sf2/nation
mkdir -p /mnt/cephfs/tpch_sf2/region

python3 gen.py L /mnt/cephfs/tpch_sf2/lineitem 1
python3 gen.py O /mnt/cephfs/tpch_sf2/orders 1
python3 gen.py S /mnt/cephfs/tpch_sf2/partsupp 1
python3 gen.py P /mnt/cephfs/tpch_sf2/part 1
python3 gen.py c /mnt/cephfs/tpch_sf2/customer 1 
python3 gen.py s /mnt/cephfs/tpch_sf2/supplier 1
python3 gen.py n /mnt/cephfs/tpch_sf2/nation 1 
python3 gen.py r /mnt/cephfs/tpch_sf2/region 1 
