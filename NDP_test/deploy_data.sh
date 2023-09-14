#!/bin/bash
set -eu

wget https://skyhook-ucsc.s3.us-west-1.amazonaws.com/64MB.uncompressed.parquet

apt update
apt install -y attr

source=64MB.uncompressed.parquet
destination=/mnt/cephfs/dataset

mkdir -p ${destination}

for ((i=1 ; i<=460 ; i++)); do
    uuid=$(uuidgen)
    filename=${destination}/${uuid}.parquet
    touch ${filename}
    setfattr -n ceph.file.layout.object_size -v 67108864 ${filename}
    echo "copying ${source} to ${filename}"
    cp ${source} ${filename}
done

sleep 2