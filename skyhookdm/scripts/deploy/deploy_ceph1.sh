#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
set -eu

if [[ $# -lt 6 ]] ; then
    echo "usage: ./deploy_ceph.sh [mon hosts] [osd hosts] [mds hosts] [mgr hosts] [blkdevice] [pool size]"
    echo " "
    echo "for example: ./deploy_ceph.sh node1,node2,node3 node4,node5,node6 node1 node1 /dev/sdb 3"
    exit 1
fi

# in default mode (without any arguments), deploy a single OSD Ceph cluster 
MON=${1:-node1}
OSD=${2:-node1}
MDS=${3:-node1}
MGR=${4:-node1}
BLKDEV=${5:-/dev/nvme0n1p4}
POOL_SIZE=${6:-1}

# split the comma separated nodes into a list
IFS=',' read -ra MON_LIST <<< "$MON"; unset IFS
IFS=',' read -ra OSD_LIST <<< "$OSD"; unset IFS
IFS=',' read -ra MDS_LIST <<< "$MDS"; unset IFS
IFS=',' read -ra MGR_LIST <<< "$MGR"; unset IFS
MON_LIST=${MON_LIST[@]}
OSD_LIST=${OSD_LIST[@]}
MDS_LIST=${MDS_LIST[@]}
MGR_LIST=${MGR_LIST[@]}

# disable host key checking
cat > ~/.ssh/config << EOF
Host *
    StrictHostKeyChecking no
EOF



echo "[1] installing common packages"
sudo apt update
sudo apt install -y python3-venv python3-pip ceph-fuse ceph-common attr

echo "[2] installing ceph-deploy"
git clone https://github.com/ceph/ceph-deploy /home/yue21/mlndp/skyhookdm/ceph-deploy
pip3 install --upgrade /home/yue21/mlndp/skyhookdm/ceph-deploy

mkdir /home/yue21/mlndp/skyhookdm/deployment
cd /home/yue21/mlndp/skyhookdm/deployment


