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

cd /home/yue21/mlndp/skyhookdm/deployment

echo "[3] initializng Ceph config"
ceph-deploy new $MON_LIST

echo "[4] installing Ceph packages on all the hosts"
ceph-deploy install --release octopus $MON_LIST $OSD_LIST $MDS_LIST $MGR_LIST

echo "[5] deploying MONs"
ceph-deploy mon create-initial
ceph-deploy admin $MON_LIST

echo "[6] deploying MGRs"
ceph-deploy mgr create $MGR_LIST

# update the Ceph config to allow pool deletion and to recognize object class libs.
cat >> ceph.conf << EOF
mon allow pool delete = true
osd class load list = *
osd op threads = 8
EOF

# deploy the updated Ceph config and restat the MONs for the config to take effect
ceph-deploy --overwrite-conf config push $MON_LIST $OSD_LIST $MDS_LIST $MGR_LIST
read -p "Press Enter to continue..."



# for node in ${MON_LIST}; do
#     ssh $node systemctl restart ceph-mon.target
# done
for node in ${MON_LIST}; do
    if [ "$node" == "worker1" ]; then
        sudo systemctl restart ceph-mon.target
    else
        ssh $node sudo systemctl restart ceph-mon.target
    fi
done
read -p "Press Enter to continue..."
# copy the config to the default location on the admin node
sudo cp ceph.conf /etc/ceph/ceph.conf
sudo cp ceph.client.admin.keyring  /etc/ceph/ceph.client.admin.keyring

# pause and let user's can take a quick look if everything is fine before deploying OSDs
ceph -s
sleep 5
read -p "Press Enter to continue..."
echo "[7] deploying OSDs"
# for node in ${OSD_LIST}; do
#     scp /home/yue21/skyhookdm/deployment/ceph.bootstrap-osd.keyring $node:/etc/ceph/ceph.keyring
#     scp /home/yue21/skyhookdm/deployment/ceph.bootstrap-osd.keyring $node:/var/lib/ceph/bootstrap-osd/ceph.keyring
#     ceph-deploy osd create --data $BLKDEV $node
# done
for node in ${OSD_LIST}; do
    if [ "$node" == "worker1" ]; then
        sudo cp /home/yue21/mlndp/skyhookdm/deployment/ceph.bootstrap-osd.keyring /etc/ceph/ceph.keyring
        sudo cp /home/yue21/mlndp/skyhookdm/deployment/ceph.bootstrap-osd.keyring /var/lib/ceph/bootstrap-osd/ceph.keyring
        ceph-deploy osd create --data $BLKDEV $node
    else
        sudo scp -o StrictHostKeyChecking=no /home/yue21/mlndp/skyhookdm/deployment/ceph.bootstrap-osd.keyring $node:/etc/ceph/ceph.keyring
        sudo scp -o StrictHostKeyChecking=no /home/yue21/mlndp/skyhookdm/deployment/ceph.bootstrap-osd.keyring $node:/var/lib/ceph/bootstrap-osd/ceph.keyring
        ceph-deploy osd create --data $BLKDEV $node
    fi
done
read -p "Press Enter to continue..."
echo "[8] deploying MDSs"
ceph-deploy mds create $MDS_LIST

echo "[9] creating pools for deploying CephFS"
sudo ceph osd pool create cephfs_data 128
sudo ceph osd pool create cephfs_metadata 16

# turn off pg autoscale
ceph osd pool set cephfs_data pg_autoscale_mode off

# set the pool sizes based on commandline arguments
sudo ceph osd pool set cephfs_data size $POOL_SIZE
sudo ceph osd pool set cephfs_metadata size $POOL_SIZE
sudo ceph osd pool set device_health_metrics size $POOL_SIZE
read -p "Press Enter to continue..."

echo "[9] deploying CephFS"
sudo apt-get install ceph-fuse
sudo ceph fs new cephfs cephfs_metadata cephfs_data
sudo mkdir -p /mnt/cephfs

read -p "Press Enter to continue..."                                    
                                                                                                                                                                                                                                             
echo "[10] mounting CephFS at /mnt/cephfs"
sleep 5
ceph-fuse /mnt/cephfs

echo "Ceph deployed successfully !"
ceph -s