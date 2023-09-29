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

if [[ $# -lt 1 ]] ; then
    echo "./deploy_skyhook.sh [nodes] [branch] [deploy CLS libs] [build python] [build java] [nproc]"
    exit 1
fi

NODES=$1
# BRANCH=${2:-arrow-master}
BRANCH=${2:-main}
DEPLOY_CLS_LIBS=${3:-true}
BUILD_PYTHON_BINDINGS=${4:-true}
BUILD_JAVA_BINDINGS=${5:-false}
NPROC=${6:-4}

IFS=',' read -ra NODE_LIST <<< "$NODES"; unset IFS

apt update 
apt install -y python3 \
               python3-pip \
               python3-venv \
               python3-numpy \
               cmake \
               libradospp-dev \
               rados-objclass-dev \
               llvm \
               default-jdk \
               maven

read -p "install python package..." 


if [ ! -d "/home/yue21/mlndp/skyhookdm/arrow" ]; then
  echo "clone skyhook repository"
  # git clone https://github.com/uccross/skyhookdm-arrow /home/yue21/mlndp/skyhookdm/arrow
  git clone https://github.com/shou123/skyhook-arrow.git /home/yue21/mlndp/skyhookdm/arrow

  cd /home/yue21/mlndp/skyhookdm/arrow
  git submodule update --init --recursive
fi

read -p "Finish clone skyhook-arrow..."  
cd /home/yue21/mlndp/skyhookdm/arrow
git fetch origin $BRANCH
git pull
git checkout $BRANCH
mkdir -p cpp/release
cd cpp/release

read -p "Finish skyhook checkout..."  
cmake -DARROW_SKYHOOK=ON \
  -DARROW_PARQUET=ON \
  -DARROW_WITH_SNAPPY=ON \
  -DARROW_WITH_ZLIB=ON \
  -DARROW_BUILD_EXAMPLES=ON \
  -DPARQUET_BUILD_EXAMPLES=ON \
  -DARROW_PYTHON=ON \
  -DARROW_ORC=ON \
  -DARROW_JAVA=ON \
  -DARROW_JNI=ON \
  -DARROW_DATASET=ON \
  -DARROW_CSV=ON \
  -DARROW_WITH_LZ4=ON \
  -DARROW_WITH_ZSTD=ON \
  ..
# read -p "cmake the skyhook..."
make -j${NPROC} install
# read -p "finish build the skyhook, press enter to continue..."
read -p "Finish cmake config and build..."  

if [[ "${BUILD_PYTHON_BINDINGS}" == "true" ]]; then
  # read -p "build python bingings..."
  export WORKDIR=${WORKDIR:-$HOME}
  export ARROW_HOME=$WORKDIR/dist
  export PYARROW_WITH_DATASET=1
  export PYARROW_WITH_PARQUET=1
  export PYARROW_WITH_SKYHOOK=1

  # read -p "copy the lib..."
  mkdir -p /root/dist/lib
  mkdir -p /root/dist/include

  cp -r /usr/local/lib/. /root/dist/lib
  cp -r /usr/local/include/. /root/dist/include
  # read -p "finish copy the lib..."

  read -p "install some python dependency..."
  cd /home/yue21/mlndp/skyhookdm/arrow/python
  pip3 install -r requirements-build.txt -r requirements-test.txt
  # read -p "install the require build..."
  pip3 install wheel
  # read -p "install the wheel..."
  rm -rf dist/*
  # read -p "remove the dist/..."

  # build_ext is used to build C/C++ extensions included in the package.
  # --inplace: This option tells setuptools to build the C/C++ extensions directly in the source directory
  # --bundle-arrow-cpp: This option is specific to Arrow. It indicates that the Arrow C++ libraries should be bundled into the package, allowing the package to be distributed with its own copy of the Arrow C++ libraries.
  # bdist_wheel: This is a command to create a Wheel distribution of the package. 
  read -p "install setup.py..."
  python3 setup.py build_ext --inplace --bundle-arrow-cpp bdist_wheel
  read -p "finish install setup.py..."
  pip3 install --upgrade dist/*.whl
  read -p "finish build python bingings..."
fi

if [[ "${DEPLOY_CLS_LIBS}" == "true" ]]; then
  cd /home/yue21/mlndp/skyhookdm/arrow/cpp/release/release
  for node in ${NODE_LIST[@]}; do
    scp libcls* $node:/usr/lib/rados-classes/
    scp libarrow* $node:/usr/lib/
    scp libparquet* $node:/usr/lib/
    ssh $node systemctl restart ceph-osd.target
  done
fi

if [[ "${BUILD_JAVA_BINDINGS}" == "true" ]]; then
    mkdir -p /home/yue21/mlndp/skyhookdm/arrow/java/dist
    cp -r /home/yue21/mlndp/skyhookdm/arrow/cpp/release/release/libarrow_dataset_jni.so* /home/yue21/skyhookdm/arrow/java/dist

    mvn="mvn -B -DskipTests -Dcheckstyle.skip -Drat.skip=true -Dorg.slf4j.simpleLogger.log.org.apache.maven.cli.transfer.Slf4jMavenTransferListener=warn"
    mvn="${mvn} -T 2C"
    cd /home/yue21/mlndp/skyhookdm/arrow/java
    ${mvn} clean install package -P arrow-jni -pl dataset,format,memory,vector -am -Darrow.cpp.build.dir=/tmp/arrow/cpp/release/release
fi

export LD_LIBRARY_PATH=/usr/local/lib
cp /usr/local/lib/libparq* /usr/lib/
