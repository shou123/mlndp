#!/bin/bash
set -x

# 节点列表
nodes=("worker4" "worker5" "worker6" "worker7")


storage_dir="/home/yue21/mlndp/NDP_test/data_collection"
# 确保存储目录存在
mkdir -p "$storage_dir"

# 执行远程命令的函数
remote_exec() {
    node=$1
    # 临时文件路径，这里假设远程节点上有足够权限写入/tmp，或者选择其他有权限的目录
    remote_tmp_file="/tmp/${node}_monitor_cpu.csv"
    # 最终文件将存储在worker4上的路径
    local_file_path="${storage_dir}/${node}_monitor_cpu.csv"

    echo "Recording CPU utilization for $node..."
    if [ "$node" == "worker4" ]; then
        # 直接记录到最终路径
        dstat -D dev/vda3 -tc 1 100 >> "$local_file_path" 2>&1
    else
        # 在远程节点上执行dstat并将输出保存到临时文件中
        ssh yue21@$node "dstat -D dev/vda3 -tc 1 100 > $remote_tmp_file 2>&1"
        # 然后将文件从远程节点传输到worker4
        scp yue21@$node:"$remote_tmp_file" "$local_file_path"
        # 可选：删除远程节点上的临时文件
        ssh yue21@$node "rm -f $remote_tmp_file"
    fi
    echo "Recording finished for $node"
}

# 对每个节点执行操作
for node in "${nodes[@]}"; do
    remote_exec "$node" &
done

# 等待所有后台任务完成
wait

echo "Recording finished for all nodes."
