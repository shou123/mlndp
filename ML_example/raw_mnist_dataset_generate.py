import torchvision
from torchvision import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

def create_dataframe(dataset):
    for item in dataset:
        pass

    data = {
        'label': [item[1] for item in dataset],
        'image': [np.array(item[0]).flatten() for item in dataset]
    }
    return pd.DataFrame(data)


def save_to_parquet(df, file_path):
    table = pa.Table.from_pandas(df)
    return pq.write_table(table, file_path)

def save_to_parquet(df, file_path):
    table = pa.Table.from_pandas(df)
    return pq.write_table(table, file_path)


def main():
    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='/home/yue21/mlndp/mnist_dataset', train=True, download=True)
    test_dataset = datasets.MNIST(root='/home/yue21/mlndp/mnist_dataset', train=False, download=True)

    train_df = create_dataframe(train_dataset)
    test_df = create_dataframe(test_dataset)    

    # Save the train and test dataframes as Parquet files
    train_parquet = save_to_parquet(train_df, '/mnt/cephfs/raw_minist_dataset/train/mnist_train.parquet')
    test_parquet = save_to_parquet(test_df, '/mnt/cephfs/raw_minist_dataset/test/mnist_test.parquet')


if __name__ == '__main__':
    main()
