# Benchmark Instructions

0. Install dependencies.
```
sudo apt-get install git make gcc
```

1. Clone the TPCH kit github repository.
```

git clone https://github.com/uccross/tpch-kit
cd tpch-kit/
./run.sh
```

2. Convert the CSV files into Parquet format.
```python
python3 csv_to_parquet.py /mnt/cephfs/tpch_sf100 /mnt/cephfs/tpch_sf100_parquet
```

3. Split the large parquet files (> 16MB) into smaller chunks.
```python
python3 split_dataset.py /mnt/cephfs/tpch_sf100_parquet
```

4. Verify if the dataset has been properly split or not. 
```python
python3 verify_dataset.py /mnt/cephfs/tpch_sf100_parquet
```
