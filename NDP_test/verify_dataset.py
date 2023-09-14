import os
import sys


if __name__ == "__main__":
    dataset_dir = str(sys.argv[1])
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".parquet"):
                file_path = os.path.join(root, file)
                file_size = os.stat(file_path).st_size/(1024*1024)

                if file_size > 16:
                    print(f"NOT OK! {file_path} = {file_size} MB")
                    raise Exception(f"Too large file: {file_path} = {file_size} MB")
                else:
                    print(f"OK ! {file_path} = {file_size} MB")
    print("All files are OK")
