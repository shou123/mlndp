import os
import sys
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def clean_string(path):
    path = path.replace(".", "_")
    path = path.replace("/", "_")
    return path


if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = (20, 10)
    # result_dir = str(sys.argv[1])    
    # queries = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    # data = list()

    # for query in queries:
    #     parquet_path = os.path.join(result_dir, "bench_result." + str(query) + ".parquet.json")
    #     skyhook_path = os.path.join(result_dir, "bench_result." + str(query) + ".skyhook.json")

    #     with open(parquet_path) as f:
    #         parquet_result = json.load(f)

    #     with open(skyhook_path) as f:
    #         skyhook_result = json.load(f)
        
    #     for point in parquet_result:
    #         data.append(point)
            
    #     for point in skyhook_result:
    #         data.append(point)

    # df = pd.DataFrame(data) 
    # print(df)

    # df.to_csv('data.csv')

    df = pd.read_csv('data_cleaned.csv')
    sns.barplot(data=df, ci="sd", x="query", y="latency", hue="format")
    plt.savefig("plot.pdf")
