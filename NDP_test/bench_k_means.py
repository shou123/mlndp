import pandas as pd
import time
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
import duckdb
import pyarrow.dataset as ds
import pyarrow.parquet as pq



if __name__ == "__main__":

    categories = ["near_data_processing","in_memory_computing","baseline"]
    # category = "near_data_processing"
    # category = "in_memory_computing"
    # category = 'baseline'

    n_clusters = [3,10,50,100] # the number of k_means
    # n_clusters = 10

    samples = [100000] # the number of samples, setup 1w sample is a base. 
    # samples = 10000 #1w
    # samples = 50000 #5w
    # samples = 100000 #10w
    # samples = 1000000 #100w


    rounds = 10

    for category in categories:

        for cluster in n_clusters:
            for sample in samples:

                if category == "near_data_processing" or category == "in_memory_computing":
                    if category == "near_data_processing":
                        format_ = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
                    elif category == "in_memory_computing":
                        format_ = "parquet"

                    dataset = ds.dataset('/mnt/cephfs/k_means',format = format_)

                    query = f'SELECT Feature_1, Feature_2 FROM dataset LIMIT {sample}'

                    for i in range (rounds):

                        s = time.time()
                        if category == "in_memory_computing" and cluster >= 20 and sample > 5000:
                            time.sleep(0.2)
                        # if category == "non_skyhook" and (sample>50000 or cluster>10):
                        #     time.sleep(2)
                        conn = duckdb.connect()
                        # df = pd.read_sql(query, conn)
                        df = conn.execute(query).fetchdf()

                        features = df[['Feature_1', 'Feature_2']]

                        kmeans = km(cluster, random_state=100,n_init=10)
                        kmeans.fit(features)
                        e = time.time()

                        with open('NDP_test/data_collection/k_means_latency.txt', 'a') as file:
                            file.write(f"category: {category}, cluster: {cluster}, samples: {sample}, round: {i}, latency: {e - s}\n")

                        print(f"category: {category}, cluster: {cluster}, samples: {sample}, round: {i}, latency: {e - s}")

                elif category == "baseline":

                    for i in range (rounds):

                        s = time.time()
                        table = pq.read_table('/mnt/cephfs/k_means')
                        df = table.to_pandas()

                        features = df[['Feature_1', 'Feature_2']].head(sample)

                        kmeans = km(cluster, random_state=100,n_init=10)
                        kmeans.fit(features)
                        e = time.time()

                        with open('NDP_test/data_collection/k_means_latency.txt', 'a') as file:
                            file.write(f"category: {category}, cluster: {cluster}, samples: {sample}, round: {i}, latency: {e - s}\n")

                        print(f"category: {category}, cluster: {cluster}, samples: {sample}, round: {i}, latency: {e - s}")

                        # plt.scatter(df['Feature_1'], df['Feature_2'], c=kmeans.labels_, cmap='viridis')
                        # plt.xlabel('Feature_1')
                        # plt.ylabel('Feature_2')
                        # plt.title('Scatter plot of Age vs YearlyIncome (Clustered)')
                        # plt.show()
                        # Close your SQL Server database connection














