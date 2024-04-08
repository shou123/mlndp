import pandas as pd
import time
from sklearn.cluster import KMeans as km
import matplotlib.pyplot as plt
import duckdb
import pyarrow.dataset as ds
import pyarrow.parquet as pq



if __name__ == "__main__":

    category = "pure_k_means"
    # category = "skyhook"
    # category = "non_skyhook"

    n_clusters = 3 #the number of k_means

    if category == "skyhook" or category == "non_skyhook":
        if category == "skyhook":
            format_ = ds.SkyhookFileFormat("parquet", "/etc/ceph/ceph.conf")
        elif category == "non_skyhook":
            format_ = "parquet"

        dataset = ds.dataset('/mnt/cephfs/k_means',format = format_)

        query = 'SELECT Feature_1, Feature_2 FROM dataset LIMIT 1000'

        for i in range (100):

            s = time.time()
            conn = duckdb.connect()
            # df = pd.read_sql(query, conn)
            df = conn.execute(query).fetchdf()

            features = df[['Feature_1', 'Feature_2']]

            kmeans = km(n_clusters, random_state=42,n_init=10)
            kmeans.fit(features)
            e = time.time()

            with open('NDP_test/data_collection/k_means_latency.txt', 'a') as file:
                file.write(f"category: {category}, round: {i}, latency: {e - s}\n")

            print(f"execute_k_means_rounds: {i}, execute_time: {e-s}")

    elif category == "pure_k_means":

        for i in range (100):

            s = time.time()
            table = pq.read_table('/mnt/cephfs/k_means')
            df = table.to_pandas()

            features = df[['Feature_1', 'Feature_2']].head(1000)

            kmeans = km(n_clusters, random_state=42,n_init=10)
            kmeans.fit(features)
            e = time.time()

            with open('NDP_test/data_collection/k_means_latency.txt', 'a') as file:
                file.write(f"category: {category}, round: {i}, latency: {e - s}\n")

            print(f"execute_k_means_rounds: {i}, execute_time: {e-s}")

            # plt.scatter(df['Feature_1'], df['Feature_2'], c=kmeans.labels_, cmap='viridis')
            # plt.xlabel('Feature_1')
            # plt.ylabel('Feature_2')
            # plt.title('Scatter plot of Age vs YearlyIncome (Clustered)')
            # plt.show()
            # Close your SQL Server database connection





