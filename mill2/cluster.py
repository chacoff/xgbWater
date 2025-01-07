'''
WITH FilteredMeasures AS (
    SELECT TimeStamp, Filename
    FROM Measures
    WHERE TimeStamp < '2024-12-20 15:22:00,000' AND Filename = 'Pass 2'
)

SELECT MIN(TimeStamp) AS TimeStamp, Filename
FROM FilteredMeasures
UNION ALL
SELECT MAX(TimeStamp) AS TimeStamp, Filename
FROM FilteredMeasures;
'''

import sqlite3
import numpy as np
from sqlite3 import Connection, Cursor
from datetime import datetime
from sklearn.cluster import KMeans

s_format = '%Y-%m-%d %H:%M:%S,%f'
con: Connection = sqlite3.connect('processed.db')
cur: Cursor = con.cursor()

res: Cursor = cur.execute("SELECT TimeStamp FROM Measures WHERE Timestamp < '2024-12-20 15:22:00,000'")

row: tuple[str]
stamps_unix: list[float] = []

for row in res:
    timestamp_unix: float = datetime.strptime(row[0], s_format).timestamp()
    stamps_unix.append(timestamp_unix)

timestamps_array = np.array(stamps_unix).reshape(-1, 1)

# KMeans for N clusters
N: int = 3
state: int = 42
kmeans = KMeans(n_clusters=N, random_state=state, n_init='auto').fit(timestamps_array)
labels = kmeans.labels_

# Group by clusters
clusters = {i: [] for i in range(3)}
for ts, label in zip(stamps_unix, labels):
    stamp_string: str = datetime.fromtimestamp(ts).strftime(s_format)
    clusters[label].append(stamp_string)

# for cluster, values in clusters.items():
#     print(f"Pass {cluster}: from {values[0]} to {values[len(values)-1]}")

sorted_clusters = sorted(clusters.values(), key=lambda x: min(x))

print("Sorted clusters:")
for i, cluster in enumerate(sorted_clusters, start=1):
    print(f"Cluster {i}: from {cluster[0]} to {cluster[len(cluster)-1]}")