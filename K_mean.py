# K-Means Clustering on Mall_Customers dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ✅ Load dataset from local file (make sure file is in same folder as script)
data = pd.read_csv("Mall_Customers.csv")

print("First 5 rows of dataset:")
print(data.head())

# Select features for clustering (Annual Income, Spending Score)
X = data.iloc[:, [3, 4]].values  # columns: Annual Income, Spending Score

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()

# Train final KMeans with chosen k (example: 5)
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels
data["Cluster"] = y_kmeans

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c="red", label="Cluster 1")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c="blue", label="Cluster 2")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c="green", label="Cluster 3")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c="cyan", label="Cluster 4")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c="magenta", label="Cluster 5")

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, c="yellow", marker="X", label="Centroids")

plt.title("K-Means Clustering (Mall Customers)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Evaluate clustering using Silhouette Score
score = silhouette_score(X, y_kmeans)
print("Silhouette Score:", score)
