🌀 K-Means Clustering on Mall Customers Dataset
📌 Objective

Perform unsupervised learning using K-Means clustering to segment customers based on their spending patterns and demographics.

🛠 Tools & Libraries

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

📂 Dataset

Mall_Customers.csv

Features: CustomerID, Gender, Age, Annual Income, Spending Score

Task: Segment customers into distinct groups.

🚀 Steps Implemented

Load and explore the dataset

Preprocess and select features

Apply Elbow Method to find optimal number of clusters (K)

Train K-Means model

Visualize clusters in 2D (using Age & Spending Score)

Evaluate clustering using Silhouette Score

📊 Results

Optimal clusters found: 5 (using Elbow Method)

Segmentation shows distinct groups of customers with different spending behavior.

▶️ How to Run
# Clone the repository
git clone https://github.com/junnu44/K-means-clustring.git
cd KMeans-Clustering

# Install dependencies
pip install -r requirements.txt

# Run the script
python kmeans_clustering.py

📸 Visualization Example

Clusters plotted based on Age vs Spending Score to observe customer groups.
