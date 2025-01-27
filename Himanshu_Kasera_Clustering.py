import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load datasets
customers_df = pd.read_csv('Customers.csv')  # Customer profile data
transactions_df = pd.read_csv('Transactions.csv')  # Transaction data

# Step 1: Aggregate transaction data by CustomerID
agg_transactions = transactions_df.groupby('CustomerID').agg(
    total_spent=('TotalValue', 'sum'),  # Total amount spent by each customer
    frequency=('TransactionID', 'count'),  # Number of transactions by each customer
    avg_spent_per_transaction=('TotalValue', 'mean')  # Average amount spent per transaction
).reset_index()

# Step 2: Merge customer profile data with aggregated transaction data
customer_data = pd.merge(customers_df, agg_transactions, on='CustomerID')

# Step 3: Handle missing values by imputing the mean
imputer = SimpleImputer(strategy='mean')
customer_data_imputed = imputer.fit_transform(customer_data[['total_spent', 'frequency', 'avg_spent_per_transaction']])

# Step 4: Standardize the data to bring all features to a comparable scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data_imputed)

# Step 5: Experiment with different cluster numbers (2 to 10) and calculate the Davies-Bouldin Index for each
best_n_clusters = None
lowest_db_index = float('inf')
db_scores = []
kmeans_models = {}

# Try different cluster numbers and calculate the DB index for each
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    db_index = davies_bouldin_score(scaled_data, cluster_labels)
    db_scores.append(db_index)
    kmeans_models[n_clusters] = kmeans
    
    # Track the best number of clusters (lowest DB Index)
    if db_index < lowest_db_index:
        lowest_db_index = db_index
        best_n_clusters = n_clusters

# Step 6: Perform clustering with the optimal number of clusters
optimal_kmeans = kmeans_models[best_n_clusters]
customer_data['Cluster'] = optimal_kmeans.predict(scaled_data)

# Step 7: Print the optimal number of clusters and the corresponding DB Index
print(f"Optimal Number of Clusters: {best_n_clusters}")
print(f"Best Davies-Bouldin Index: {lowest_db_index:.4f}")

# Step 8: Plot the Davies-Bouldin Index vs. Number of Clusters
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), db_scores, marker='o', linestyle='--')
plt.title('Davies-Bouldin Index vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index (Lower is Better)')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()

# Step 9: Visualize the clusters in a 2D space using PCA (Principal Component Analysis)
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=customer_data['Cluster'], cmap='viridis', s=50, alpha=0.8)
plt.title(f'Customer Segmentation Visualization (k={best_n_clusters})')
plt.xlabel('Principal Component 1 (Customer Behavior)')
plt.ylabel('Principal Component 2 (Customer Behavior)')
plt.colorbar(label='Cluster')
plt.show()

# Step 10: Aggregate metrics for each cluster to summarize the characteristics
cluster_summary = customer_data.groupby('Cluster').agg(
    avg_total_spent=('total_spent', 'mean'),
    avg_frequency=('frequency', 'mean'),
    avg_spent_per_transaction=('avg_spent_per_transaction', 'mean'),
    customer_count=('CustomerID', 'count')
).reset_index()

# Print cluster summary
print("\nCluster-wise Summary:")
print(cluster_summary)

# Step 11: Save the final results to CSV files
customer_data.to_csv('clustered_customers.csv', index=False)  # Final data with cluster labels
cluster_summary.to_csv('cluster_summary.csv', index=False)  # Summary of cluster characteristics
