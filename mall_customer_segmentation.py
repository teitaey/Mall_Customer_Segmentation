import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import os

# Create an output directory for saving the plots
os.makedirs("output_images", exist_ok=True)

# 1. Load Dataset
print("Loading dataset...")
dataset = pd.read_csv('Mall_Customers.csv')
print(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

# We only need Annual Income (index 3) and Spending Score (index 4)
X = dataset.iloc[:, [3, 4]].values

# 2. Using the Elbow Method to find the optimal number of clusters
print("Running Elbow Method...")
wcss = []
for i in range(1, 11):
    # k-means++ is used to avoid random initialization trap
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', color='purple', linestyle='--')
plt.title('The Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig("output_images/elbow_method.png", dpi=300, bbox_inches='tight')
print("Saved elbow curve to 'output_images/elbow_method.png'")
plt.close()

# 3. Training the K-Means model on the dataset with optimal K=5
print("Training K-Means model with K=5...")
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 4. Visualising the Clusters
print("Generating cluster visualization...")
plt.figure(figsize=(12, 8))

# Define colors and intuitive labels based on the clusters formed
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Careful (High Income, Low Spend)', 
          'Standard (Mid Income, Mid Spend)', 
          'Target / Stars (High Income, High Spend)', 
          'Careless (Low Income, High Spend)', 
          'Sensible (Low Income, Low Spend)']

# Plot each cluster
for i in range(5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                s=100, c=colors[i], label=labels[i], alpha=0.7, edgecolors='black')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', edgecolors='black', label='Centroids', marker='*')

plt.title('Mall Customer Segmentation', fontsize=18, fontweight='bold')
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("output_images/customer_clusters.png", dpi=300, bbox_inches='tight')
print("Saved cluster visualization to 'output_images/customer_clusters.png'")
plt.close()

# 5. Save the segmented data to a new CSV
dataset['Cluster'] = y_kmeans
dataset['Segment_Name'] = dataset['Cluster'].map({
    0: 'Careful',
    1: 'Standard',
    2: 'Target / Stars',
    3: 'Careless',
    4: 'Sensible'
})
dataset.to_csv("output_images/Segmented_Customers.csv", index=False)
print("Saved segmented dataset to 'output_images/Segmented_Customers.csv'")
print("Project execution completed successfully!")
