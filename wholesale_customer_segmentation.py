import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Create an output directory for saving the plots
os.makedirs("wholesale_output_images", exist_ok=True)

# 1. Load Real Public Dataset (UCI Wholesale Customers)
print("Loading Wholesale Customers dataset...")
dataset = pd.read_csv('Wholesale_Customers.csv')
print(f"Dataset loaded with {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

# We drop 'Channel' and 'Region' to focus strictly on continuous spending behaviors
# Features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
X = dataset.drop(['Channel', 'Region'], axis=1).values

# 2. Feature Scaling (Crucial for real-world multi-dimensional data)
print("Applying Standard Scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Dimensionality Reduction (PCA) to squash 6 dimensions down to 2 for 2D visualization
print("Applying Principal Component Analysis (PCA)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained Variance by 2 Components: {sum(pca.explained_variance_ratio_)*100:.2f}%")

# 4. Using the Elbow Method on the PCA data to find the optimal K
print("Running Elbow Method...")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', color='darkred', linestyle='--')
plt.title('The Elbow Method for Wholesale Customers', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('WCSS', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig("wholesale_output_images/elbow_method.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Training K-Means with optimal K (Let's use K=4 based on the typical wholesale elbow curve)
print("Training K-Means model with K=4...")
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# 6. Visualising the Clusters in 2D PCA Space
print("Generating cluster visualization...")
plt.figure(figsize=(12, 8))

colors = ['blue', 'green', 'orange', 'purple']
labels = ['Cluster 1 (Retail/Grocery Heavy)', 
          'Cluster 2 (Small Cafes/Fresh Food)', 
          'Cluster 3 (High-Volume Supermarkets)', 
          'Cluster 4 (Bulk/Wholesale Giants)']

for i in range(4):
    plt.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
                s=70, c=colors[i], label=labels[i], alpha=0.6, edgecolors='black')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='yellow', edgecolors='black', label='Centroids', marker='*')

plt.title('Wholesale Customer Segmentation (PCA reduced)', fontsize=18, fontweight='bold')
plt.xlabel('Principal Component 1 (Grocery & Detergents)', fontsize=14)
plt.ylabel('Principal Component 2 (Fresh & Frozen Foods)', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("wholesale_output_images/customer_clusters.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Save segmented real data
dataset['Cluster'] = y_kmeans
dataset.to_csv("wholesale_output_images/Segmented_Wholesale_Customers.csv", index=False)
print("Project execution completed successfully!")
