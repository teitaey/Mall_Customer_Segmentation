# Real-World Wholesale Customer Segmentation
## Presentation Slide Deck

### Slide 1: Title & Objective
**Title:** Real-World Customer Segmentation using PCA & K-Means Clustering
**Presenter:** [Your Name]
**Objective:** Apply unsupervised machine learning to a real-world public dataset from the UCI Machine Learning Repository to identify distinct purchasing patterns among wholesale clients.

---

### Slide 2: The Dataset & The Challenge
- **Dataset:** Wholesale Customers Data Set (Publicly available via UCI).
- **The Data:** 440 distinct wholesale clients (clients of a distributor in Portugal).
- **Features:** Annual spending across 6 categories: 
  - `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`
- **The Challenge:** Unlike standard 2D tutorial datasets, this dataset has **6 dimensions**. We cannot easily plot or visualize 6-dimensional spending habits. How do we solve this?

---

### Slide 3: The Methodology (Scaling & PCA)
- **Step 1: Data Scaling**
  - Because "Fresh" food spending might be in the tens of thousands and "Delicassen" in the hundreds, we used a **Standard Scaler**. This ensures K-Means doesn't get biased by large numbers.
- **Step 2: Principal Component Analysis (PCA)**
  - We applied **PCA** (Dimensionality Reduction) to compress the 6 spending categories down to exactly **2 Principal Components**.
  - *Result:* Our 2 components successfully captured **72.46% of the total variance** in the data! We can now visualize this mathematically complex data in 2D.

---

### Slide 4: Finding the Magic Number (The Elbow Method)
*(Insert Image: `wholesale_output_images/elbow_method.png`)*
- Now that our data is scaled and reduced to 2D, how many customer segments do we have?
- We calculated the WCSS (Within-Cluster Sum of Squares) for 1 to 10 clusters.
- The "elbow bend" naturally occurs around **K = 4**. This tells us the wholesale distributor essentially has 4 types of clients.

---

### Slide 5: The Results (Customer Segments)
*(Insert Image: `wholesale_output_images/customer_clusters.png`)*
- Our model successfully grouped all 440 clients into **4 distinct business personas**.
- The yellow stars represent the centroid (average behavior) of each client group.

---

### Slide 6: Business Persona Breakdown
By mapping the PCA clusters back to their original spending habits, we identified:

1. **Cluster 1 (Blue): Retail & Grocery Heavy**
   - High spending in Milk, Grocery, and Detergents.
   - *Likely Identity:* Standard Retail Supermarkets.
2. **Cluster 2 (Green): Small Cafes & Fresh Food**
   - High spending in Fresh Food and Frozen Goods, lower in paper/detergents.
   - *Likely Identity:* Small restaurants, cafes, and local eateries (Horeca).
3. **Cluster 3 (Orange): High-Volume Supermarkets**
   - Extremely high spend across all categories, heavily leaning right on Principal Component 1.
   - *Likely Identity:* Large national supermarket chains.
4. **Cluster 4 (Purple): Bulk/Wholesale Giants**
   - Massive outliers who buy an immense amount of Fresh and Frozen goods (leaning high on Principal Component 2).
   - *Likely Identity:* Massive bulk distributors or very large hospitality networks.

---

### Slide 7: Conclusion & Business Impact
- **Technical Achievement:** Successfully combined Data Scaling, Dimensionality Reduction (PCA), and Unsupervised Learning (K-Means) on real-world, high-dimensional public data.
- **Business Impact:** The wholesale distributor can now stop sending generic catalogs. They can create customized supply contracts: offering bulk fresh-food discounts to the cafes (Green) and detergent/grocery loyalty programs to the retailers (Blue).
