# Mall Customer Segmentation
## Presentation Slide Deck

### Slide 1: Title & Objective
**Title:** Customer Segmentation using K-Means Clustering
**Presenter:** [Your Name]
**Objective:** Transform raw mall customer data into distinct, actionable segments using Unsupervised Machine Learning, enabling the marketing team to run targeted, cost-effective campaigns.

---

### Slide 2: The Dataset & Problem Statement
- **The Problem:** The mall has customer data but doesn't know who its distinct customer personas are. Sending the same promotion to everyone is inefficient and expensive.
- **The Solution:** Use Machine Learning to automatically discover these personas.
- **The Data:** 
  - `CustomerID`, `Gender`, `Age` (Demographics)
  - `Annual Income` (How much they make)
  - `Spending Score` (How much they spend at the mall, from 1-100)

*(Speaker Notes: We focused specifically on Annual Income and Spending Score because these two metrics are the strongest indicators of purchasing behavior.)*

---

### Slide 3: The Methodology (K-Means Clustering)
- **Algorithm Used:** K-Means Clustering (Unsupervised Learning)
- **Why Unsupervised?** We do not have "labeled" data. We don't know the categories beforehand; the algorithm discovers them for us.
- **How it works:** 
  1. Chooses *K* random center points (centroids).
  2. Assigns each customer to the closest centroid.
  3. Re-calculates the center of each group and repeats until stable.

---

### Slide 4: Finding the Magic Number (The Elbow Method)
*(Insert Image: `output_images/elbow_method.png`)*
- **Question:** How do we know how many clusters (groups) to create?
- **Answer:** The Elbow Method!
- We calculate the "WCSS" (Within-Cluster Sum of Squares) for 1 to 10 clusters.
- As the graph shows, there is a clear "elbow bend" at **K = 5**. This mathematically proves that 5 is the optimal number of customer segments for this mall.

---

### Slide 5: The Results (Customer Segments)
*(Insert Image: `output_images/customer_clusters.png`)*
- Our model successfully grouped all customers into **5 distinct personas**.
- The yellow stars represent the geometric center (centroid) of each persona group.

---

### Slide 6: Business Persona Breakdown
Based on the graph, here are the 5 target personas we discovered:

1. **Target / Stars (Green):** High Income, High Spend
   - *Action:* Offer VIP memberships, luxury brand promotions, and exclusive early-access sales.
2. **Careful (Red):** High Income, Low Spend
   - *Action:* They have the money but need convincing. Send targeted, high-value discount bundles.
3. **Careless (Cyan):** Low Income, High Spend
   - *Action:* Young, impulsive buyers. Offer loyalty reward points and trendy, affordable items.
4. **Sensible (Magenta):** Low Income, Low Spend
   - *Action:* Budget-conscious shoppers. Send clearance sale notifications.
5. **Standard (Blue):** Average Income, Average Spend
   - *Action:* The baseline shopper. General mass-market marketing.

---

### Slide 7: Conclusion & Business Impact
- **Impact:** By shifting from a "one-size-fits-all" marketing strategy to a targeted, data-driven strategy, the mall can drastically increase conversion rates and reduce marketing waste.
- **Next Steps:** Integrate these segments into our CRM software so every new customer is automatically assigned a marketing persona based on their early spending habits.
