import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛒 Customer Insights & Persona Dashboard")
st.markdown("Welcome! This dashboard automatically groups your wholesale customers into distinct **Shopping Personas** based on their annual purchasing habits. Use the slider on the left to see how the groups change.")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('Wholesale_Customers.csv')

dataset = load_data()

# Sidebar for controls
st.sidebar.markdown("### **Dashboard Controls**")
k_clusters = st.sidebar.slider("How many customer personas do you want to create?", min_value=2, max_value=8, value=4)

# Data Preprocessing
X = dataset.drop(['Channel', 'Region'], axis=1).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Model Training
kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# Give human-friendly names to the clusters
dataset['Customer Persona'] = [f"Persona {c + 1}" for c in y_kmeans]

# --- DYNAMIC BUSINESS INSIGHTS ENGINE ---
persona_insights = {}
spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
global_mean = dataset[spending_cols].mean() * 80

for c in range(k_clusters):
    p_name = f"Persona {c + 1}"
    cluster_data = dataset[dataset['Customer Persona'] == p_name][spending_cols]
    if len(cluster_data) == 0:
        continue
    cluster_mean = cluster_data.mean() * 80
    ratios = cluster_mean / global_mean
    
    max_cat = ratios.idxmax()
    max_ratio = ratios.max()
    min_cat = ratios.idxmin()
    
    cat_names = {
        'Fresh': 'Fresh Produce', 'Milk': 'Dairy & Milk', 'Grocery': 'General Grocery',
        'Frozen': 'Frozen Goods', 'Detergents_Paper': 'Detergents & Paper', 'Delicassen': 'Delicatessen'
    }
    
    if max_ratio < 0.9:
        title = "Budget / Small-Scale Buyers"
        desc = f"Spends below the global average across all categories, heavily avoiding {cat_names.get(min_cat, min_cat)}. Likely small cafes or corner shops."
    else:
        title = f"{cat_names.get(max_cat, max_cat)} Giants"
        desc = f"They spend **{max_ratio:.1f}x** the global average on {cat_names.get(max_cat, max_cat)}. This is their absolute core business driver."
    
    persona_insights[p_name] = {'title': title, 'desc': desc}
# ----------------------------------------

st.sidebar.divider()
st.sidebar.markdown("### **Predict New Customer**")
st.sidebar.markdown("Enter annual spending (₹) to see which persona this customer belongs to.")
in_fresh = st.sidebar.number_input("Fresh Food (₹)", min_value=0, value=None, step=10)
in_milk = st.sidebar.number_input("Milk (₹)", min_value=0, value=None, step=10)
in_grocery = st.sidebar.number_input("Grocery (₹)", min_value=0, value=None, step=10)
in_frozen = st.sidebar.number_input("Frozen (₹)", min_value=0, value=None, step=10)
in_detergents = st.sidebar.number_input("Detergents & Paper (₹)", min_value=0, value=None, step=10)
in_delicassen = st.sidebar.number_input("Delicatessen (₹)", min_value=0, value=None, step=10)

if 'new_customers' not in st.session_state:
    st.session_state['new_customers'] = []

if st.sidebar.button("Predict Persona", type="primary"):
    v_fresh = in_fresh or 0
    v_milk = in_milk or 0
    v_grocery = in_grocery or 0
    v_frozen = in_frozen or 0
    v_detergents = in_detergents or 0
    v_delicassen = in_delicassen or 0
    
    # Reverse the *80 multiplier used for INR display to match the model's raw scale
    raw_data = np.array([[v_fresh/80, v_milk/80, v_grocery/80, v_frozen/80, v_detergents/80, v_delicassen/80]])
    scaled_data = scaler.transform(raw_data)
    new_customer_pca = pca.transform(scaled_data)
    predicted_cluster = kmeans.predict(new_customer_pca)[0]
    predicted_persona = f"Persona {predicted_cluster + 1}"
    
    # Store in session state so it doesn't get overwritten
    st.session_state['new_customers'].append({
        'fresh': v_fresh, 'milk': v_milk, 'grocery': v_grocery,
        'frozen': v_frozen, 'detergents': v_detergents, 'delicassen': v_delicassen,
        'pca_x': new_customer_pca[0, 0], 'pca_y': new_customer_pca[0, 1],
        'persona': predicted_persona
    })

if len(st.session_state['new_customers']) > 0:
    st.sidebar.markdown("### **Added Customers**")
    total_existing = len(dataset)
    for idx, cust in enumerate(st.session_state['new_customers']):
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"Customer {total_existing + idx + 1} ({cust['persona']})")
        if col2.button("❌", key=f"del_{idx}"):
            st.session_state['new_customers'].pop(idx)
            st.rerun()
            
    if st.sidebar.button("Clear ALL Customers", type="secondary"):
        st.session_state['new_customers'] = []
        st.rerun()

# Main Layout
if len(st.session_state['new_customers']) > 0:
    total_existing = len(dataset)
    st.success(f"🎉 **Prediction Complete!** You have manually added {len(st.session_state['new_customers'])} new customers.")
    for idx, cust in enumerate(st.session_state['new_customers']):
        insight = persona_insights[cust['persona']]
        st.markdown(f"- **Customer {total_existing + idx + 1}** belongs to **{cust['persona']} ({insight['title']})**. *{insight['desc']}*")

st.markdown(f"### **Visualizing your {k_clusters} Customer Personas**")
st.markdown("Each dot represents a customer. Customers grouped close together share similar buying habits.")

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot each persona cluster separately so they appear in the legend with their color
colors = plt.cm.viridis(np.linspace(0, 1, k_clusters))
for i in range(k_clusters):
    ax2.scatter(X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], color=colors[i], s=60, alpha=0.8, edgecolors='black', label=f'Persona {i+1}')

# Plot centroids
centroids = kmeans.cluster_centers_
ax2.scatter(centroids[:, 0], centroids[:, 1], s=250, c='red', marker='*', edgecolors='black', label='Average Customer (Center)')

for idx, cust in enumerate(st.session_state['new_customers']):
    lbl = 'Manual Customer Input' if idx == 0 else ""
    ax2.scatter(cust['pca_x'], cust['pca_y'], s=150, c='black', marker='X', edgecolors='white', linewidth=1, label=lbl, zorder=5)

ax2.set_xlabel('← Buys more Fresh/Frozen --- Buys more Grocery/Everyday Items →')
ax2.set_ylabel('← Smaller Orders --- Bulk/Wholesale Orders →')
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.5)
st.pyplot(fig2)

st.divider()

st.markdown("### **💡 Persona Breakdowns (Plain English)**")
st.markdown("We analyzed how each group's spending compares to the global average to determine their true business identity.")

# Display the insights dynamically in columns
insight_cols = st.columns(min(k_clusters, 4))
for i in range(k_clusters):
    col = insight_cols[i % 4]
    p_name = f"Persona {i+1}"
    if p_name in persona_insights:
        insight = persona_insights[p_name]
        with col:
            st.info(f"**{p_name}: {insight['title']}**\n\n{insight['desc']}")

st.divider()

colA, colB = st.columns([1, 2])

with colA:
    st.markdown("### **Market Size (Headcount)**")
    st.markdown("How many customers belong to each persona?")
    headcount = dataset['Customer Persona'].value_counts().sort_index()
    st.bar_chart(headcount)

with colB:
    st.markdown("### **Average Spending (₹) per Persona**")
    st.markdown("Exactly how much each persona spends on average annually.")
    # Displaying the average spend of each cluster in the original dimensions, formatted in Indian Rupees
    summary = dataset.groupby('Customer Persona').mean(numeric_only=True).drop(['Channel', 'Region'], axis=1, errors='ignore').round(2)
    # Multiply by 80 to make the numbers look like realistic INR annual spends (since original data is likely in euros/dollars)
    summary = summary * 80
    summary_inr = summary.map(lambda x: f"₹ {x:,.0f}")
    
    # Inject the business identity directly into the table rows
    summary_inr.index = [f"{p} ({persona_insights[p]['title']})" if p in persona_insights else p for p in summary_inr.index]
    
    # If new customers were predicted, add their exact inputs to the table for direct comparison
    for idx, cust in enumerate(st.session_state['new_customers']):
        new_cust_row = pd.DataFrame({
            'Fresh': [f"₹ {cust['fresh']:,.0f}"],
            'Milk': [f"₹ {cust['milk']:,.0f}"],
            'Grocery': [f"₹ {cust['grocery']:,.0f}"],
            'Frozen': [f"₹ {cust['frozen']:,.0f}"],
            'Detergents_Paper': [f"₹ {cust['detergents']:,.0f}"],
            'Delicassen': [f"₹ {cust['delicassen']:,.0f}"]
        }, index=[f"🎯 Customer {len(dataset) + idx + 1} ({persona_insights[cust['persona']]['title']})"])
        summary_inr = pd.concat([summary_inr, new_cust_row])
        
    st.dataframe(summary_inr, use_container_width=True)

st.success("Tip: You can also manually add new customer data in the sidebar to predict their persona!")
