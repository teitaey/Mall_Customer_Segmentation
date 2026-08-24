# 🛒 Wholesale Customer Segmentation Dashboard

**🌍 Live Demo:** [https://mallcustomersegmentation-teitaey.streamlit.app](https://mallcustomersegmentation-teitaey.streamlit.app)

An interactive Machine Learning web application that automatically groups wholesale customers into distinct "Shopping Personas" based on their annual purchasing habits. 

Built entirely with Python, this project uses **Unsupervised Machine Learning (K-Means Clustering)** and **Dimensionality Reduction (PCA)** to transform raw, multi-dimensional sales data into an easy-to-understand, interactive business dashboard.

## 🚀 Features

- **Dynamic K-Means Clustering:** Use a slider to instantly command the ML model to re-cluster the dataset into anywhere from 2 to 8 distinct Customer Personas on the fly.
- **PCA Visualization:** Complex 6-dimensional spending data (Fresh, Milk, Grocery, Frozen, Detergents, Delicatessen) is mathematically reduced to a stunning 2D scatter plot, showing exactly *who* shops similarly.
- **Financial Breakdown:** Automatically calculates and displays the average annual spending (in ₹) for each distinct Persona in a clear, formatted table alongside a headcount bar chart.
- **Predict New Customers:** Includes a manual data entry sidebar. Type in the spending habits of a brand new customer, and the algorithm will instantly predict which Persona they belong to, plotting them dynamically as a new 'X' on the graph!
- **Session Memory:** Built-in state management allows you to add, track, and selectively delete multiple custom predictions at once without reloading the page.

## 🛠️ Technologies Used

- **Python 3**
- **Streamlit** (for the interactive frontend UI)
- **Scikit-Learn** (for the `StandardScaler`, `PCA`, and `KMeans` models)
- **Pandas** (for data manipulation)
- **NumPy** (for mathematical operations)
- **Matplotlib** (for rendering the 2D cluster scatter plot)

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/wholesale-customer-segmentation.git
   cd wholesale-customer-segmentation
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install pandas numpy scikit-learn matplotlib streamlit
   ```

3. **Run the Streamlit App:**
   ```bash
   python3 -m streamlit run dashboard.py
   ```

4. **View the Dashboard:**
   Open your web browser and go to `http://localhost:8501`.

## 📂 Dataset
This project uses a modified version of the **Wholesale Customers Dataset** from the UCI Machine Learning Repository. It contains the annual spending of 440 clients of a wholesale distributor across 6 different product categories.
