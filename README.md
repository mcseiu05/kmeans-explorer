# 📊 K Means Explorer App

A senior audit manager requested a way to segment their vendors based on purchase data to better understand patterns, detect irregularities, and support decision-making. To address this, I built an interactive app using **Streamlit** and **K-Means clustering**.

This app allows users to upload their own dataset (e.g., vendor transaction records), choose relevant features, and automatically cluster vendors into groups for analysis. The results are shown with visualizations and can be exported for further use.

---

## ✨ Features
- 📂 Upload vendor data (CSV/Excel)  
- 🔎 Select numeric features for clustering  
- ⚖️ Automatic feature scaling with `StandardScaler`  
- 🔢 Choose number of clusters (K) interactively  
- 📊 View cluster distribution and 2D visualization  
- 💾 Export clustered dataset as CSV  

---

## 🚀 Usage

Clone the repository and run the Streamlit app locally:

```bash
git clone https://github.com/mcseiu05/kmeans-explorer.git
cd kmeans-explorer
pip install -r requirements.txt
streamlit run app.py
```
Or access the deployed app directly here: https://mcseiu05-kmeans-explorer.streamlit.app/

## 📌 Use Cases

-🛒 Vendor Segmentation – group suppliers by pricing, volume, or other attributes
-📊 Procurement Analytics – identify vendor performance clusters
-🧾 Audit Support – detect unusual vendor groups or outliers
-⚙️ Business Intelligence – improve vendor management strategy
