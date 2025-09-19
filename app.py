import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🔍 K-Means Explorer")

# Sidebar controls
st.sidebar.header("Clustering Options")
n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300, step=50)
n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
random_state = st.sidebar.number_input("Random seed", 0, 100, 42)

# Generate synthetic dataset
X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, 
                  cluster_std=1.0, random_state=random_state)

# Run KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
y_pred = kmeans.fit_predict(X)

# Convert to DataFrame for display
df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
df["Cluster"] = y_pred

# Plot results
fig, ax = plt.subplots()
scatter = ax.scatter(df["Feature 1"], df["Feature 2"], c=df["Cluster"], cmap="viridis", alpha=0.7)
centers = kmeans.cluster_centers_
ax.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X", label="Centers")
ax.set_title("K-Means Clustering Result")
ax.legend()

st.pyplot(fig)

# Show data
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head(20))

