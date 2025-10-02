import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="K-Means Clustering App", layout="wide")
st.title("K-Means explorer")
st.write("Upload your dataset (e.g., Vendor segmentation) and explore clusters.")
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    selected_features = st.sidebar.multiselect("Select features for clustering", numeric_columns, default=numeric_columns)

    if selected_features:
        X = df[selected_features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.sidebar.slider("Select number of clusters (K)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        st.subheader("Clustered Data")
        st.dataframe(df.head())

        st.subheader("Cluster Distribution")
        st.bar_chart(df['Cluster'].value_counts().sort_index())

        # --- Visualization ---
        st.subheader("Cluster Visualization")

        if len(selected_features) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=X[selected_features[0]],
                y=X[selected_features[1]],
                hue=df['Cluster'],
                palette="Set2",
                s=100
            )
            plt.xlabel(selected_features[0])
            plt.ylabel(selected_features[1])
            plt.title("2D Cluster Visualization")
            st.pyplot(fig)


        # Allow download
        st.subheader("Download Clustered Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="clustered_data.csv", mime="text/csv")
