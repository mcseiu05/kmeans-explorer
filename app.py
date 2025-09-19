import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("🔍 K-Means Explorer")

# Sidebar controls
st.sidebar.header("Clustering Options")

