import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Retail Recommendation System",
    page_icon="🛒",
    layout="wide"
)

# Header
st.title("🛒 Retail Recommendation System")
st.markdown("### Product Recommendation Engine using Machine Learning")

# Load Dataset
data = pd.read_csv("retail_data.csv")

# Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers", data["CustomerID"].nunique())

with col2:
    st.metric("Products", data["Product"].nunique())

with col3:
    st.metric("Transactions", len(data))

st.divider()

# Create Customer Product Matrix
customer_product_matrix = pd.crosstab(
    data["CustomerID"],
    data["Product"]
)

# Similarity Matrix
similarity_matrix = cosine_similarity(
    customer_product_matrix.T
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

# Product Selection
selected_product = st.selectbox(
    "Select a Product",
    similarity_df.columns
)

# Recommendation Button
if st.button("Generate Recommendations"):

    recommendations = (
        similarity_df[selected_product]
        .sort_values(ascending=False)
        .drop(selected_product)
    )

    recommendations = recommendations[
        recommendations > 0
    ].head(5)

    st.subheader("Recommended Products")

    for product, score in recommendations.items():

        st.progress(float(score))

        st.success(
            f"{product} | Similarity Score: {score:.2f}"
        )

st.divider()

# Product Popularity
st.subheader("Most Popular Products")

product_counts = (
    data["Product"]
    .value_counts()
)

st.bar_chart(product_counts)

st.divider()

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(data)