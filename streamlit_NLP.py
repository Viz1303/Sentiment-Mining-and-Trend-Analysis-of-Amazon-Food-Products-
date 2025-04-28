# NLP for Retail: Sentiment Dashboard
# -----------------------------------
# Streamlit app showcasing sentiment trends, spikes, product filtering, topic modeling,
# keyword trend analysis, and alert notifications.

import streamlit as st
st.set_page_config(
    page_title="NLP Retail Sentiment Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Download VADER lexicon
nltk.download('vader_lexicon')

# Sidebar settings
st.sidebar.header("Dashboard Settings")
min_reviews = st.sidebar.slider("Min reviews per month", 0, 500, 100, step=50)
alert_threshold = st.sidebar.slider("Alert threshold (|Δ sentiment|)", 0.0, 1.0, 0.1, step=0.05)

# Keyword tracking
default_keywords = ["fresh","delicious","stale","delay","packaging","shipping"]
keywords = st.sidebar.multiselect("Select keywords to track", default_keywords, default=default_keywords)

@st.cache_data
# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

@st.cache_data
# Load and preprocess data
def load_data(path):
    df = pd.read_csv(path)
    df['reviewTime'] = pd.to_datetime(df['Time'], unit='s')
    df['clean_text'] = df['Text'].map(clean_text)
    sid = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['clean_text'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df_ts = df.set_index('reviewTime')
    return df, df_ts

# Load data
DATA_PATH = 'sample_reviews.csv'  # adjust path
df, df_ts = load_data(DATA_PATH)

# Product selection
top_products = df['ProductId'].value_counts().nlargest(50).index.tolist()
selected_product = st.sidebar.selectbox("Select Product", ['All Products'] + top_products)
if selected_product != 'All Products':
    mask = df_ts['ProductId'] == selected_product
    df_ts = df_ts[mask]
    df = df[mask]

# Compute monthly stats
stats = df_ts['sentiment_score'].resample('M').agg(['mean','count']).rename(columns={'mean':'sentiment','count':'reviews'})
filtered = stats[stats['reviews'] >= min_reviews].copy()
filtered['diff'] = filtered['sentiment'].diff()

# Alert panel data
alerts = filtered[filtered['diff'].abs() >= alert_threshold]

# Metrics
total_reviews = len(df)
avg_sentiment = df['sentiment_score'].mean()
alert_count = len(alerts)

# Layout
st.title("🛒 NLP Retail Sentiment Dashboard")
st.markdown("Explore sentiment trends, product insights, topic themes, keyword patterns, and alerts.")

# Show metrics
col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Reviews", f"{total_reviews}")
col_b.metric("Avg. Sentiment", f"{avg_sentiment:.3f}")
col_c.metric("Alert Months", f"{alert_count}")

# Sentiment trend
st.subheader("Average Sentiment Over Time")
st.line_chart(filtered['sentiment'])

# Keyword trends
if keywords:
    st.subheader("Keyword Frequency Over Time")
    # Calculate monthly keyword presence frequency with proper word-boundary matching
    kw_presence = {
        kw: df_ts['clean_text']
                  .str.contains(rf"\b{kw}\b", regex=True)
                  .resample('M')
                  .mean()
        for kw in keywords
    }
    kw_df = pd.DataFrame(kw_presence)
    # Align with filtered index (months with sufficient reviews)
    kw_df = kw_df.reindex(filtered.index)
    st.line_chart(kw_df)

# Alerts table
if not alerts.empty:
    st.subheader("Sentiment Alerts")
    st.dataframe(alerts[['sentiment','diff','reviews']])

# Topic modeling on selected month
# Month selection
all_shifts = pd.concat([filtered['diff'].nlargest(5), filtered['diff'].nsmallest(5)])
month_opts = [d.strftime('%Y-%m') for d in all_shifts.index]
selected_month = st.selectbox("Select Month for Drill-Down", month_opts)

# Filter sample for month
mask_month = df['reviewTime'].dt.to_period('M').astype(str) == selected_month
sample = df[mask_month]

st.subheader(f"Sample Reviews for {selected_month}")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top Positive Reviews**")
    st.write(sample.nlargest(5,'sentiment_score')[['reviewTime','ProductId','Score','clean_text']])
with col2:
    st.markdown("**Top Negative Reviews**")
    st.write(sample.nsmallest(5,'sentiment_score')[['reviewTime','ProductId','Score','clean_text']])

# Topic modeling
if st.checkbox("Show topic modeling for this month"):
    doc_texts = sample['clean_text'].tolist()
    vec = CountVectorizer(max_features=500, stop_words='english')
    dtm = vec.fit_transform(doc_texts)
    lda = LatentDirichletAllocation(n_components=3, random_state=0)
    lda.fit(dtm)
    st.subheader("Emerging Topics")
    terms = vec.get_feature_names_out()
    for i, comp in enumerate(lda.components_):
        terms_idx = np.argsort(comp)[-10:][::-1]
        st.markdown(f"**Topic #{i+1}:** " + ", ".join(terms[terms_idx]))
