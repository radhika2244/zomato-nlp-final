import streamlit as st
import pandas as pd
import zipfile
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter

# 🛠 Page setup
st.set_page_config(
    page_title="Zomato NLP Recommender",
    page_icon="🍽️",
    layout="wide"
)

# 🎨 Basic style
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: tomato;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Page header
st.markdown("<h1>🍽️ Zomato Cuisine Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Powered by Machine Learning and NLP</p>", unsafe_allow_html=True)

# 📁 Read CSV from ZIP
try:
    with zipfile.ZipFile("zomato.zip") as z:
        with z.open("zomato.csv") as f:
            df = pd.read_csv(f)
except Exception as e:
    st.error(f"❌ Failed to read CSV file: {e}")
    st.stop()

# ✅ Check required columns
required_columns = ['cuisines', 'name', 'location']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    st.error(f"⚠️ The uploaded CSV is missing these required columns: {missing}")
    st.stop()

# 📋 Preview Data
st.subheader("🔍 Preview of Uploaded Data")
st.dataframe(df.head())

# 📦 Preprocessing
df = df.dropna(subset=['cuisines'])
df['cuisines'] = df['cuisines'].apply(lambda x: [i.strip() for i in x.split(',')])
mlb = MultiLabelBinarizer()
cuisine_matrix = mlb.fit_transform(df['cuisines'])

# 📊 Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')  # Fix crash for sklearn >= 1.4
df['Cluster'] = kmeans.fit_predict(cuisine_matrix)

# 🎯 Sidebar for filters
st.sidebar.header("🔍 Search & Filters")

# 📍 Location filter
location = st.sidebar.selectbox("📍 Filter by Location", ["All"] + sorted(df['location'].dropna().unique()))
filtered_df = df if location == "All" else df[df['location'] == location]

# 📌 Cluster filter
selected_cluster = st.sidebar.selectbox("🎯 Choose a Cuisine Cluster", sorted(filtered_df['Cluster'].unique()))

# 📊 Cluster distribution
with st.expander("📊 View Cuisine Cluster Distribution"):
    st.bar_chart(filtered_df['Cluster'].value_counts().sort_index())

# 📋 Top restaurants
st.markdown("#### ✅ Top Restaurants in Selected Cluster")
st.dataframe(filtered_df[filtered_df['Cluster'] == selected_cluster][['name', 'cuisines', 'location']].head(10))

# 🔎 Search
search_term = st.sidebar.text_input("Search by dish/cuisine:")
if search_term:
    matched = filtered_df[filtered_df['cuisines'].apply(lambda x: search_term.lower() in ' '.join(x).lower())]
    st.markdown(f"#### 🔍 Restaurants matching '{search_term}'")
    st.dataframe(matched[['name', 'cuisines', 'location']].head(10))

# 🧠 NLP Section
st.header("🧠 Natural Language Processing (NLP)")

# Add fake reviews if not present
if 'Review' not in df.columns:
    df['Review'] = df['name'].apply(lambda x: f"{x} has great food and friendly staff.")

# 🌥️ Word Cloud
with st.expander("🌥️ View Word Cloud of Reviews"):
    all_reviews = ' '.join(df['Review'].dropna().astype(str).head(1000))  # Limit to avoid memory crash
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

# 😊 Sentiment Analysis
with st.expander("😊 Sentiment Analysis of Reviews"):
    if 'rate' in df.columns:
        df['Sentiment'] = df['rate'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        st.dataframe(df[['name', 'rate', 'Sentiment']].sort_values(by='Sentiment', ascending=False).head(10))

        # 📤 Download sentiment results
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df[['name', 'rate', 'Sentiment']])
        st.download_button("📥 Download Sentiment Results", csv, "sentiment_analysis.csv", "text/csv")
    else:
        st.warning("⚠️ 'rate' column not found. Sentiment analysis skipped.")

# 📊 Top cuisines chart
with st.expander("🍱 View Top 10 Cuisines"):
    all_cuisines = [cuisine for sublist in df['cuisines'] for cuisine in sublist]
    top_cuisines = dict(Counter(all_cuisines).most_common(10))
    st.bar_chart(top_cuisines)

# ℹ️ About section
with st.expander("ℹ️ About This App"):
    st.markdown("""
    - 📌 Clustering is done using KMeans on multi-label cuisine data
    - 🧠 NLP is used for word cloud and sentiment analysis (TextBlob)
    - ✅ Built with Streamlit for fast web deployment
    - 🧪 Filter by location, cluster, or search any cuisine
    """)

# 📝 Footer
st.markdown("""
---
<p style='text-align:center; font-size: 14px;'>Made with ❤️ by Radhika | Zomato Recommender NLP Project</p>
""", unsafe_allow_html=True)
