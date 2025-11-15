import streamlit as st
import pandas as pd
from collections import Counter
import requests
# from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# import hdbscan
# from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

#API_KEY = os.getenv('TMDB_API_KEY')
HEADER_TOKEN = os.getenv('TMDB_HEADER')
 
# Set page configuration
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== TMDB API Functions ====================
headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {HEADER_TOKEN}"
}

@st.cache_data
def search(query):
    url = f"https://api.themoviedb.org/3/search/multi?query={query}&language=en-US&page=1"
    response = requests.get(url, headers=headers).json()

    if len(response["results"]) == 0:
        return None

    movie = response["results"][0]
    movie_data = {
        "id": movie["id"],
        "name": movie.get("original_name", "") if movie.get("media_type") == "tv" else movie.get("original_title", ""),
        "media_type": movie.get("media_type", ""),
        "overview": movie.get("overview", ""),
        "poster_path": movie.get("poster_path", "")
    }
    return movie_data

@st.cache_data
def get_movie_reviews(movie_id):
    page = 1
    reviews = []
    detailed_reviews = []

    while True:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/reviews?language=en-US&page={page}"
        response = requests.get(url, headers=headers).json()
        results = response.get("results", [])

        if not results:
            break
        
        for r in results:
            author = r.get("author", "Unknown")
            content = r.get("content", "").strip()
            reviews.append(content)
            detailed_reviews.append([author, content])

        if page >= response.get("total_pages", 1):
            break
        
        page += 1

    return reviews, detailed_reviews

# ==================== NLP Models Loading ====================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model=r"G:\NLP_Project\finetuned-roberta"
    )

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(r"G:\NLP_Project\all-MiniLM-L6-v2")

# ==================== Analysis Functions ====================
def analyze_sentiments(reviews, sentiment_pipe):
    sentiments = []
    for review in reviews:
        try:
            result = sentiment_pipe(review[:512])[0]
            sentiments.append({"label": result["label"]})
        except:
            sentiments.append({"label": "ERROR"})
    return sentiments

def get_cluster_keywords(reviews, labels, top_n=2):
    cluster_keywords = {}
    for cluster in set(labels):
        if cluster == -1:
            continue
        cluster_texts = [reviews[i] for i in range(len(reviews)) if labels[i] == cluster]
        if len(cluster_texts) == 0:
            continue
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        X = vectorizer.fit_transform(cluster_texts)
        scores = np.array(X.sum(axis=0)).flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        cluster_keywords[cluster] = keywords
    return cluster_keywords

def split_text(text, max_length=300):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def summarize_text(text, summarizer_pipe, max_length=25, min_length=5):
    if not text.strip():
        return ""
    try:
        return summarizer_pipe(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    except:
        return "Unable to summarize"

def summarize_review(review, summarizer_pipe, max_chunk_length=300):
    sub_chunks = split_text(review, max_chunk_length)
    sub_summaries = [summarize_text(chunk, summarizer_pipe) for chunk in sub_chunks]
    
    if len(sub_summaries) > 1:
        combined_summary = " ".join(sub_summaries)
        return summarize_text(combined_summary, summarizer_pipe)
    else:
        return sub_summaries[0]

def summarize_reviews(reviews, summarizer_pipe, max_reviews=20, max_chunk_length=300):
    reviews_to_process = reviews[:max_reviews]
    individual_summaries = [summarize_review(r, summarizer_pipe, max_chunk_length) for r in reviews_to_process]
    
    combined_summary_text = " ".join(individual_summaries)
    final_summary = summarize_text(combined_summary_text, summarizer_pipe)
    
    return final_summary, individual_summaries

# ==================== Streamlit UI ====================
st.markdown('<p class="main-header">🎬 Movie Sentiment Analysis Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔍 Search Movie")
    movie_name = st.text_input("Enter Movie Name:", placeholder="e.g., Spider-Man")
    analyze_button = st.button("Analyze Reviews", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This tool analyzes movie reviews using NLP for sentiment analysis, topic classification, and summarization.")
    st.markdown("**Model:** RoBERTa fine-tuned on IMDb")

# Main content
if analyze_button and movie_name:
    with st.spinner(f"🔍 Searching for '{movie_name}'..."):
        data = search(movie_name)
        
        if not data:
            st.error("❌ Movie not found. Please try a different search term.")
            st.stop()
    
    with st.spinner("📥 Fetching reviews..."):
        reviews, detailed_reviews = get_movie_reviews(data["id"])
        
        if len(reviews) == 0:
            st.warning("⚠️ No reviews found for this movie.")
            st.stop()
    
    # Display movie information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Movie Information")
        if data.get('poster_path'):
            poster_url = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
            st.image(poster_url, use_container_width=True)
        else:
            st.info("No poster available")
    
    with col2:
        st.subheader(data.get('name', 'Unknown Title'))
        st.write(f"**Type:** {data.get('media_type', 'N/A').title()}")
        st.write(f"**Overview:**")
        st.write(data.get('overview', 'No overview available'))
    
    st.markdown("---")
    
    # Load models and analyze
    with st.spinner("🤖 Loading AI models..."):
        sentiment_pipe = load_sentiment_model()
        summarizer_pipe = load_summarizer()
        embedding_model = load_embedding_model()
    
    with st.spinner("🔬 Analyzing sentiments..."):
        sentiment_results = analyze_sentiments(reviews, sentiment_pipe)
    
    with st.spinner("🧠 Performing topic clustering..."):
        embeddings = embedding_model.encode(reviews, show_progress_bar=False)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
        cluster_labels = clusterer.fit_predict(embeddings)
        cluster_keywords = get_cluster_keywords(reviews, cluster_labels, top_n=3)
    
    with st.spinner("📝 Generating summary..."):
        final_summary, individual_summaries = summarize_reviews(reviews, summarizer_pipe)
    
    # Sentiment Analysis Results
    st.subheader("📊 Sentiment Analysis Results")
    
    # Count sentiments - Your model uses LABEL_0 (Negative) and LABEL_1 (Positive)
    sentiment_counts = Counter([item['label'] for item in sentiment_results])
    total_reviews = len(reviews)
    
    # According to the model: LABEL_0 = Negative, LABEL_1 = Positive
    positive_count = sentiment_counts.get('LABEL_1', 0)
    negative_count = sentiment_counts.get('LABEL_0', 0)
    error_count = sentiment_counts.get('ERROR', 0)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("✅ Positive Reviews", positive_count, 
                 f"{(positive_count/total_reviews*100):.1f}%")
    
    with col2:
        st.metric("❌ Negative Reviews", negative_count,
                 f"{(negative_count/total_reviews*100):.1f}%")
    
    with col3:
        st.metric("📝 Total Reviews", total_reviews)
    
    # Overall outcome
    if positive_count > negative_count:
        overall_sentiment = "Positive ✅"
        sentiment_color = "positive"
        sentiment_emoji = "😊"
    elif negative_count > positive_count:
        overall_sentiment = "Negative ❌"
        sentiment_color = "negative"
        sentiment_emoji = "😞"
    else:
        overall_sentiment = "Mixed ⚖️"
        sentiment_color = "neutral"
        sentiment_emoji = "😐"
    
    st.markdown("### Overall Sentiment Outcome")
    st.markdown(f'<p class="{sentiment_color}" style="font-size: 2rem; text-align: center;">{sentiment_emoji} {overall_sentiment}</p>', 
               unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Summary and Topics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Reviews Summary")
        st.info(final_summary)
    
    with col2:
        st.subheader("🏷️ Main Topics Discussed")
        if cluster_keywords:
            for cluster, keywords in cluster_keywords.items():
                st.write(f"**Topic {cluster + 1}:** {', '.join(keywords)}")
        else:
            st.write("No distinct topics identified (reviews are too similar)")
    
    st.markdown("---")
    
    # Reviews Table
    st.subheader("📋 All Reviews")
    
    # Create DataFrame
    label_mapping = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Positive',
        'ERROR': 'Error'
    }
    
    reviews_df = pd.DataFrame({
        'Author': [dr[0] for dr in detailed_reviews],
        'Review': reviews,
    })
    
    reviews_df.insert(0, '#', range(1, len(reviews_df) + 1))
    
    # Color coding function
    def color_sentiment(val):
        if val == 'Positive':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Negative':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #fff3cd; color: #856404'
    
    styled_df = reviews_df.style.applymap(
        color_sentiment, subset=['Sentiment']
    )
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download option
    csv = reviews_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Reviews as CSV",
        data=csv,
        file_name=f"{movie_name}_reviews_analysis.csv",
        mime="text/csv",
    )

elif not movie_name and analyze_button:
    st.warning("⚠️ Please enter a movie name to search.")

else:
    # Welcome screen
    st.markdown("""
        ### Welcome to Movie Sentiment Analysis Dashboard! 👋
        
        This application analyzes movie reviews using state-of-the-art NLP models:
        
        **🤖 AI Models Used:**
        - **Sentiment Analysis:** RoBERTa fine-tuned on IMDb dataset
        - **Summarization:** DistilBART-CNN
        - **Topic Clustering:** HDBSCAN with Sentence Transformers
        
        **✨ Features:**
        - 🎯 Automatic sentiment classification (Positive/Negative)
        - 📊 Visual sentiment distribution
        - 📝 AI-powered review summarization
        - 🏷️ Topic extraction and keyword identification
        - 📋 Comprehensive reviews table with author information
        - 📥 Export results to CSV
        
        **📖 How to use:**
        1. Enter a movie name in the sidebar
        2. Click "Analyze Reviews"
        3. Explore the insights!
        
        ---
        *Powered by Transformers, TMDB API, and Streamlit*
    """)
    