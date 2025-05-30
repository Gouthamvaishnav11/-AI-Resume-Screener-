import streamlit as st
import PyPDF2
import docx2txt
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# --- Text Extraction Functions ---
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(uploaded_file):
    return docx2txt.process(uploaded_file)

# --- Text Preprocessing ---
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# --- TF-IDF & Cosine Similarity ---
def compute_similarity(resumes_text, job_desc_text):
    # Combine all texts
    all_texts = resumes_text + [job_desc_text]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Compute Cosine Similarity
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return similarity_scores[0] * 100  # Convert to percentage

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    
    # --- Custom CSS Styling ---
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button { background-color: #4CAF50; color: white; }
        .stButton>button:hover { background-color: #45a049; }
        .stDataFrame { background-color: white; border-radius: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stAlert { background-color: #e7f3fe; border-left: 6px solid #2196F3; }
        h1 { color: #2c3e50; }
        h2 { color: #3498db; }
    </style>
    """, unsafe_allow_html=True)
    
    # --- Home Page ---
    st.title("📄 AI Resume Screener (NLP-Based)")
    st.write("""
    This tool automates resume screening using **Natural Language Processing (NLP)**.
    - Upload resumes (PDF/DOCX) and a job description.
    - Get ranked candidates based on similarity scores.
    """)
    
    if st.button("Start Resume Screening"):
        st.session_state.page = "upload"
        st.rerun()
    
    # --- Upload Page ---
    if "page" in st.session_state and st.session_state.page == "upload":
        st.title("📤 Upload Resumes & Job Description")
        
        # Upload Resumes
        st.subheader("Upload Resumes (PDF/DOCX)")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx"], accept_multiple_files=True)
        
        # Upload Job Description
        st.subheader("Upload Job Description (TXT) or Enter Manually")
        job_desc_option = st.radio("Select Option:", ["Upload File", "Enter Manually"])
        
        if job_desc_option == "Upload File":
            job_desc_file = st.file_uploader("Upload Job Description (TXT)", type=["txt"])
            if job_desc_file:
                job_desc_text = job_desc_file.read().decode("utf-8")
        else:
            job_desc_text = st.text_area("Enter Job Description Here", height=200)
        
        # Process Button
        if st.button("Process"):
            if not uploaded_files or not job_desc_text:
                st.error("Please upload resumes and job description!")
            else:
                st.session_state.uploaded_files = uploaded_files
                st.session_state.job_desc_text = job_desc_text
                st.session_state.page = "results"
                st.rerun()
    
    # --- Results Page ---
    if "page" in st.session_state and st.session_state.page == "results":
        st.title("📊 Resume Screening Results")
        
        # Extract and Preprocess Resumes
        resumes_text = []
        candidate_names = []
        
        for file in st.session_state.uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_docx(file)
            
            # Extract candidate name (first line as a simple heuristic)
            lines = text.split("\n")
            candidate_name = lines[0].strip() if lines else "Unknown Candidate"
            candidate_names.append(candidate_name)
            
            # Preprocess text
            preprocessed_text = preprocess_text(text)
            resumes_text.append(preprocessed_text)
        
        # Preprocess Job Description
        preprocessed_job_desc = preprocess_text(st.session_state.job_desc_text)
        
        # Compute Similarity
        similarity_scores = compute_similarity(resumes_text, preprocessed_job_desc)
        
        # Create Results DataFrame
        results_df = pd.DataFrame({
            "Candidate Name": candidate_names,
            "Match Score (%)": [round(score, 2) for score in similarity_scores]
        })
        
        # Sort by Score
        results_df = results_df.sort_values(by="Match Score (%)", ascending=False)
        
        # Display Results
        st.subheader("Ranked Candidates")
        st.dataframe(results_df.style.highlight_max(subset=["Match Score (%)"], color='lightgreen'))
        
        # Horizontal Bar Chart with Percentage Labels
        st.subheader("Top  Matching Resumes")
        top_5 = results_df.head(5)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Horizontal bar chart
        bars = ax.barh(top_5["Candidate Name"], top_5["Match Score (%)"], color='skyblue')
        
        # Formatting
        ax.set_xlabel("Match Score (%)", fontsize=12)
        ax.set_title("Top 5 Candidates by Match Score", fontsize=14)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        
        # Add percentage labels at the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f"{width:.1f}%", 
                    ha='left', va='center', fontsize=10)
        
        st.pyplot(fig)
        
        # Download Results
        st.subheader("Download Results")
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="resume_screening_results.csv">Download CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Back Button
        if st.button("Back to Upload"):
            st.session_state.page = "upload"
            st.rerun()

if __name__ == "__main__":
    main()