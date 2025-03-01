import streamlit as st
import pdfplumber
import os
import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# Download necessary NLP models
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")

# --- Function to Extract Text from PDFs ---
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text if text else "No text found"

# --- Function to Preprocess Text ---
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase and tokenize
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# --- Function to Rank Resumes ---
def rank_resumes(job_description, resume_texts):
    texts = [preprocess_text(job_description)] + [preprocess_text(resume) for resume in resume_texts]
    
    # Debug: Print extracted and preprocessed texts
    print("\n🔹 Job Description Processed:", texts[0])
    for i, txt in enumerate(texts[1:]):
        print(f"🔹 Resume {i+1} Processed:", txt[:200])  # Print first 200 characters

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Debug: Print TF-IDF matrix shape
    print("\n TF-IDF Matrix Shape:", tfidf_matrix.shape)

    job_vector = tfidf_matrix[0]  # Job description vector
    resume_vectors = tfidf_matrix[1:]  # Resume vectors

    similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

    # Debug: Print similarity scores
    print("\nSimilarity Scores:", similarity_scores)

    return similarity_scores

# --- Streamlit UI ---
def main():
    st.title("📄 AI-Powered Resume Screening & Ranking System")
    st.write("🔍 Upload resumes and compare them with the job description.")

    # --- Job Description Input ---
    job_desc = st.text_area(" Paste Job Description Here")

    # --- Resume Upload Section ---
    uploaded_files = st.file_uploader(" Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"])
    
    if st.button("Process & Rank Resumes") and job_desc and uploaded_files:
        resume_texts = []
        file_names = []

        # --- Extract Text from Uploaded Resumes ---
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name
            
            text = extract_text_from_pdf(temp_path)
            if text.strip() == "":
                st.error(f" No text extracted from {uploaded_file.name}. Please check the PDF.")
                continue

            resume_texts.append(text)
            file_names.append(uploaded_file.name)
            
            os.remove(temp_path)  # Cleanup temp files

        if not resume_texts:
            st.error("No valid resumes were processed. Please check the files.")
            return

        # --- Rank Resumes ---
        scores = rank_resumes(job_desc, resume_texts)
        ranked_resumes = sorted(zip(file_names, scores), key=lambda x: x[1], reverse=True)

        # --- Display Results ---
        st.subheader(" Ranked Resumes")
        results_df = pd.DataFrame(ranked_resumes, columns=["Resume", "Match Score"])
        results_df["Match Score"] = results_df["Match Score"].apply(lambda x: round(x * 100, 2))
        st.dataframe(results_df)

if __name__ == "__main__":
    main()
