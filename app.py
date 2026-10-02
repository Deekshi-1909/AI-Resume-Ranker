import streamlit as st
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a lightweight, powerful open-source embedding model from Hugging Face
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("🎯 AI-Powered Enterprise Resume Ranker")
st.subheader("Rank candidates based on semantic relevance, not just keywords.")

col1, col2 = st.columns(2)

with col1:
    st.write("### Step 1: Target Profile")
    job_description = st.text_area("Paste the Job Description (JD) here:", height=250)

with col2:
    st.write("### Step 2: Candidate Pool")
    uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF format only):", type=["pdf"], accept_multiple_files=True)

if st.button("⚡ Run Smart Ranking System"):
    if not job_description:
        st.error("Please paste a job description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume PDF.")
    else:
        with st.spinner("Analyzing candidate text matching algorithms..."):
            jd_embedding = model.encode([job_description])
            results = []
            
            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file)
                resume_embedding = model.encode([resume_text])
                similarity_score = cosine_similarity(jd_embedding, resume_embedding)
                match_percentage = round(float(similarity_score[0][0]) * 100, 2)
                
                results.append({
                    "Candidate Name": file.name.replace(".pdf", ""),
                    "Match Score": f"{match_percentage}%",
                    "Score_Raw": match_percentage
                })
            
            df = pd.DataFrame(results).sort_values(by="Score_Raw", ascending=False)
            df = df.drop(columns=["Score_Raw"]).reset_index(drop=True)
            
            st.success("Analysis complete!")
            st.write("### 📊 Ranked Candidate Leaderboard")
            st.dataframe(df, use_container_width=True)
