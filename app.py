import streamlit as st
from job_loader import create_vectorstore_from_jobs
from resume_parser import extract_text_from_pdf
from recommender import get_job_recommendations

# Load job data and create vector store
vectorstore=create_vectorstore_from_jobs()

st.set_page_config(page_title="Job Recommender", layout="centered")
st.title("AI Job Recommender")
st.markdown("Upload your resume (PDF format) to get started.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:

    resume_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume parsed successfully!")
    
    if st.button("🔍 Recommend Jobs"):
        
        # Run similarity search and storing most relevant jobs
        with st.spinner("Finding top job matches..."):
            matches = get_job_recommendations(resume_text, vectorstore, k=3)

        st.subheader("🎯 Top Job Recommendations")
        st.markdown(matches)