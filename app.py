import io
import logging
import re

import streamlit as st
from job_loader import JOB_SOURCES, SOURCE_SITES, create_vectorstore, get_embeddings
from resume_parser import extract_text_from_pdf
from recommender import MAX_RESUME_WORDS, PROVIDER_NAMES, get_job_recommendations, get_provider, limit_resume

st.set_page_config(page_title="Job Recommender", layout="centered")

SOURCE_TTL = 6 * 3600  # Remotive asks for at most ~4 requests a day
RETRY_TTL = 30 * 60    # wait this long before retrying a job board that failed


# Successful fetches are cached for 6 hours (exceptions are never cached here)
@st.cache_data(ttl=SOURCE_TTL, show_spinner=False)
def fetch_source(name):
    return JOB_SOURCES[name]()


# A failed fetch is remembered as empty for 30 minutes so every page load doesn't hit the API again
@st.cache_data(ttl=RETRY_TTL, show_spinner=False)
def fetch_source_or_empty(name):
    try:
        return fetch_source(name)
    except Exception:
        logging.exception("Could not fetch jobs from %s", name)
        return []


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embeddings()


# Rebuilt only when the set of jobs changes; _jobs is not hashed (job_ids is the cache key)
@st.cache_resource(max_entries=1, show_spinner="Indexing job listings...")
def build_vectorstore(job_ids, _jobs):
    return create_vectorstore(_jobs, load_embeddings())


# Escape markdown so text from job boards or the LLM can't add links, images or formatting
def md_escape(text):
    return re.sub(r"([\\`*_{}\[\]()<>#+\-.!|:~$])", r"\\\1", str(text))


def job_link(meta):
    title = md_escape(meta["title"] or "Untitled job")
    url = meta["url"]
    if url.startswith(("https://", "http://")):
        return f"[{title}]({url.replace(')', '%29').replace(' ', '%20')})"
    return title


# Job title, company, location and source (Remotive requires naming it and linking back)
def job_line(meta):
    source = meta["source"]
    parts = [job_link(meta), md_escape(meta["company"]), md_escape(meta["location"]),
             f"via [{source}]({SOURCE_SITES[source]})"]
    return " · ".join(part for part in parts if part)


st.title("AI Job Recommender")
st.markdown("Upload your resume (PDF format) to get started.")

try:
    llm_name = PROVIDER_NAMES[get_provider()]
except ValueError:
    llm_name = "the configured AI model"
st.caption(f"This app doesn't save your resume. Its text is sent to {llm_name} to generate recommendations.")

with st.spinner("Loading latest job listings..."):
    source_jobs = {name: fetch_source_or_empty(name) for name in JOB_SOURCES}
jobs = [job for name_jobs in source_jobs.values() for job in name_jobs]
if not jobs:
    st.error("Could not load job listings right now. Please try again later.")
    st.stop()

try:
    vectorstore = build_vectorstore(tuple(job["id"] for job in jobs), jobs)
except Exception:
    logging.exception("Could not build the job index")
    st.error("Could not prepare the job listings right now. Please try again later.")
    st.stop()

# Show which job boards loaded and credit them
loaded = {name: len(name_jobs) for name, name_jobs in source_jobs.items() if name_jobs}
credits = " and ".join(f"[{name}]({SOURCE_SITES[name]}) ({count})" for name, count in loaded.items())
st.caption(f"{len(jobs)} jobs from {credits}.")
missing = [name for name in source_jobs if name not in loaded]
if missing:
    st.warning(f"Could not load jobs from {', '.join(missing)} right now. Showing the jobs that did load.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:

    try:
        resume_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
    except Exception:
        st.error("Could not read this PDF. Please try a different file.")
        st.stop()

    if not resume_text:
        st.warning("No text found in this PDF. If it is a scanned image, please upload a text-based PDF.")
        st.stop()

    st.success("✅ Resume parsed successfully!")
    if limit_resume(resume_text)[1]:
        st.info(f"Your resume is long, so only the first {MAX_RESUME_WORDS} words are used.")

    if st.button("🔍 Recommend Jobs"):

        # Run similarity search and ask the LLM to pick the best matches
        with st.spinner("Finding top job matches..."):
            try:
                picks, note, candidates = get_job_recommendations(resume_text, vectorstore, k=3)
            except ValueError as e:  # missing or invalid API key settings
                st.error(str(e))
                st.stop()
            except Exception:
                logging.exception("Could not generate recommendations")
                st.error("Could not generate recommendations right now. Please try again in a minute.")
                st.stop()

        st.subheader("🎯 Top Job Recommendations")
        if note:
            st.info(note)
        for pick in picks:
            st.markdown(f"#### {job_line(pick['doc'].metadata)}")
            st.markdown(md_escape(pick["why"]))
            if pick["gaps"]:
                st.markdown(f"**Gaps to address:** {md_escape(pick['gaps'])}")

        with st.expander("Other jobs considered"):
            for doc in candidates:
                st.markdown(f"- {job_line(doc.metadata)}")
