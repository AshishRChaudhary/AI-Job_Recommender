'''Fetch jobs from job board APIs (Arbeitnow + Remotive) → embed them as vectors → store them
in a vector store for fast similarity matching.'''

import html
import re

import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
# Remotive terms: link back to each job's Remotive URL, name Remotive as the source,
# and keep requests to a few per day (app.py caches each source for 6 hours).
REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Job descriptions come back as HTML, so strip the tags before embedding.
# Some Arbeitnow descriptions are HTML-escaped twice (&lt;p&gt;), so unescape those once first.
def clean_html(text):
    text = text or ""
    if len(re.findall(r"&lt;/?[a-zA-Z]", text)) > len(re.findall(r"</?[a-zA-Z]", text)):
        text = html.unescape(text)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()


def get_json(url):
    res = requests.get(url, timeout=20)
    res.raise_for_status()
    return res.json()


# Each source is normalized to the same fields; ids are prefixed so they stay unique across sources.
def fetch_arbeitnow_jobs():
    jobs = get_json(ARBEITNOW_API_URL).get("data", [])  # API returns 'data' as list of jobs
    return [
        {
            "id": f"arbeitnow:{job.get('slug') or idx}",
            "title": job.get("title") or "",
            "company": job.get("company_name") or "",
            "location": job.get("location") or "",
            "remote": bool(job.get("remote", False)),
            "url": job.get("url") or "",
            "tags": job.get("tags") or [],
            "description": job.get("description") or "",
            "source": "Arbeitnow",
        }
        for idx, job in enumerate(jobs)
    ]


def fetch_remotive_jobs():
    jobs = get_json(REMOTIVE_API_URL).get("jobs", [])
    return [
        {
            "id": f"remotive:{job.get('id') or idx}",
            "title": job.get("title") or "",
            "company": job.get("company_name") or "",
            "location": job.get("candidate_required_location") or "",
            "remote": True,  # Remotive only lists remote jobs
            "url": job.get("url") or "",
            "tags": job.get("tags") or [],
            "description": job.get("description") or "",
            "source": "Remotive",
        }
        for idx, job in enumerate(jobs)
    ]


JOB_SOURCES = {"Arbeitnow": fetch_arbeitnow_jobs, "Remotive": fetch_remotive_jobs}
SOURCE_SITES = {"Arbeitnow": "https://www.arbeitnow.com", "Remotive": "https://remotive.com"}


# Convert Jobs into Documents for Vector Store.
def load_jobs_as_documents(jobs):
    documents = []

    for job in jobs:
        tags = ", ".join(str(tag) for tag in job["tags"])
        description = clean_html(job["description"])
        content = f"{job['title']}\nCompany: {job['company']}\nLocation: {job['location']}\nTags: {tags}\n\n{description}"
        metadata = {key: job[key] for key in ("id", "title", "company", "location", "remote", "url", "source")}
        documents.append(Document(page_content=content, metadata=metadata))

    return documents


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# Converts jobs to vectors and stores them in FAISS.
def create_vectorstore(jobs, embeddings):
    return FAISS.from_documents(load_jobs_as_documents(jobs), embeddings)
