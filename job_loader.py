'''Fetch jobs from an job board API → embed them as vectors → store them in a vector store 
for fast similarity matching.'''

import requests
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Fetch Jobs via API
def fetch_jobs():
    res = requests.get("https://www.arbeitnow.com/api/job-board-api")
    data = res.json()
    jobs = data.get("data", []) # API returns 'data' as list of jobs
    return jobs


# Convert Jobs into Documents for Vector Store.
def load_jobs_as_documents(jobs): 
    documents = []

    for idx, job in enumerate(jobs):
        title = job.get("title", "")
        description = job.get("description", "")
        company = job.get("company_name", "")
        content = f"{title}\nCompany: {company}\n\n{description}"
        doc = Document(page_content=content, metadata={"id": idx, "title": title})
        documents.append(doc)

    return documents

# Converts documents to vectors and stores in FAISS.
def create_vectorstore_from_jobs():
    jobs = fetch_jobs()
    documents=load_jobs_as_documents(jobs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
