from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Gemini Chat Model
def get_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env")

    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  # Gemini Flash behavior is routed through this
        temperature=0.5,
        google_api_key=google_api_key,
    )

# RAG pipeline
def rag_chain(vectorstore, k):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# Returns job recommendations based on resume and vectorstore
def get_job_recommendations(resume_text, vectorstore, k):
    chain = rag_chain(vectorstore, k)

    query = f"""
    Based on this resume:
    {resume_text}

    Suggest top {k} job roles from the job dataset that match this candidate well.
    Focus on matching skills, technologies, and responsibilities.
    Output only relevant job titles and summaries.
    """

    result = chain.invoke(query)
    answer = result['result']
    return answer
