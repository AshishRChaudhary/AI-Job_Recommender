

# 💼 Job Recommender with Resume Parser - https://ai-jobrecommender-87yfnoyp3koenabvcnhs9e.streamlit.app/
---
This project is an end-to-end job recommender system that analyzes a user's resume and recommends the top 3 most relevant job roles using **vector embeddings, FAISS**, and an LLM (**OpenAI GPT-6 Luna** or **Gemini Flash**, via LangChain RAG). It combines real-time job listings from **Arbeitnow** and **Remotive**, resume parsing, and AI-powered matching logic—presented through an interactive **Streamlit interface**.

---

## 🔄 Functional Pipeline

```
Resume (PDF)
↓
Text Extraction (pdfplumber)
↓
Resume split into chunks
↓
Job Fetching (Arbeitnow + Remotive APIs, each cached for 6 hours)
↓
Embedding + Vector Store (MiniLM + FAISS)
↓
Similarity Search (best match per job across resume chunks)
↓
LLM (OpenAI or Gemini) picks the top 3 by job number, with reasons and skill gaps
↓
App builds each result's title, link and source from the job data
```

---

## 🧠 Tech Stack :

* **Framework**: LangChain
* **LLM**: OpenAI `gpt-6-luna` (`langchain-openai`) or Gemini Flash (`langchain-google-genai`)
* **Job Data**: Arbeitnow and Remotive public job APIs
* **Embedding Model**: all-MiniLM-L6-v2 from Hugging Face
* **Vector DB**: FAISS for fast similarity search
* **PDF Parsing**: pdfplumber
* **Frontend**: Streamlit

---

## 🗂️ Project Structure

| File / Folder         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `app.py`              | Main Streamlit app to run the interface                                 |
| `resume_parser.py`    | Extracts clean text from uploaded resume PDFs                           |
| `job_loader.py`       | Fetches jobs from Arbeitnow + Remotive, converts them to documents, embeds and stores in FAISS |
| `recommender.py`      | Picks the LLM provider and runs the RAG-based recommendation pipeline   |
| `.env`                | Stores your OpenAI / Gemini API key (Not included in this repository)   |
| `requirements.txt`    | Lists all required dependencies                                         |
| `.gitignore`          | Keeps `.env` and other local files out of git                          |

---

## 🔐 API Key Setup

The app works with **OpenAI** or **Google Gemini**. You only need one key:

* `OPENAI_API_KEY` set → uses OpenAI (default model `gpt-6-luna`)
* only `GOOGLE_API_KEY` set → uses Gemini (default model `gemini-2.5-flash`)
* both set → uses OpenAI, unless `LLM_PROVIDER` says otherwise

**Local:** create a `.env` file in your project root:

````
OPENAI_API_KEY="your_openai_api_key"
# or
GOOGLE_API_KEY="your_gemini_api_key"

# Optional: force a provider ("openai" or "gemini")
# LLM_PROVIDER="openai"
# Optional: override the models
# OPENAI_MODEL="gpt-6-luna"
# GEMINI_MODEL="gemini-2.5-flash"
````

**Streamlit Community Cloud:** add the same keys under **App settings → Secrets** as top-level entries (Streamlit exposes them to the app as environment variables):

````toml
OPENAI_API_KEY = "your_openai_api_key"
# LLM_PROVIDER = "openai"
````

> Note: `gpt-6-luna` is a reasoning model, so the app does not send a `temperature` for OpenAI (custom values are rejected). If you set `OPENAI_MODEL` to another model, it uses that model's default temperature.

---

## 🌐 Job Sources

| Source | What it adds |
|--------|--------------|
| [Arbeitnow](https://www.arbeitnow.com) | Tech jobs, mostly in Europe |
| [Remotive](https://remotive.com) | Remote jobs, many open to US / worldwide candidates (listings are delayed 24 hours) |

* Both boards are fetched together; if one is down, the app keeps going with the other and shows a warning.
* Each board's results are cached for **6 hours**, because Remotive asks API users to fetch at most ~4 times a day. If a board fails, the app keeps going with the other one and retries the failed board after 30 minutes.
* Every job links back to its original listing, and the app names (and links) each board it loaded jobs from, as Remotive's terms require.

**Jobs data provided by [Arbeitnow](https://www.arbeitnow.com) and [Remotive](https://remotive.com).**

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/AshishRChaudhary/AI-Job_Recommender.git
cd AI-Job_Recommender

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚔️ Challenges Faced & Solutions

  | Challenge                | Solution                                                      |
  | ------------------------ | ------------------------------------------------------------- |
  | OpenAI API limit         | Switched to **Gemini Flash** (fast + free)                    |
  | LangChain module errors  | Used `langchain-community` for updated imports                |
  | Vector embedding failure | Used `all-MiniLM-L6-v2` from HuggingFace for job descriptions |
  | Resume parsing issues    | Used `pdfplumber` for reliable and clean text extraction      |
| LangChain 1.0 removed `langchain.chains` / `langchain.vectorstores` | Pinned dependency versions and moved to `langchain-community`, `langchain-huggingface` and a direct retrieval + prompt flow |
| Embedding model only reads ~256 tokens, so long resumes were cut off | Split the resume into chunks and rank jobs by their best-matching chunk |
| Arbeitnow is mostly European jobs | Added **Remotive** remote jobs as a second source, with per-source fallback |
| GPT-6 reasoning models reject `temperature` | Only send `temperature` to Gemini; OpenAI uses the model default |
| Job listings could prompt-inject the LLM into adding fake links or tracking images | The LLM returns only job numbers + reasons (structured output); titles, links and sources are built in code and all text is markdown-escaped |
| Default PyTorch install pulls ~3 GB of unused CUDA libraries on Linux | `requirements.txt` installs the CPU-only `torch` build on Linux |

---


## 🧪 This project was created to understand:

* Resume parsing and text extraction
* Vector similarity search using **FAISS**
* **Embeddings** with HuggingFace transformers
* **Retrieval-Augmented Generation** using LangChain
* Switching between OpenAI and Gemini with LangChain
* Streamlit-based app development

---

## 🚀 Future Enhancements

- 🔍 **Improved Resume Parsing**: Support for more file types (e.g., .docx) and structured section extraction (skills, experience, education).
- 🧠 **LLM Upgrades**: Integrate more powerful or fine-tuned LLMs (e.g., Gemini Pro, larger GPT-6 models, or open-source models) for deeper resume-job matching.
- 📊 **ATS Scoring Integration**: Add an ATS (Applicant Tracking System) score for each job match to quantify alignment.
- 🧠 **Advanced Embeddings**: Upgrade to larger transformer models (e.g., BGE, InstructorXL) for deeper job-resume understanding.
- 🌐 **More Job Sources**: Add more boards (e.g., USAJobs) on top of Arbeitnow and Remotive for broader job options.
- 📬 **Email or Download Option**: Allow users to receive recommended job list via email or download as PDF.
- 🛠️ **Deployment Pipeline**: Dockerize and deploy via cloud platforms (e.g., AWS, GCP, Railway) with CI/CD support.

