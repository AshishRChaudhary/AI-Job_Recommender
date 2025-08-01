

# 💼 Job Recommender with Resume Parser - https://ai-jobrecommender-87yfnoyp3koenabvcnhs9e.streamlit.app/
---
This project is an end-to-end job recommender system that analyzes a user's resume and recommends the top 3 most relevant job roles using **vector embeddings, FAISS**, and **Gemini Flash (via LangChain RAG)**. It combines real-time job listings, resume parsing, and AI-powered matching logic—presented through an interactive **Streamlit interface**.

---

## 🔄 Functional Pipeline

Resume (PDF)
↓
Text Extraction (pdfplumber)
↓
Job Fetching (Arbeitnow API)
↓
Embedding + Vector Store (MiniLM + FAISS)
↓
Similarity Search
↓
RAG Chain with Gemini Flash
↓
Top 3 Job Recommendations

---

## 🧠 Tech Stack :

* **Framework**: LangChain
* **LLM**: Gemini Flash via LangChain
* **Embedding Model**: all-MiniLM-L6-v2 from Hugging Face
* **Vector DB**: FAISS for fast similarity search
* **PDF Parsing**: pdfplumber
* **Frontend**: Streamlit

---

````

## 🗂️ Project Structure

| File / Folder         | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `app.py`              | Main Streamlit app to run the interface                                 |
| `resume_parser.py`    | Extracts clean text from uploaded resume PDFs                           |
| `job_loader.py`       | Fetches job data, converts it to documents, embeds and stores in FAISS  |
| `recommender.py`      | Builds and runs the RAG-based recommendation pipeline                   |
| `.env`                | Stores your Gemini API key (Not included in this repository)            |
| `requirements.txt`    | Lists all required dependencies                                         |

````
---

## 🔐 API Key Setup

Create a `.env` file in your project root:

````
GOOGLE_API_KEY="your_gemini_flash_api_key"
````

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/job-recommender.git
cd job-recommender

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

---


## 🧪 This project was created to understand:

* Resume parsing and text extraction
* Vector similarity search using **FAISS**
* **Embeddings** with HuggingFace transformers
* **Retrieval-Augmented Generation** using LangChain
* Using Gemini Flash (free model) with LangChain
* Streamlit-based app development

---

## 🚀 Future Enhancements

- 🔍 **Improved Resume Parsing**: Support for more file types (e.g., .docx) and structured section extraction (skills, experience, education).
- 🧠 **LLM Upgrades**: Integrate more powerful or fine-tuned LLMs (e.g., Gemini Pro, GPT-4, or open-source models) for deeper resume-job matching.
- 📊 **ATS Scoring Integration**: Add an ATS (Applicant Tracking System) score for each job match to quantify alignment.
- 🧠 **Advanced Embeddings**: Upgrade to larger transformer models (e.g., BGE, InstructorXL) for deeper job-resume understanding.
- 🌐 **Multi-source Job Scraping**: Combine multiple job boards (e.g., Remotive, USAJobs) for broader job options.
- 📬 **Email or Download Option**: Allow users to receive recommended job list via email or download as PDF.
- 🛠️ **Deployment Pipeline**: Dockerize and deploy via cloud platforms (e.g., AWS, GCP, Railway) with CI/CD support.

