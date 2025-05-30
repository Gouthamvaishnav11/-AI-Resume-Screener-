📄 AI Resume Screener (NLP-Based)
An intelligent resume screening tool built using Natural Language Processing (NLP) and Streamlit, designed to help recruiters and hiring managers efficiently rank candidate resumes based on a provided job description.

🚩 Problem Statement
Recruiters often receive hundreds of resumes for a single job posting. Manually reading and shortlisting candidates is:
Time-consuming ⌛
Inconsistent 😓
Prone to human bias ❌
To solve this, we need an automated, fair, and fast resume screening solution using AI.
✅ Solution
This project provides a user-friendly web app that:
Accepts multiple resumes (PDF or DOCX).
Accepts or inputs a job description (TXT or manual entry).
Extracts and preprocesses resume text.
Calculates semantic similarity between each resume and the job description.
Ranks candidates by match percentage.
Displays top candidates visually.
Exports results to a downloadable CSV file.

🧠 Techniques Used
Module	Technique/Library Used
Text Extraction	PyPDF2, docx2txt
Text Preprocessing	NLTK, Regex, Stopwords removal
Vectorization	TfidfVectorizer from scikit-learn
Similarity Metric	Cosine Similarity
Frontend	Streamlit
Visualization	Matplotlib, Pandas

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/ai-resume-screener.git
cd ai-resume-screener

3. Install Requirements
Make sure you have Python 3.7+ installed.
pip install -r requirements.txt
Or manually install:
pip install streamlit PyPDF2 docx2txt nltk scikit-learn pandas matplotlib
Then, download NLTK stopwords:
import nltk
nltk.download('stopwords')
3. Run the App
streamlit run app.py
Open the provided URL (typically http://localhost:8501) in your browser.
📦 Features
✅ Multi-resume upload (PDF & DOCX)
✅ Manual or file-based JD input
✅ Resume text extraction & NLP preprocessing
✅ Similarity scoring and ranking
✅ Interactive visualization of top candidates
✅ CSV export of results
