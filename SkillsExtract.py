import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Ensure NLTK data is available (run this once)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

# ---------- TEXT CLEANING HELPERS ----------

def keep_alpha_char(text):
    """Keep only alphabetic characters."""
    alpha_only_string = re.sub(r'[^a-zA-Z]', ' ', text)
    return re.sub(r'\s+', ' ', alpha_only_string)

def nltk_pos_tagger(nltk_tag):
    """Convert NLTK POS tags to WordNet POS tags."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    return None

def lemmatize_sentence(sentence):
    """Lemmatize text based on POS tags."""
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tagged = map(lambda x: (x[0], nltk_pos_tagger(x[1])), nltk_tagged)
    lemmatized_sentence = [
        lemmatizer.lemmatize(word, tag) if tag else word
        for word, tag in wordnet_tagged
    ]
    return " ".join(lemmatized_sentence)

def remove_stop_words(text):
    """Remove English stopwords."""
    words = nltk.word_tokenize(str(text))
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)

# ---------- RESUME PROCESSING ----------

def read_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def pre_process_resume(resume_text):
    """Clean and preprocess resume text."""
    resume_text = keep_alpha_char(resume_text)
    resume_text = lemmatize_sentence(resume_text)
    resume_text = remove_stop_words(resume_text)
    return resume_text.lower()

# ---------- JOB DATA PROCESSING ----------

def load_and_preprocess_job_data(csv_path):
    """Load and preprocess job data from CSV."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None, None, None

    job_df = pd.read_csv(csv_path)
    job_df.columns = job_df.columns.str.strip().str.lower()
    job_df.dropna(subset=['internship_title'], inplace=True)

    job_df['data'] = job_df['internship_title'].apply(keep_alpha_char)
    job_df['data'] = job_df['data'].apply(lemmatize_sentence)
    job_df['data'] = job_df['data'].apply(remove_stop_words)
    job_df['data'] = job_df['data'].str.lower()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(job_df['data'])

    return job_df, tfidf_vectorizer, tfidf_matrix

# ---------- RECOMMENDATION ENGINE ----------

def recommend_job(resume_text, tfidf_matrix, tfidf_vectorizer, df):
    """Find similar jobs using cosine similarity."""
    resume_text_vector = tfidf_vectorizer.transform([resume_text])
    cosine_similarities = cosine_similarity(resume_text_vector, tfidf_matrix)
    job_indices = cosine_similarities.argsort()[0][::-1]
    top_recommendations = [df.iloc[index] for index in job_indices]
    return pd.DataFrame(top_recommendations)

def get_lacking_skills(resume_text, job_text):
    """Return skills from the job description missing in the resume."""
    resume_skills = set(resume_text.lower().split())
    job_skills = set(job_text.lower().split())
    lacking_skills = list(job_skills - resume_skills)
    return lacking_skills

def recommend_from_resume(resume_path, csv_path, top_n=10):
    """Generate top job recommendations for a resume."""
    job_df, tfidf_vectorizer, tfidf_matrix = load_and_preprocess_job_data(csv_path)

    if job_df is None:
        return []

    resume_text = read_pdf(resume_path)
    if not resume_text:
        print("Resume text could not be extracted.")
        return []

    preprocessed_resume = pre_process_resume(resume_text)
    recommended_jobs_df = recommend_job(preprocessed_resume, tfidf_matrix, tfidf_vectorizer, job_df)

    final_recommendations = []
    for _, row in recommended_jobs_df.head(top_n).iterrows():
        job_description = row['internship_title']
        lacking_skills = get_lacking_skills(preprocessed_resume, job_description)

        final_recommendations.append({
            'company_name': row.get('company_name', 'Unknown'),
            'internship_title': row['internship_title'],
            'location': row.get('location', 'N/A'),
            'start_date': row.get('start_date', 'N/A'),
            'duration': row.get('duration', 'N/A'),
            'stipend': row.get('stipend', 'N/A'),
            'lacking_skills': lacking_skills
        })

    return final_recommendations

# ---------- SEARCH HELPERS ----------

def search_by_query(csv_path, search_query="", location="", category="", num_results=10):
    """Search jobs manually by title, company, or filters."""
    job_df, _, _ = load_and_preprocess_job_data(csv_path)
    if job_df is None:
        return []

    filtered_df = job_df.copy()

    if search_query:
        filtered_df = filtered_df[
            filtered_df['internship_title'].str.contains(search_query, case=False, na=False) |
            filtered_df['company_name'].str.contains(search_query, case=False, na=False)
        ]

    if location and location.lower() != 'all':
        filtered_df = filtered_df[filtered_df['location'].str.contains(location, case=False, na=False)]

    if category and category.lower() != 'all':
        filtered_df = filtered_df[filtered_df['internship_title'].str.contains(category, case=False, na=False)]

    return filtered_df.head(num_results).to_dict(orient="records")

def get_unique_locations(csv_path):
    """Return a list of unique job locations."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    return sorted(df['location'].dropna().unique().tolist()) if 'location' in df.columns else []

def get_unique_titles(csv_path):
    """Return a list of unique job titles."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    return sorted(df['internship_title'].dropna().unique().tolist()) if 'internship_title' in df.columns else []

# ---------- EXAMPLE USAGE ----------
if __name__ == "__main__":
    csv_path = r"C:\Users\Lenovo\Desktop\python\MiniProject\internship.csv"
    resume_path = r"C:\Users\Lenovo\Desktop\python\MiniProject\resume.pdf"

    recommendations = recommend_from_resume(resume_path, csv_path)
    for r in recommendations:
        print(f"{r['company_name']} - {r['internship_title']} ({r['location']})")
        print(f"Missing skills: {', '.join(r['lacking_skills'])}")
        print("-" * 80)
