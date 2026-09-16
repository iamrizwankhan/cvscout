import io
import re
import time
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import pdfplumber
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer


# ============================================================
# CvScout
# Data Science Resume Screening System
# ============================================================

APP_NAME = "CvScout"
APP_SUBTITLE = "Data Science Resume Screening System"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>

    .main {
        background-color: #050816;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: white;
    }

    .metric-card {
        background: #0d1220;
        border: 1px solid #20283b;
        border-radius: 12px;
        padding: 18px;
    }

    .score-good {
        color: #21c97a;
        font-weight: 700;
    }

    .score-review {
        color: #f2c94c;
        font-weight: 700;
    }

    .score-bad {
        color: #ff5c5c;
        font-weight: 700;
    }

    .small-text {
        color: #9aa4b2;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Skill dictionary
# ============================================================

SKILLS = [
    # Programming
    "Python",
    "R",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "PHP",
    "Scala",

    # Data Science
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "Scikit Learn",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Natural Language Processing",
    "Computer Vision",
    "Data Science",
    "Data Analysis",
    "Statistics",
    "Statistical Analysis",
    "Predictive Analytics",
    "Regression",
    "Classification",
    "Clustering",
    "Time Series",

    # BI
    "Power BI",
    "Tableau",
    "Looker",
    "Looker Studio",
    "Excel",
    "VBA",
    "DAX",

    # SQL / Databases
    "SQL",
    "MySQL",
    "PostgreSQL",
    "Oracle",
    "SQL Server",
    "MongoDB",
    "Snowflake",
    "Databricks",

    # Big Data
    "Spark",
    "PySpark",
    "Hadoop",
    "Hive",
    "Airflow",

    # Cloud
    "AWS",
    "Azure",
    "GCP",
    "Google Cloud",

    # DevOps
    "Docker",
    "Kubernetes",
    "Terraform",
    "Jenkins",
    "Git",
    "GitHub",
    "GitLab",
    "Linux",

    # Web
    "HTML",
    "CSS",
    "React",
    "Angular",
    "Node.js",
    "Django",
    "Flask",
    "FastAPI",
    "REST API",

    # Project / Management
    "Agile",
    "Scrum",
    "Project Management",
    "Stakeholder Management",
    "Business Analysis",
    "Business Intelligence",

    # Recruitment
    "Recruitment",
    "Talent Acquisition",
    "Candidate Management",
    "Sourcing",
]


# ============================================================
# Stopwords
# ============================================================

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "have",
    "has", "will", "are", "you", "your", "our", "their", "they",
    "them", "who", "what", "when", "where", "which", "into",
    "about", "over", "under", "using", "used", "use", "work",
    "working", "role", "roles", "job", "jobs", "candidate",
    "candidates", "experience", "years", "year", "required",
    "requirements", "requirement", "responsibilities",
    "responsibility", "skills", "skill", "ability", "strong",
    "good", "knowledge", "team", "teams", "join", "company",
    "position", "including", "such", "must", "should", "would",
    "can", "able", "also", "other", "more", "than", "through",
    "within", "across", "based", "provide", "support", "ensure",
    "develop", "development", "manage", "management"
}


# ============================================================
# Model
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


# ============================================================
# Text extraction
# ============================================================

def extract_text_from_pdf(uploaded_file) -> str:

    text_parts = []

    try:
        uploaded_file.seek(0)

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text_parts.append(page_text)

    except Exception as e:
        return f"ERROR_READING_PDF: {e}"

    return "\n".join(text_parts)


def extract_text_from_docx(uploaded_file) -> str:

    try:
        uploaded_file.seek(0)

        document = Document(uploaded_file)

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        # Also read tables
        for table in document.tables:

            for row in table.rows:

                for cell in row.cells:

                    if cell.text.strip():
                        paragraphs.append(cell.text)

        return "\n".join(paragraphs)

    except Exception as e:
        return f"ERROR_READING_DOCX: {e}"


def extract_text(uploaded_file) -> str:

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    if filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)

    return ""


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower()

    # Normalize common variants
    replacements = {
        "scikit learn": "scikit-learn",
        "machine-learning": "machine learning",
        "data-analytics": "data analytics",
        "powerbi": "power bi",
        "sqlserver": "sql server",
        "nodejs": "node.js",
        "google cloud platform": "google cloud",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Contact extraction
# ============================================================

def extract_email(text: str) -> str:

    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    match = re.search(pattern, text)

    return match.group(0) if match else ""


def extract_phone(text: str) -> str:

    patterns = [
        r"(?:\+44\s?7\d{3}|\+44\s?\d{4}|07\d{3})\s?\d{3}\s?\d{3}",
        r"\+?\d[\d\s().-]{8,}\d"
    ]

    for pattern in patterns:

        matches = re.findall(pattern, text)

        if matches:

            cleaned = re.sub(r"\s+", " ", matches[0]).strip()

            return cleaned

    return ""


def extract_linkedin(text: str) -> str:

    pattern = r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+"

    match = re.search(pattern, text, re.IGNORECASE)

    if match:

        return match.group(0)

    return ""


def extract_github(text: str) -> str:

    pattern = r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_%]+"

    match = re.search(pattern, text, re.IGNORECASE)

    if match:

        return match.group(0)

    return ""


# ============================================================
# Name extraction
# ============================================================

def extract_candidate_name(text: str, filename: str = "") -> str:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    ignored = {
        "curriculum vitae",
        "resume",
        "cv",
        "profile",
        "professional profile",
        "personal profile",
        "contact",
        "contact details"
    }

    # First few lines are usually the best place
    for line in lines[:20]:

        clean = re.sub(r"[^A-Za-z .'-]", "", line).strip()

        words = clean.split()

        if 2 <= len(words) <= 5:

            lower = clean.lower()

            if lower in ignored:
                continue

            if any(
                word.lower() in ignored
                for word in words
            ):
                continue

            # Avoid obvious headings
            if clean.upper() == clean and len(clean) > 25:
                continue

            return clean

    # Fallback to filename
    base = re.sub(
        r"\.(pdf|docx)$",
        "",
        filename,
        flags=re.IGNORECASE
    )

    base = re.sub(
        r"[\(\[].*?[\)\]]",
        "",
        base
    )

    base = re.sub(
        r"[-_]+",
        " ",
        base
    )

    base = re.sub(
        r"\b(totaljobs|cv|resume)\b",
        "",
        base,
        flags=re.IGNORECASE
    )

    base = re.sub(r"\s+", " ", base).strip()

    return base.title()


def extract_first_name(name: str) -> str:

    if not name:
        return ""

    return name.split()[0]


# ============================================================
# Experience extraction
# ============================================================

def extract_experience_years(text: str) -> float:

    values = []

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:
                value = float(match)

                if 0 <= value <= 50:
                    values.append(value)

            except:
                pass

    if not values:
        return 0.0

    return max(values)


def extract_required_experience(text: str) -> float:

    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant\s*)?experience",
        r"minimum\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
        r"at least\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:
                values.append(float(match))
            except:
                pass

    if not values:
        return 0.0

    return max(values)


def experience_score(
    candidate_years: float,
    required_years: float
) -> float:

    if required_years <= 0:
        return 70.0

    if candidate_years <= 0:
        return 20.0

    if candidate_years >= required_years:
        # Extra experience gets a modest bonus
        bonus = min(
            (candidate_years - required_years) * 5,
            20
        )

        return min(100, 80 + bonus)

    ratio = candidate_years / required_years

    return max(20, ratio * 80)


# ============================================================
# Location extraction
# ============================================================

UK_CITIES = [
    "London",
    "Birmingham",
    "Manchester",
    "Leeds",
    "Liverpool",
    "Bristol",
    "Sheffield",
    "Nottingham",
    "Leicester",
    "Coventry",
    "Reading",
    "Milton Keynes",
    "Cambridge",
    "Oxford",
    "Southampton",
    "Portsmouth",
    "Brighton",
    "Edinburgh",
    "Glasgow",
    "Cardiff",
    "Newcastle",
    "Sunderland",
    "Belfast",
    "Derby",
    "Bradford",
    "York",
    "Peterborough",
    "Norwich",
    "Exeter",
    "Bath",
    "Slough",
    "Watford",
]


def extract_location(text: str) -> str:

    for city in UK_CITIES:

        if re.search(
            rf"\b{re.escape(city)}\b",
            text,
            flags=re.IGNORECASE
        ):
            return city

    return ""


# ============================================================
# Job title / company
# ============================================================

def extract_job_title(text: str) -> str:

    title_patterns = [
        r"(?:current\s+)?job\s+title\s*[:\-]\s*(.+)",
        r"(?:current\s+)?position\s*[:\-]\s*(.+)",
        r"title\s*[:\-]\s*(.+)"
    ]

    for pattern in title_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            if 1 <= len(value) <= 100:
                return value

    # Scan lines for common professional titles
    title_keywords = [
        "manager",
        "engineer",
        "analyst",
        "developer",
        "consultant",
        "director",
        "administrator",
        "technician",
        "specialist",
        "coordinator",
        "recruiter",
        "scientist",
        "architect"
    ]

    for line in text.splitlines()[:80]:

        line_clean = line.strip()

        if not line_clean:
            continue

        lower = line_clean.lower()

        if any(k in lower for k in title_keywords):

            if 2 <= len(line_clean.split()) <= 10:

                return line_clean

    return ""


def extract_current_company(text: str) -> str:

    patterns = [
        r"current\s+company\s*[:\-]\s*(.+)",
        r"company\s*[:\-]\s*(.+)",
        r"employer\s*[:\-]\s*(.+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip()

            if 1 <= len(value) <= 100:
                return value

    return ""


# ============================================================
# Education
# ============================================================

def extract_education(text: str) -> str:

    education_terms = [
        "phd",
        "doctorate",
        "master of science",
        "msc",
        "master of arts",
        "ma",
        "master of business administration",
        "mba",
        "bachelor of science",
        "bsc",
        "bachelor of arts",
        "ba",
        "bachelor of engineering",
        "beng",
        "be",
        "degree",
        "diploma"
    ]

    found = []

    for line in text.splitlines():

        lower = line.lower()

        if any(term in lower for term in education_terms):

            line_clean = line.strip()

            if line_clean and line_clean not in found:

                found.append(line_clean)

    return " | ".join(found[:4])


def education_score(
    jd_text: str,
    cv_text: str
) -> float:

    jd = normalize_text(jd_text)
    cv = normalize_text(cv_text)

    levels = [
        (
            ["phd", "doctorate"],
            100
        ),
        (
            ["master of science", "msc", "master's", "master"],
            85
        ),
        (
            ["mba"],
            85
        ),
        (
            ["bachelor", "bsc", "bsc.", "ba", "beng", "be"],
            70
        )
    ]

    for terms, score in levels:

        jd_has = any(term in jd for term in terms)

        if jd_has:

            cv_has = any(term in cv for term in terms)

            if cv_has:
                return score

            # Higher education may still satisfy lower requirements
            if any(
                term in cv
                for term in [
                    "msc",
                    "master",
                    "phd",
                    "doctorate"
                ]
            ):
                return 90

            return 20

    return 70


# ============================================================
# Skill matching
# ============================================================

def find_skills(text: str) -> List[str]:

    normalized = normalize_text(text)

    found = []

    for skill in SKILLS:

        skill_norm = normalize_text(skill)

        if skill_norm in normalized:

            if skill not in found:
                found.append(skill)

    return found


def calculate_skill_score(
    jd_text: str,
    cv_text: str
) -> Tuple[float, List[str], List[str]]:

    jd_skills = find_skills(jd_text)
    cv_skills = find_skills(cv_text)

    if not jd_skills:

        return 70.0, [], []

    jd_set = set(jd_skills)
    cv_set = set(cv_skills)

    matched = sorted(jd_set.intersection(cv_set))

    missing = sorted(jd_set.difference(cv_set))

    score = (
        len(matched) /
        max(len(jd_set), 1)
    ) * 100

    return score, matched, missing


# ============================================================
# Keyword extraction
# ============================================================

def extract_keywords(
    text: str,
    max_words: int = 40
) -> List[str]:

    text = normalize_text(text)

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.\-]{2,}\b",
        text
    )

    words = [
        word.lower()
        for word in words
        if word.lower() not in STOPWORDS
    ]

    counts = Counter(words)

    return [
        word
        for word, _ in counts.most_common(max_words)
    ]


def keyword_coverage_score(
    jd_text: str,
    cv_text: str
) -> Tuple[float, List[str]]:

    jd_keywords = extract_keywords(
        jd_text,
        max_words=35
    )

    cv_keywords = set(
        extract_keywords(
            cv_text,
            max_words=120
        )
    )

    if not jd_keywords:
        return 70.0, []

    matched = [
        word
        for word in jd_keywords
        if word in cv_keywords
    ]

    coverage = (
        len(matched) /
        max(len(jd_keywords), 1)
    )

    # Calibration:
    # lexical overlap is usually naturally low.
    score = min(
        100,
        coverage * 170
    )

    return score, matched


# ============================================================
# TF-IDF
# ============================================================

def calculate_tfidf_similarity(
    jd_text: str,
    cv_text: str
) -> float:

    jd = normalize_text(jd_text)
    cv = normalize_text(cv_text)

    if not jd or not cv:
        return 0.0

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            [jd, cv]
        )

        raw = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        # Convert naturally small cosine values to useful
        # 0-100 screening scale.
        score = min(
            100,
            raw * 250
        )

        return round(score, 2)

    except Exception:
        return 0.0


# ============================================================
# Sentence Transformer semantic similarity
# ============================================================

def calculate_semantic_similarity(
    jd_text: str,
    cv_text: str,
    model
) -> float:

    jd = normalize_text(jd_text)
    cv = normalize_text(cv_text)

    if not jd or not cv:
        return 0.0

    try:

        embeddings = model.encode(
            [jd, cv],
            normalize_embeddings=True
        )

        raw = float(
            np.dot(
                embeddings[0],
                embeddings[1]
            )
        )

        # Sentence-transformer cosine similarities for
        # real-world JD/CV pairs are often around 0.2-0.6.
        # Calibrate that into a useful screening range.
        score = ((raw - 0.10) / 0.45) * 100

        score = float(
            np.clip(score, 0, 100)
        )

        return round(score, 2)

    except Exception:
        return 0.0


# ============================================================
# ATS score
# ============================================================

def calculate_ats_score(text: str) -> float:

    if not text:
        return 0.0

    score = 0

    # Length
    if len(text) > 1200:
        score += 20

    if len(text) > 2500:
        score += 10

    # Contact information
    if extract_email(text):
        score += 20

    if extract_phone(text):
        score += 10

    if extract_linkedin(text):
        score += 10

    # Career / technical information
    if extract_experience_years(text) > 0:
        score += 10

    if find_skills(text):
        score += 15

    if extract_education(text):
        score += 5

    return min(
        100,
        float(score)
    )


# ============================================================
# Final scoring engine
# ============================================================

def calculate_final_score(
    semantic_score: float,
    tfidf_score: float,
    skill_score: float,
    keyword_score: float,
    experience_score_value: float,
    education_score_value: float
) -> float:

    """
    Final CvScout scoring model.

    Semantic similarity      = 30%
    TF-IDF similarity        = 15%
    Skill match              = 25%
    Keyword coverage         = 15%
    Experience match         = 10%
    Education match          = 5%
    """

    final_score = (
        semantic_score * 0.30 +
        tfidf_score * 0.15 +
        skill_score * 0.25 +
        keyword_score * 0.15 +
        experience_score_value * 0.10 +
        education_score_value * 0.05
    )

    return round(
        float(np.clip(final_score, 0, 100)),
        1
    )


def get_decision(score: float) -> str:

    if score >= 60:
        return "Take"

    if score >= 40:
        return "Review"

    return "Reject"


# ============================================================
# Candidate processor
# ============================================================

def process_candidate(
    uploaded_file,
    jd_text: str,
    model
) -> Dict:

    filename = uploaded_file.name

    cv_text = extract_text(uploaded_file)

    if not cv_text.strip():
        return {
            "File Name": filename,
            "Full Name": "",
            "First Name": "",
            "Email": "",
            "Phone": "",
            "City": "",
            "Country": "UK",
            "LinkedIn Link": "",
            "Current Job Title": "",
            "Experience (Years)": 0,
            "Current Company": "",
            "Skills": "",
            "Education": "",
            "Certificates": "",
            "Location": "",
            "ATS Score": 0,
            "Github/Website/Portfolio": "",
            "Matched Skills": "",
            "Missing Skills": "",
            "TF-IDF Similarity": 0,
            "Semantic Similarity": 0,
            "Skill Match": 0,
            "Keyword Match": 0,
            "Experience Match": 0,
            "Education Match": 0,
            "Relevancy Score": 0,
            "Decision": "Reject",
            "Match Summary": "Could not extract resume text."
        }

    # --------------------------------------------------------
    # Extraction
    # --------------------------------------------------------

    name = extract_candidate_name(
        cv_text,
        filename
    )

    first_name = extract_first_name(name)

    email = extract_email(cv_text)

    phone = extract_phone(cv_text)

    linkedin = extract_linkedin(cv_text)

    github = extract_github(cv_text)

    experience = extract_experience_years(cv_text)

    current_company = extract_current_company(cv_text)

    job_title = extract_job_title(cv_text)

    location = extract_location(cv_text)

    education = extract_education(cv_text)

    skills = find_skills(cv_text)

    ats_score = calculate_ats_score(cv_text)

    # --------------------------------------------------------
    # Data Science scoring
    # --------------------------------------------------------

    tfidf_score = calculate_tfidf_similarity(
        jd_text,
        cv_text
    )

    semantic_score = calculate_semantic_similarity(
        jd_text,
        cv_text,
        model
    )

    skill_score, matched_skills, missing_skills = (
        calculate_skill_score(
            jd_text,
            cv_text
        )
    )

    keyword_score, matched_keywords = (
        keyword_coverage_score(
            jd_text,
            cv_text
        )
    )

    required_exp = extract_required_experience(
        jd_text
    )

    exp_score = experience_score(
        experience,
        required_exp
    )

    edu_score = education_score(
        jd_text,
        cv_text
    )

    # --------------------------------------------------------
    # Final score
    # --------------------------------------------------------

    relevancy = calculate_final_score(
        semantic_score,
        tfidf_score,
        skill_score,
        keyword_score,
        exp_score,
        edu_score
    )

    decision = get_decision(
        relevancy
    )

    # --------------------------------------------------------
    # Match explanation
    # --------------------------------------------------------

    if matched_skills:

        summary = (
            f"Matched {len(matched_skills)} required skill(s)"
        )

    else:

        summary = "Low direct skill overlap"

    if required_exp > 0:

        summary += (
            f" | Required: {required_exp:g} yrs"
            f" | Candidate: {experience:g} yrs"
        )

    return {
        "File Name": filename,
        "Full Name": name,
        "First Name": first_name,
        "Email": email,
        "Phone": phone,
        "City": location,
        "Country": "UK",
        "LinkedIn Link": linkedin,
        "Current Job Title": job_title,
        "Experience (Years)": experience,
        "Current Company": current_company,
        "Skills": ", ".join(skills),
        "Education": education,
        "Certificates": "",
        "Location": location,
        "ATS Score": ats_score,
        "Github/Website/Portfolio": github,
        "Matched Skills": ", ".join(matched_skills),
        "Missing Skills": ", ".join(missing_skills),
        "TF-IDF Similarity": tfidf_score,
        "Semantic Similarity": semantic_score,
        "Skill Match": round(skill_score, 1),
        "Keyword Match": round(keyword_score, 1),
        "Experience Match": round(exp_score, 1),
        "Education Match": round(edu_score, 1),
        "Relevancy Score": relevancy,
        "Decision": decision,
        "Match Summary": summary
    }


# ============================================================
# Export
# ============================================================

EXPORT_COLUMNS = [
    "Relevancy Score",
    "Decision",
    "File Name",
    "Full Name",
    "First Name",
    "Email",
    "Phone",
    "City",
    "Country",
    "LinkedIn Link",
    "Current Job Title",
    "Experience (Years)",
    "Current Company",
    "Skills",
    "Education",
    "Certificates",
    "Location",
    "ATS Score",
    "Github/Website/Portfolio",
    "Matched Skills",
    "Missing Skills",
    "TF-IDF Similarity",
    "Semantic Similarity",
    "Skill Match",
    "Keyword Match",
    "Experience Match",
    "Education Match",
    "Match Summary"
]


def create_excel(df: pd.DataFrame) -> bytes:

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="CvScout Results"
        )

    output.seek(0)

    return output.getvalue()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="margin-bottom:0;">CvScout</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Data Science Resume Screening"
    )

    st.markdown("---")

    st.markdown(
        """
        **Data Science Pipeline**

        • Resume Parsing

        • NLP Preprocessing

        • TF-IDF

        • Cosine Similarity

        • Sentence Transformers

        • Feature Engineering

        • Candidate Scoring
        """
    )

    st.markdown("---")

    st.caption(
        "Free/open-source machine learning and NLP pipeline"
    )


# ============================================================
# Main header
# ============================================================

st.title("CvScout")

st.caption(
    "Data Science Resume Screening System"
)


st.markdown(
    """
    Upload one Job Description and multiple CVs.
    CvScout extracts structured candidate information and
    calculates a data-driven relevance score using NLP,
    TF-IDF, semantic embeddings and feature engineering.
    """
)


# ============================================================
# Upload section
# ============================================================

col1, col2 = st.columns(2)

with col1:

    jd_file = st.file_uploader(
        "Upload Job Description",
        type=["pdf", "docx"],
        key="jd_upload"
    )


with col2:

    resume_files = st.file_uploader(
        "Upload Resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_upload"
    )


# ============================================================
# Job description
# ============================================================

jd_text = ""

if jd_file:

    jd_text = extract_text(
        jd_file
    )

    if jd_text.strip():

        st.success(
            "Job Description loaded successfully."
        )

        with st.expander(
            "Preview Job Description"
        ):

            st.text_area(
                "JD Text",
                jd_text[:10000],
                height=250,
                label_visibility="collapsed"
            )

    else:

        st.error(
            "Could not extract text from the Job Description."
        )


# ============================================================
# Resume upload confirmation
# ============================================================

if resume_files:

    st.success(
        f"{len(resume_files)} resume(s) uploaded."
    )


# ============================================================
# Analyze
# ============================================================

analyze = st.button(
    "🚀 Analyze Resumes",
    use_container_width=True,
    type="primary"
)


# ============================================================
# Analysis
# ============================================================

if analyze:

    if not jd_text.strip():

        st.error(
            "Please upload a valid Job Description first."
        )

        st.stop()

    if not resume_files:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    with st.spinner(
        "Loading Sentence Transformer model..."
    ):

        model = load_embedding_model()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    start_time = time.time()

    results = []

    progress = st.progress(
        0
    )

    status = st.empty()

    total = len(resume_files)

    for index, resume in enumerate(
        resume_files,
        start=1
    ):

        status.write(
            f"Processing {resume.name} "
            f"({index}/{total})..."
        )

        result = process_candidate(
            resume,
            jd_text,
            model
        )

        results.append(
            result
        )

        progress.progress(
            index / total
        )

    elapsed = round(
        time.time() - start_time,
        1
    )

    progress.empty()
    status.empty()

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        results
    )

    df = df[
        EXPORT_COLUMNS
    ]

    df = df.sort_values(
        by="Relevancy Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save in session
    # --------------------------------------------------------

    st.session_state["results"] = df
    st.session_state["analysis_time"] = elapsed


# ============================================================
# Results
# ============================================================

if "results" in st.session_state:

    df = st.session_state["results"]

    elapsed = st.session_state.get(
        "analysis_time",
        0
    )

    st.markdown("---")

    st.header(
        "Screening Results"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_candidates = len(df)

    take_count = int(
        (df["Decision"] == "Take").sum()
    )

    review_count = int(
        (df["Decision"] == "Review").sum()
    )

    reject_count = int(
        (df["Decision"] == "Reject").sum()
    )

    average_score = round(
        df["Relevancy Score"].mean(),
        1
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Candidates",
            total_candidates
        )

    with m2:

        st.metric(
            "Take",
            take_count
        )

    with m3:

        st.metric(
            "Review",
            review_count
        )

    with m4:

        st.metric(
            "Average Relevancy",
            f"{average_score}%"
        )

    st.caption(
        f"Processed in {elapsed} seconds"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    st.markdown("### Decision Distribution")

    dist_df = pd.DataFrame(
        {
            "Decision": [
                "Take",
                "Review",
                "Reject"
            ],
            "Candidates": [
                take_count,
                review_count,
                reject_count
            ]
        }
    )

    st.bar_chart(
        dist_df.set_index("Decision")
    )

    # --------------------------------------------------------
    # Main result table
    # --------------------------------------------------------

    st.markdown(
        "### Ranked Candidates"
    )

    display_columns = [
        "Relevancy Score",
        "Decision",
        "File Name",
        "Full Name",
        "First Name",
        "Email",
        "Phone",
        "City",
        "Current Job Title",
        "Experience (Years)",
        "Current Company",
        "Skills",
        "Matched Skills",
        "Missing Skills",
        "ATS Score",
        "TF-IDF Similarity",
        "Semantic Similarity",
        "Skill Match",
        "Keyword Match",
        "Experience Match",
        "Education Match"
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
        height=550
    )

    # --------------------------------------------------------
    # Candidate details
    # --------------------------------------------------------

    st.markdown(
        "### Candidate Details"
    )

    selected_file = st.selectbox(
        "Select candidate",
        df["File Name"].tolist()
    )

    selected = df[
        df["File Name"] == selected_file
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Relevancy",
            f"{selected['Relevancy Score']}%"
        )

    with c2:

        st.metric(
            "Semantic Similarity",
            f"{selected['Semantic Similarity']}%"
        )

    with c3:

        st.metric(
            "Skill Match",
            f"{selected['Skill Match']}%"
        )

    d1, d2 = st.columns(2)

    with d1:

        st.markdown(
            f"""
            **Candidate:** {selected['Full Name']}

            **Email:** {selected['Email']}

            **Phone:** {selected['Phone']}

            **Current Job:** {selected['Current Job Title']}

            **Company:** {selected['Current Company']}

            **Experience:** {selected['Experience (Years)']} years

            **Location:** {selected['Location']}
            """
        )

    with d2:

        st.markdown(
            f"""
            **Decision:** {selected['Decision']}

            **Matched Skills:** {selected['Matched Skills']}

            **Missing Skills:** {selected['Missing Skills']}

            **TF-IDF:** {selected['TF-IDF Similarity']}%

            **Keyword Match:** {selected['Keyword Match']}%

            **Experience Match:** {selected['Experience Match']}%

            **Education Match:** {selected['Education Match']}%
            """
        )

    # --------------------------------------------------------
    # Export
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "### Export Results"
    )

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")

    excel_data = create_excel(
        df
    )

    e1, e2 = st.columns(2)

    with e1:

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="cvscout_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with e2:

        st.download_button(
            "Download Excel",
            data=excel_data,
            file_name="cvscout_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )