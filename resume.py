import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import base64
import io
from textblob import Word
from textblob import TextBlob
import spacy

# Download NLTK stopwords (for removing common words)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Download spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.write("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from Word document
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text
    return text

# Function to process the text and analyze skills
def analyze_resume(text):
    words = re.findall(r'\b\w+\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]

    # Example skills list to look for
    skills_list = [
        "python", "java", "c++", "html", "css", "javascript", "sql", "machine learning",
        "data science", "tensorflow", "excel", "operations", "team leadership", "management",
        "communication", "project planning", "problem solving", "design", "user experience",
        "front-end", "visual design", "user interface", "product management", "strategy",
        "business analysis", "leadership", "software development", "technical expertise",
        "data analysis", "data visualization", "statistics", "analytics", "research",
        "artificial intelligence", "big data", "cloud technologies", "data pipelines"
    ]
    skill_counts = {skill: filtered_words.count(skill) for skill in skills_list if skill in filtered_words}

    # Rank skills by count
    ranked_skills = {key: value for key, value in sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)}

    # Extract years of experience (very basic, assuming a format like "5 years")
    experience = re.findall(r'(\d+)\s*(year|yrs|years)', text.lower())
    total_experience = sum(int(exp[0]) for exp in experience)

    return ranked_skills, total_experience

# Function to extract education details (very basic pattern recognition)
def extract_education_details(text):
    education_keywords = ["bachelor", "master", "degree", "university", "college", "phd", "engineering", "science"]
    education_details = []
    for line in text.splitlines():
        if any(keyword in line.lower() for keyword in education_keywords):
            education_details.append(line.strip())
    return education_details

# Function to determine suitable job positions based on skills
def suggest_job_positions(skills):
    job_positions = {
        "Data Scientist": ["python", "machine learning", "data science", "tensorflow", "analytics", "statistics"],
        "Machine Learning Engineer": ["python", "machine learning", "tensorflow", "data science", "algorithm design"],
        "Software Developer": ["python", "java", "c++", "html", "css", "javascript", "software development", "mongodb", "php"],
        "Web Developer": ["html", "css", "javascript", "python", "web development", "frontend", "backend"],
        "AI Researcher": ["python", "machine learning", "data science", "tensorflow", "research", "artificial intelligence"],
        "Data Analyst": ["python", "data science", "sql", "machine learning", "data visualization", "statistics", "excel", "powerbi", "tableau"],
        "Project Manager": ["management", "team leadership", "communication", "project planning", "problem solving"],
        "Product Manager": ["product management", "leadership", "strategy", "communication", "project management"],
        "Business Analyst": ["business analysis", "data analysis", "communication", "problem solving", "project management"],
        "Technical Lead": ["leadership", "project management", "software development", "communication", "technical expertise"],
        "Data Engineer": ["python", "data science", "sql", "big data", "cloud technologies", "data pipelines"],
        "UX/UI Designer": ["design", "user experience", "front-end", "visual design", "user interface"],
        "Operations Manager": ["operations", "management", "team leadership", "communication", "problem solving", "research"],
        "Database Manager": ["sql", "database management", "data analysis", "communication", "problem solving", "mongodb"],
        "Entrepreneur": ["leadership", "strategy", "communication", "problem solving", "innovation", "finance", "business"]
    }

    suitable_jobs = []
    threshold = 0.5  # Adjust the threshold as needed
    for job, required_skills in job_positions.items():
        matching_skills = [skill for skill in required_skills if skill in skills]
        if len(matching_skills) / len(required_skills) >= threshold:
            suitable_jobs.append(job)

    return suitable_jobs

# Function to display the resume (for PDF or DOCX files)
def display_resume(uploaded_file):
    if uploaded_file.type == "application/pdf":
        # Display PDF
        pdf_data = uploaded_file.read()
        st.download_button("Download Resume", data=pdf_data, file_name="resume.pdf", mime="application/pdf")
        st.markdown(f'<embed src="data:application/pdf;base64,{base64.b64encode(pdf_data).decode()}" width="100%" height="600px" type="application/pdf">', unsafe_allow_html=True)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Display DOCX content
        docx_data = uploaded_file.read()
        doc = Document(io.BytesIO(docx_data))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        st.text_area("Uploaded Resume", text, height=300)

# Function to extract personal information using spaCy
def extract_personal_info(text):
    personal_info = {}
    doc = nlp(text)

    # Extract name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            personal_info['Name'] = ent.text.strip()
            break

    # Extract email
    email_pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    email_match = re.search(email_pattern, text)
    if email_match:
        personal_info['Email'] = email_match.group(1).strip()

    # Extract phone number
    phone_pattern = r'(\+?\d[\d\s-]{7,}\d)'
    phone_match = re.search(phone_pattern, text)
    if phone_match:
        personal_info['Phone'] = phone_match.group(1).strip()

    return personal_info

def perform_swot_analysis(text):
    swot_analysis = {
        "Strengths": [],
        "Weaknesses": [],
        "Opportunities": [],
        "Threats": []
    }

    # Keywords for SWOT analysis
    strengths_keywords = ["experience", "expertise", "skills", "achievements", "strengths", "leadership", "knowledge"]
    weaknesses_keywords = ["weaknesses", "improve", "develop", "lack", "limited"]
    opportunities_keywords = ["opportunities", "growth", "potential", "future", "expand"]
    threats_keywords = ["challenges", "threats", "competition", "risk", "obstacles"]

    # Extract sentences containing SWOT keywords
    for sentence in text.split("."):
        sentence = sentence.strip()
        if any(keyword in sentence.lower() for keyword in strengths_keywords):
            swot_analysis["Strengths"].append(sentence)
        if any(keyword in sentence.lower() for keyword in weaknesses_keywords):
            swot_analysis["Weaknesses"].append(sentence)
        if any(keyword in sentence.lower() for keyword in opportunities_keywords):
            swot_analysis["Opportunities"].append(sentence)
        if any(keyword in sentence.lower() for keyword in threats_keywords):
            swot_analysis["Threats"].append(sentence)

    return swot_analysis


# Streamlit UI
st.title("AI-Based Resume Analyzer")
st.write("Upload a resume (PDF or DOCX) to analyze the candidate's skills and experience.")

# Sidebar for user selection
user_type = st.sidebar.selectbox("Select User Type", ["Personal Resume Analyzer", "Business Resume Analyzer"])

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file:
    # Display the uploaded resume
    display_resume(uploaded_file)

    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(uploaded_file)

    # Analyze the resume
    skills, experience = analyze_resume(resume_text)

    # Display skills with visualization (Bar Chart)
    st.subheader("Skills")
    skill_names = list(skills.keys())
    skill_counts = list(skills.values())

    # # Display skills graph
    # plt.figure(figsize=(10, 6))
    # plt.barh(skill_names, skill_counts, color='skyblue')
    # plt.xlabel("Skill Count")
    # plt.title("Skills Found in the Resume")
    # st.pyplot(plt)

    # Display total experience
    st.subheader("Total Experience")
    st.write(f"{experience} years of experience")

    # Display job suggestions
    st.subheader("Suggested Job Positions Based on Skills")
    job_positions = suggest_job_positions(skills)
    if job_positions:
        for job in job_positions:
            st.write(f"- {job}")
    else:
        st.write("No specific job suggestions found based on the resume's skills.")

    # Extract and display educational details
    st.subheader("Educational Details")
    education_details = extract_education_details(resume_text)
    if education_details:
        for edu in education_details:
            st.write(f"- {edu}")
    else:
        st.write("No educational details found.")

    # Display personal information
    st.subheader("Personal Information")
    personal_info = extract_personal_info(resume_text)
    if personal_info:
        for key, value in personal_info.items():
            st.write(f"{key}: {value}")
    else:
        st.write("No personal information found.")

    # Display sentiment analysis
    st.subheader("Sentiment Analysis")
    analysis = TextBlob(resume_text)
    sentiment = analysis.sentiment
    st.write(f"Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")

    # Display grammar and spell check
    st.subheader("Grammar and Spell Check")
    words = resume_text.split()
    corrected_words = [str(Word(word).correct()) for word in words]
    corrected_text = ' '.join(corrected_words)
    st.text_area("Corrected Resume", corrected_text, height=300)

    # Display personality traits analysis (basic example)
    st.subheader("Personality Traits Analysis")
    personality_traits = {
        "Leadership": ["leadership", "management", "team"],
        "Creativity": ["creative", "innovative", "design"],
        "Analytical": ["analytical", "data", "analysis"],
        "Communication": ["communication", "presentation", "writing"]
    }
    traits_found = {trait: any(keyword in resume_text.lower() for keyword in keywords) for trait, keywords in personality_traits.items()}
    for trait, found in traits_found.items():
        st.write(f"{trait}: {'Found' if found else 'Not Found'}")

    st.subheader("SWOT Analysis")
    swot_analysis = perform_swot_analysis(resume_text)
    for category, items in swot_analysis.items():
        st.write(f"**{category}:**")
        if items:
            for item in items:
                st.write(f"- {item}")
        else:
            st.write("No information found.")

    # Visualization Dashboard
    st.subheader("Visualization Dashboard")
    st.write("This section provides a visual overview of the resume analysis.")

    # Skill Distribution Pie Chart
    st.subheader("Skill Distribution")
    plt.figure(figsize=(8, 8))
    plt.pie(skill_counts, labels=skill_names, autopct='%1.1f%%', startangle=140)
    plt.title("Skill Distribution")
    st.pyplot(plt)

    # Experience Timeline (basic example)
    st.subheader("Experience Timeline")
    experience_timeline = re.findall(r'(\d+)\s*(year|yrs|years)\s*at\s*(.*)', resume_text.lower())
    if experience_timeline:
        for exp in experience_timeline:
            st.write(f"{exp[0]} years at {exp[2]}")
    else:
        st.write("No experience timeline found.")

    # # Additional content based on user type
    # if user_type == "Business Resume Analyzer":
    #     st.subheader("Business-Specific Analysis")
    #     st.write("This section provides additional insights tailored for business purposes.")
    #     # Add business-specific analysis and insights here
    # elif user_type == "Personal Resume Analyzer":
    #     st.write("Here are some personalized feedback and recommendations to improve your resume:")
    #     st.write("- Ensure your resume is tailored to the job description.")
    #     st.write("- Highlight your key achievements and responsibilities.")
    #     st.write("- Use action verbs to start your bullet points.")
    #     st.write("- Keep your resume concise and to the point.")
