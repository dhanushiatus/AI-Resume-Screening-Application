import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader
from datetime import datetime

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('maxent_ne_chunker', quiet=True)
        nltk.download('words', quiet=True)
    except:
        st.warning("NLTK data download failed. Some features may be limited.")

# Load ML models with caching
@st.cache_resource
def load_models():
    try:
        clf = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        return clf, tfidf
    except FileNotFoundError:
        st.error("Model files not found. Please ensure clf.pkl and tfidf.pkl are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# Text cleaning function
def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text.strip()

# PDF text extraction
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text.strip() if text else "No text could be extracted from PDF"
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

# IMPROVED name extraction - ONLY name, no titles
import re

def extract_name(text):
    """
    Extract only Firstname Lastname (and Middlename if available)
    Strictly removes extra words like Science, Data, Engineer etc.
    """

    lines = text.strip().split('\n')
    first_lines = [line.strip() for line in lines[:10] if line.strip()]

    exclude_words = {
        'resume', 'cv', 'curriculum', 'vitae', 'application',
        'senior', 'junior', 'lead', 'head', 'chief', 'officer',
        'manager', 'engineer', 'developer', 'scientist',
        'analyst', 'consultant', 'professional', 'experienced',
        'seeking', 'objective', 'summary', 'undergraduate',
        'graduate', 'student', 'candidate', 'profile',
        'phone', 'email', 'address', 'contact',
        'linkedin', 'github', 'portfolio', 'website',
        'data', 'science', 'software', 'intern'
    }

    titles = {'dr', 'mr', 'mrs', 'ms', 'miss', 'prof', 'md', 'phd'}

    for line in first_lines:

        clean_line = re.sub(r'[^A-Za-z\s]', '', line)
        words = clean_line.split()

        name_words = []

        for word in words:
            word_lower = word.lower()

            # Stop immediately if we hit a non-name word AFTER collecting 2 words
            if word_lower in exclude_words:
                break

            if (word_lower not in titles and
                word.isalpha() and
                len(word) > 1):

                name_words.append(word.capitalize())

            # Stop after collecting 3 words max
            if len(name_words) == 3:
                break

        # Return only if we got at least 2 words
        if len(name_words) >= 2:
            return " ".join(name_words[:2])  # Only first & last name

    return "Not found"


# IMPROVED email extraction - get full email
import re

def extract_email(text):
    """
    Extract the most valid/realistic email address
    """

    # Standard email pattern
    email_pattern = r'\b[a-zA-Z0-9._%+-]{3,}@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    
    emails = re.findall(email_pattern, text)

    if not emails:
        return "Not found"

    # Filter unrealistic short usernames (like m@, a@)
    valid_emails = []
    for email in emails:
        username = email.split('@')[0]
        if len(username) >= 4:   # minimum realistic length
            valid_emails.append(email)

    if valid_emails:
        # Return the longest email (usually the real one)
        return max(valid_emails, key=len)

    # If nothing passes filter, return longest email found
    return max(emails, key=len)


# IMPROVED phone extraction - handle all formats properly
def extract_phone(text):
    """Extract phone number in proper format"""
    
    # First, clean the text around potential phone numbers
    # Look for phone indicators
    phone_indicators = ['phone', 'mobile', 'cell', 'tel', 'contact', 'call']
    
    # Method 1: Look for lines containing phone indicators
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(indicator in line_lower for indicator in phone_indicators):
            # Extract numbers from this line
            numbers = re.findall(r'[\d\+\-\(\)\s]{7,}', line)
            if numbers:
                phone_candidate = numbers[0].strip()
                # Clean up the phone number
                digits = re.sub(r'\D', '', phone_candidate)
                if 10 <= len(digits) <= 15:
                    return format_phone_number(phone_candidate)
    
    # Method 2: Standard phone patterns
    patterns = [
        # International: +1 (123) 456-7890 or +91 98765 43210
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{4,10}',
        
        # US/Canada: (123) 456-7890 or 123-456-7890 or 123.456.7890
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        
        # Simple 10-digit number (maybe with country code)
        r'\b\d{10,15}\b',
        
        # Numbers with spaces: 123 456 7890
        r'\b\d{3}\s\d{3}\s\d{4}\b',
    ]
    
    for pattern in patterns:
        phones = re.findall(pattern, text)
        if phones:
            phone = phones[0]
            # Verify it has enough digits
            digits = re.sub(r'\D', '', phone)
            if 10 <= len(digits) <= 15:
                return format_phone_number(phone)
    
    return "Not found"

def format_phone_number(phone):
    """Format phone number nicely"""
    # Remove all non-digit characters except +
    digits = re.sub(r'[^\d\+]', '', phone)
    
    # Format based on length
    if len(digits) == 10 and digits.isdigit():
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits.startswith('1'):
        return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    elif len(digits) == 12 and digits.startswith('91'):
        return f"+91 {digits[2:7]} {digits[7:]}"
    else:
        # Return as is but clean
        return phone.strip()

def extract_skills(text):
    skills_keywords = ["Python", "Java", "SQL", "Machine Learning", "Deep Learning",
                       "Computer Vision", "NLP", "React", "Streamlit", "Kotlin",
                       "JavaScript", "HTML", "CSS", "Django", "Flask", "AWS",
                       "Docker", "Git", "TensorFlow", "PyTorch", "Scikit-learn",
                       "Pandas", "NumPy", "Matplotlib", "Tableau", "Power BI",
                       "Azure", "GCP", "Kubernetes", "Jenkins", "Agile", "Scrum"]
    found = [skill for skill in skills_keywords if skill.lower() in text.lower()]
    return found if found else ["Not found"]

def extract_languages(text):
    languages_keywords = ["Python", "Java", "SQL", "JavaScript", "Kotlin", "C++", 
                         "C#", "Ruby", "PHP", "Swift", "Go", "Rust", "TypeScript",
                         "Scala", "R", "MATLAB", "Perl", "Haskell"]
    found = [lang for lang in languages_keywords if lang.lower() in text.lower()]
    return found if found else ["Not found"]

def extract_summary(text):
    # Extract first few sentences that might be a summary/profile
    sentences = re.split(r'[.!?]+', text)
    
    # Look for summary section
    summary_indicators = ['summary', 'profile', 'objective', 'about me', 'professional profile']
    
    for i, sentence in enumerate(sentences[:20]):
        sentence_lower = sentence.lower()
        if any(indicator in sentence_lower for indicator in summary_indicators):
            # Get the next 2-3 sentences as summary
            summary = sentence
            for j in range(1, 4):
                if i + j < len(sentences):
                    summary += ". " + sentences[i + j]
            return summary.strip()
    
    # If no summary section found, return first 300 characters
    if text:
        return text[:300] + "..." if len(text) > 300 else text
    return "No summary available"

# Category mapping
CATEGORY_MAPPING = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
    20: "Python Developer", 24: "Web Designing", 12: "HR",
    13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales",
    16: "Mechanical Engineer", 1: "Arts", 7: "Database",
    11: "Electrical Engineering", 14: "Health and Fitness",
    19: "PMO", 4: "Business Analyst", 9: "Dotnet Developer",
    2: "Automation Testing", 17: "Network Security Engineer",
    21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

# Dark theme only
def apply_dark_theme():
    st.markdown("""
    <style>
    /* Dark Elegance Theme */
    .stApp {
        background: linear-gradient(135deg, #2C3E50 0%, #4A569D 100%);
    }
    
    .stApp, .stMarkdown, .stText, h1, h2, h3, p, li, span {
        color: #ffffff !important;
    }
    
    .header-container {
        background: linear-gradient(135deg, #2C3E50 0%, #3498db 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #2C3E50 0%, #3498db 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px auto;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        max-width: 600px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-item {
        margin: 15px 0;
        padding: 12px;
        background: rgba(0,0,0,0.3);
        border-radius: 12px;
        font-size: 18px;
        border: 1px solid rgba(255,255,255,0.1);
        color: white !important;
    }
    
    .info-label {
        font-weight: bold;
        margin-right: 10px;
        color: #f1c40f !important;
    }
    
    .category-badge {
        background: linear-gradient(135deg, #2C3E50 0%, #3498db 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 50px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 20px auto;
        max-width: 500px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .section-card {
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .section-title {
        color: #ffffff !important;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 20px;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    
    .summary-text {
        background: rgba(0,0,0,0.3);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #3498db;
        font-size: 16px;
        line-height: 1.6;
        color: #ffffff !important;
    }
    
    .skill-tag {
        background: linear-gradient(135deg, #2C3E50 0%, #3498db 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 14px;
        margin: 5px;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .lang-tag {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 14px;
        margin: 5px;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2C3E50 0%, #3498db 100%);
        color: white;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    
    .stFileUploader {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #3498db !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: white !important;
        background-color: rgba(255,255,255,0.1) !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Resume Screening App",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
def main():
    # Download NLTK data
    download_nltk_data()
    
    # Load models
    clf, tfidf = load_models()
    
    # Apply dark theme
    apply_dark_theme()
    
    # Simple sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #4CAF50;">📄 Resume Scanner</h1>
        </div>
        """, unsafe_allow_html=True)
        
        menu_options = ["📄 Home", "ℹ️ About"]
        choice = st.radio("Navigation", menu_options, key="navigation")
        
        st.markdown("---")
        st.caption(f"Version: 2.0.0")
    
    # About page
    if choice == "ℹ️ About":
        st.markdown("""
        <div class="header-container">
            <h1>ℹ️ About Resume Screening App</h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-card">
                <h3>📋 Features</h3>
                <ul style="list-style-type: none; padding: 0;">
                    <li>✓ Upload resumes (PDF/TXT)</li>
                    <li>✓ Automatic job category prediction</li>
                    <li>✓ Extract personal information</li>
                    <li>✓ Skills and languages detection</li>
                    <li>✓ Professional summary extraction</li>
                    <li>✓ Dark theme only</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="section-card">
                <h3>🔧 Technology</h3>
                <ul style="list-style-type: none; padding: 0;">
                    <li>✓ Python 3.8+</li>
                    <li>✓ Streamlit</li>
                    <li>✓ Scikit-learn</li>
                    <li>✓ NLTK</li>
                    <li>✓ PyPDF2</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="section-card">
                <h3>📊 Categories</h3>
                <p>The model predicts 25 job categories including:</p>
                <div style="columns: 2;">
                    <ul>
                        <li>Java Developer</li>
                        <li>Python Developer</li>
                        <li>Data Science</li>
                        <li>DevOps Engineer</li>
                        <li>Web Designing</li>
                        <li>HR</li>
                        <li>Business Analyst</li>
                        <li>Testing</li>
                        <li>And more...</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="section-card">
                <h3>📧 Contact</h3>
                <p>Email: support@resumescanner.com<br>
                Website: www.resumescanner.com<br>
                GitHub: github.com/resumescanner</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Home page
    else:
        # Header
        st.markdown("""
        <div class="header-container">
            <h1>📄 Resume Screening App</h1>
            <p>Upload a resume to automatically extract information and predict the job category</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered upload section
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 📤 Upload Resume")
            upload_file = st.file_uploader(
                "Choose a file (PDF or TXT)",
                type=["txt", "pdf"],
                help="Maximum file size: 200MB",
                label_visibility="collapsed"
            )
        
        if upload_file is not None:
            # Show loading spinner
            with st.spinner("Processing resume..."):
                # Extract text
                if upload_file.type == "application/pdf":
                    resume_text = extract_text_from_pdf(upload_file)
                else:
                    resume_text = upload_file.read().decode('utf-8', errors='ignore')
                
                # Clean text
                resume_text = clean_text(resume_text)
                
                if resume_text and "Error" not in resume_text:
                    
                    # Extract information
                    name = extract_name(resume_text)
                    email = extract_email(resume_text)
                    phone = extract_phone(resume_text)
                    
                    # Personal Information Card - Centered
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center;">
                        <div class="info-card">
                            <h3>👤 Personal Information</h3>
                            <div class="info-item">
                                <span class="info-label">Name:</span> {name}
                            </div>
                            <div class="info-item">
                                <span class="info-label">Email:</span> {email}
                            </div>
                            <div class="info-item">
                                <span class="info-label">Phone:</span> {phone}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Prediction
                    input_features = tfidf.transform([resume_text])
                    prediction_id = clf.predict(input_features)[0]
                    predicted_category = CATEGORY_MAPPING.get(prediction_id, "Unknown")
                    
                    # Centered prediction
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(f"""
                        <div class="category-badge">
                            🎯 {predicted_category}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence (if available)
                        if hasattr(clf, 'predict_proba'):
                            probabilities = clf.predict_proba(input_features)[0]
                            confidence = max(probabilities) * 100
                            st.progress(confidence/100)
                            st.caption(f"Confidence: {confidence:.1f}%")
                    
                    # Summary Section
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📋 Professional Summary</div>', unsafe_allow_html=True)
                    summary = extract_summary(resume_text)
                    st.markdown(f'<div class="summary-text">{summary}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Skills and Languages
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="section-card">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">💻 Skills</div>', unsafe_allow_html=True)
                        skills = extract_skills(resume_text)
                        for skill in skills:
                            if skill != "Not found":
                                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                        if skills == ["Not found"]:
                            st.write("No skills detected")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="section-card">', unsafe_allow_html=True)
                        st.markdown('<div class="section-title">🔤 Programming Languages</div>', unsafe_allow_html=True)
                        languages = extract_languages(resume_text)
                        for lang in languages:
                            if lang != "Not found":
                                st.markdown(f'<span class="lang-tag">{lang}</span>', unsafe_allow_html=True)
                        if languages == ["Not found"]:
                            st.write("No programming languages detected")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Preview Summary Expander
                    st.markdown("---")
                    
                    # Create formatted summary text for preview
                    preview_text = f"""
═══════════════════════════════════════════
           CANDIDATE SCREENING REPORT
═══════════════════════════════════════════

👤 PERSONAL INFORMATION
───────────────────────────────────────────
Name:  {name}
Email: {email}
Phone: {phone}

🎯 PREDICTED ROLE
───────────────────────────────────────────
{predicted_category}
Confidence: {confidence:.1f}% ({datetime.now().strftime('%Y-%m-%d %H:%M')})

📋 PROFESSIONAL SUMMARY
───────────────────────────────────────────
{summary}

💻 SKILLS
───────────────────────────────────────────
{', '.join(skills) if skills != ['Not found'] else 'None detected'}

🔤 PROGRAMMING LANGUAGES
───────────────────────────────────────────
{', '.join(languages) if languages != ['Not found'] else 'None detected'}

═══════════════════════════════════════════
Generated by Resume Screening App
═══════════════════════════════════════════
                    """
                    
                    # Preview in expander
                    with st.expander("👁️ Preview Summary (before copying)"):
                        st.text(preview_text)
                
                else:
                    st.error("Could not extract text from the file. Please ensure it's a valid resume.")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 30px; color: #6c757d; font-size: 14px; border-top: 1px solid #dee2e6;">
        <p>© 2024 Resume Screening App | Made with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()