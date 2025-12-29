import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Page config - makes it look pro
st.set_page_config(
    page_title="Internship Radar",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1d4ed8;
        --secondary: #8b5cf6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --light: #f8fafc;
        --dark: #1e293b;
    }
    
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .match-score {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 25px;
        border-radius: 20px;
        margin: 25px 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 1px;
    }
    
    .high-score { 
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .medium-score { 
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .low-score { 
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .skill-chip {
        display: inline-block;
        padding: 8px 18px;
        margin: 6px;
        background: #e0e7ff;
        border-radius: 25px;
        font-size: 0.95rem;
        color: #1e40af;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border: 2px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .skill-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.2);
    }
    
    .missing-chip {
        background: #fee2e2;
        color: #991b1b;
        font-weight: 600;
        border: 2px solid #ef4444;
    }
    
    .missing-chip:hover {
        box-shadow: 0 5px 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Improve text areas */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 12px;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        transition: all 0.3s ease;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header with description
st.markdown('<h1 class="main-header">🔍 Internship Radar</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
    Paste your resume and a job description to get instant feedback on your fit<br>
    <em>Identifies skill gaps and suggests improvements</em>
    </p>
</div>
""", unsafe_allow_html=True)

# Sample data for new users
SAMPLE_RESUME = """Computer Science Student at University of California
• Proficient in Python, Java, and JavaScript
• Experience with React and Flask frameworks
• Built a task management web application using Python and SQL
• Familiar with Git version control and Docker
• Strong problem-solving and teamwork skills
• Coursework: Data Structures, Algorithms, Database Systems"""

SAMPLE_JOB = """Software Engineering Intern - Summer 2024
Requirements:
• Strong programming skills in Python or Java
• Experience with React.js or similar frontend frameworks
• Knowledge of AWS cloud services (EC2, S3)
• Familiarity with Docker and Kubernetes
• Understanding of REST APIs and microservices
• Experience with SQL and NoSQL databases (MongoDB preferred)
• Machine learning basics is a plus
• Excellent communication and collaboration skills"""

# Sidebar for instructions
with st.sidebar:
    st.header("📚 How to Use")
    st.markdown("""
    1. **Paste your resume** in the left box
    2. **Paste a job description** in the right box
    3. Click **Analyze Match**
    4. Review your score and missing skills
    5. Use suggestions to improve your resume!
    
    **Tip:** Use the sample data to test first!
    """)
    
    if st.button("📋 Load Sample Data"):
        st.session_state.resume_text = SAMPLE_RESUME
        st.session_state.job_text = SAMPLE_JOB
        st.rerun()

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Your Resume")
    resume_text = st.text_area(
        "Paste your resume here:",
        height=250,
        value=st.session_state.get('resume_text', ''),
        placeholder="Paste your resume text here...",
        key="resume_input"
    )

with col2:
    st.subheader("📋 Job Description")
    job_text = st.text_area(
        "Paste job posting here:",
        height=250,
        value=st.session_state.get('job_text', ''),
        placeholder="Paste the job description here...",
        key="job_input"
    )

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🚀 Analyze Match Score", type="primary", use_container_width=True)

if analyze_btn and resume_text and job_text:
    with st.spinner("Analyzing your resume... This might take a few seconds"):
        # Calculate match score
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        match_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
        match_score = round(match_score, 1)
        
        # Extract skills (simple version)
        skill_categories = {
            'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift'],
            'Web Development': ['react', 'angular', 'vue', 'django', 'flask', 'node.js', 'html', 'css'],
            'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'Cloud & DevOps': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'Tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'figma'],
            'Data Science': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'tableau']
        }
        
        resume_skills = {}
        job_skills = {}
        
        for category, skills in skill_categories.items():
            resume_skills[category] = [s for s in skills if s in resume_text.lower()]
            job_skills[category] = [s for s in skills if s in job_text.lower()]
        
        # Find missing skills
        missing_skills = {}
        for category in job_skills:
            missing = [s for s in job_skills[category] if s not in resume_skills[category]]
            if missing:
                missing_skills[category] = missing
    
    # Display results
    st.divider()
    
    # Match Score with visual
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if match_score >= 70:
            st.markdown(f'<div class="match-score high-score">🎯 {match_score}% Match</div>', unsafe_allow_html=True)
            st.success("**Great match!** You have most required skills.")
        elif match_score >= 40:
            st.markdown(f'<div class="match-score medium-score">📊 {match_score}% Match</div>', unsafe_allow_html=True)
            st.warning("**Moderate match.** Consider adding some missing skills.")
        else:
            st.markdown(f'<div class="match-score low-score">⚠️ {match_score}% Match</div>', unsafe_allow_html=True)
            st.error("**Needs improvement.** Focus on learning missing skills.")
    
    # Skills Analysis in tabs
    tab1, tab2, tab3 = st.tabs(["✅ Your Skills", "🎯 Missing Skills", "💡 Recommendations"])
    
    with tab1:
        st.subheader("Skills Found in Your Resume")
        for category, skills in resume_skills.items():
            if skills:
                st.markdown(f"**{category}:**")
                for skill in skills:
                    st.markdown(f'<span class="skill-chip">{skill.title()}</span>', unsafe_allow_html=True)
                st.write("")
    
    with tab2:
        if missing_skills:
            st.subheader("Skills to Add to Your Resume")
            for category, skills in missing_skills.items():
                st.markdown(f"**{category}:**")
                for skill in skills:
                    st.markdown(f'<span class="skill-chip missing-chip">{skill.title()}</span>', unsafe_allow_html=True)
                st.write("")
        else:
            st.success("🎉 Perfect! No missing skills found!")
    
    with tab3:
        st.subheader("Actionable Next Steps")
        
        if missing_skills:
            # Learning plan
            st.markdown("### 📚 What to Learn This Week")
            for category, skills in missing_skills.items():
                if skills:
                    top_skill = skills[0]
                    st.markdown(f"""
                    **For {top_skill.title()}:**
                    - Complete a tutorial on freeCodeCamp or Codecademy
                    - Build a small project (e.g., a to-do app)
                    - Add it to your resume as: *"Learning {top_skill.title()} through hands-on projects"*
                    """)
                    break  # Just focus on one skill
            
            # Resume suggestions
            st.markdown("### ✨ Resume Bullet Suggestions")
            suggestions = [
                "Developed applications using [SKILL] to improve [OUTCOME]",
                "Implemented [SKILL] in projects resulting in [METRIC] improvement",
                "Applied [SKILL] to solve [PROBLEM] with [RESULT]",
                "Utilized [SKILL] to build [PROJECT] that [ACHIEVEMENT]"
            ]
            
            for i, suggestion in enumerate(suggestions[:3], 1):
                st.markdown(f"{i}. {suggestion}")
        
        # General tips
        st.markdown("### 💼 General Tips")
        st.markdown("""
        - **Quantify achievements**: Instead of "Built an app", say "Built an app used by 100+ users"
        - **Use action verbs**: "Developed", "Implemented", "Optimized", "Led"
        - **Tailor for each application**: Adjust keywords based on job description
        - **Show projects**: GitHub links > course lists
        """)
    
    # Export option
    st.divider()
    st.subheader("📥 Save Your Analysis")
    
    export_text = f"""# Internship Radar Analysis
## Match Score: {match_score}%
## Missing Skills:
{missing_skills}
## Action Plan:
1. Focus on learning: {list(missing_skills.values())[0][0] if missing_skills else 'No missing skills!'}
2. Update resume with suggested bullet points
3. Practice interview questions on these topics
---
*Generated by Internship Radar - Your AI Career Assistant*
"""
    
    st.download_button(
        label="Download Analysis as Text File",
        data=export_text,
        file_name="internship_analysis.txt",
        mime="text/plain",
        use_container_width=True
    )

elif analyze_btn:
    st.error("⚠️ Please paste both your resume and a job description!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9rem;'>
    <p>Built with ❤️ by a CS student for CS students | Uses TF-IDF + Cosine Similarity</p>
    <p>Perfect for internship applications and resume optimization</p>
</div>
""", unsafe_allow_html=True)