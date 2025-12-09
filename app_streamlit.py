"""
Streamlit web app for cover letter generation.

Run with:
    streamlit run app_streamlit.py
"""

import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import generate_rag_lora_model

st.set_page_config(
    page_title="Cover Letter Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        min-height: 200px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> AI Cover Letter Generator</h1>', unsafe_allow_html=True)
st.markdown("---")


with st.sidebar:
    st.header("ℹ How to Use")
    st.markdown("""
    1. **Paste your resume** in the left text area
    2. **Paste the job description** in the right text area
    3. Click **Generate Cover Letter**
    4. Review and copy your personalized cover letter!
    
    ---
    
    **Note:** The first generation may take a minute to load the model.
    Subsequent generations will be faster.
    """)
    
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.markdown("- RAG (Retrieval-Augmented Generation)")
    st.markdown("- LoRA Fine-tuned T5")
    st.markdown("- FAISS Vector Search")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Resume")
    resume_text = st.text_area(
        "Paste your resume text here:",
        height=400,
        placeholder="""Example:
John Doe
Email: john@example.com

EXPERIENCE
Software Engineer Intern, Tech Corp (Summer 2023)
- Developed web applications using Python and React
- Collaborated with team of 5 engineers
- Improved API response time by 30%

PROJECTS
Personal Portfolio Website
- Built responsive website using HTML, CSS, JavaScript
- Deployed on AWS
""",
        help="Paste your full resume text. The more detail, the better the cover letter!"
    )

with col2:
    st.subheader("Job Description")
    job_text = st.text_area(
        "Paste the job description of the company here:",
        height=400,
        placeholder="""Example:
Software Engineer Intern

Company: Tech Innovations Inc.

We are looking for a motivated Software Engineer Intern to join our team.
The ideal candidate will have:
- Experience with Python and web development
- Strong problem-solving skills
- Ability to work in a team environment

Responsibilities:
- Develop and maintain web applications
- Collaborate with cross-functional teams
- Write clean, maintainable code
""",
        help="Paste the full job description including company name, requirements, and responsibilities."
    )

# Generate button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    generate_button = st.button(
        "Generate Cover Letter",
        type="primary",
        use_container_width=True,
        help="Click to generate your personalized cover letter"
    )

if generate_button:
    if not resume_text.strip():
        st.error("Please paste your resume text in the left column.")
    elif not job_text.strip():
        st.error("Please paste the job description in the right column.")
    else:
        with st.spinner("Generating your cover letter... This may take 30-60 seconds."):
            try:
                cover_letter = generate_rag_lora_model(resume_text, job_text)
                
                st.markdown("---")
                st.subheader("Your Generated Cover Letter")
                
                st.text_area(
                    "Cover Letter:",
                    value=cover_letter,
                    height=500,
                    label_visibility="collapsed"
                )
                
                st.download_button(
                    label="Download as .txt",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.success("Cover letter generated successfully! Review it above and make any edits you'd like.")
                
            except FileNotFoundError as e:
                error_msg = str(e)
                if "best_model.pt" in error_msg:
                    st.error("""
                    **Model file not found on Streamlit Cloud!**
                    
                    The trained model file (951MB) is not in your GitHub repository because it's ignored by `.gitignore`.
                    
                    **To fix this for Streamlit Cloud deployment:**
                    1. **Upload model to HuggingFace Hub** (recommended), OR
                    2. **Use Git LFS** to store the model file in your repo
                    
                    See `STREAMLIT_CLOUD_SETUP.md` for detailed instructions.
                    """)
                elif "faiss_index.bin" in error_msg or "metadata.jsonl" in error_msg:
                    st.warning("FAISS index not found. Attempting to build it...")
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, "scripts/build_embeddings.py"],
                            capture_output=True,
                            text=True,
                            timeout=300
                        )
                        if result.returncode == 0:
                            st.success("FAISS index built successfully! Please try generating again.")
                            st.rerun()
                        else:
                            st.error(f"Failed to build FAISS index: {result.stderr}")
                    except Exception as build_error:
                        st.error(f"Could not build FAISS index: {str(build_error)}")
                        st.info("Make sure you've run `python scripts/build_original_dataset.py` first.")
                else:
                    st.error(f"File not found: {error_msg}")
            except Exception as e:
                st.error(f"An error occurred while generating the cover letter:\n\n{str(e)}")
                st.info("""
                **Troubleshooting:**
                1. Check Streamlit Cloud logs for detailed error messages
                2. Ensure all required files are in your repository
                3. See `STREAMLIT_CLOUD_SETUP.md` for deployment instructions
                """)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built using RAG + LoRA</div>",
    unsafe_allow_html=True
)

