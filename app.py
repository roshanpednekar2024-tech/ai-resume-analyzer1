import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

# Updated to a currently active model endpoint
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

st.title("📄 AI Resume Analyzer")
st.subheader("Developed by Roshan R Pednekar")
st.write("Upload your resume and get AI suggestions")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("Resume uploaded successfully!")

    if st.button("Analyze Resume"):
        prompt = f"""
You are an expert HR.

Analyze this resume and give output in this format:

Resume Score: (give score out of 100)

Strengths:
Weaknesses:
Missing Skills:
Final Suggestions:

Resume:
{text[:4000]}
"""

        # Pass prompt directly as a string to avoid BadRequestError
        response = model.invoke(prompt)
        result = response.content

        st.write("## 📊 Analysis Result")
        st.write(result)

        # highlight score
        for line in result.split("\n"):
            if "score" in line.lower():
                st.success(line)

        # download button
        st.download_button(
            label="📥 Download Report",
            data=result,
            file_name="resume_analysis.txt",
            mime="text/plain"
        )
