from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from io import StringIO

# Load environment variables
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Streamlit app header
st.header('Tehkeek Tool')
st.write('This tool allows you to generate a summary based on your input or a predefined research paper.')

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Option selection
option = st.radio(
    "Choose an option:",
    ["Enter Custom Input", "Select Predefined Research Paper"]
)

# Initialize variables
user_input = ""
paper_input = ""
style_input = ""
length_input = ""
tone_input = ""
language_input = ""

# Custom input option
if option == "Enter Custom Input":
    user_input = st.text_area('Enter Your Text', placeholder='Type your text here...')
    uploaded_file = st.file_uploader("Or upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            # Extract text from PDF
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            user_input = text
        elif uploaded_file.type == "text/plain":
            # Read text from TXT file
            user_input = uploaded_file.getvalue().decode("utf-8")
        st.write("File uploaded successfully!")

# Predefined research paper option
else:
    paper_input = st.selectbox(
        "Select Research Paper Name",
        [
            "NEURAL MACHINE TRANSLATION",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-3: Language Models are Few-Shot Learners",
            "Attention Is All You Need",
            "Sequence to Sequence Learningwith Neural Networks",
            "Universal Language Model Fine-tuning for Text Classification",
            "Diffusion Models Beat GANs on Image Synthesis"
        ]
    )

    style_input = st.selectbox(
        "Select Explanation Style",
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
    )

    length_input = st.selectbox(
        "Select Explanation Length",
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
    )

    tone_input = st.selectbox(
        "Select Tone",
        ["Formal", "Casual", "Academic", "Technical"]
    )

    language_input = st.selectbox(
        "Select Language",
        ["English", "Spanish", "French", "German", "Chinese", "Urdu"]
    )

# Define the prompt template for predefined research papers
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" in {language_input} with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  
Tone: {tone_input}  
1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  
If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style, length, tone, and language.
""",
    input_variables=['paper_input', 'style_input', 'length_input', 'tone_input', 'language_input']
)

# Button to trigger the model
if st.button('Generate Summary'):
    with st.spinner("Generating summary..."):
        if user_input.strip() or (paper_input and style_input and length_input):
            try:
                if option == "Enter Custom Input":
                    result = model.invoke(f"Summarize the following text in a clear and concise manner:\n\n{user_input}")
                else:
                    prompt = template.invoke({
                        'paper_input': paper_input,
                        'style_input': style_input,
                        'length_input': length_input,
                        'tone_input': tone_input,
                        'language_input': language_input
                    })
                    result = model.invoke(prompt)

                # Save summary to history
                st.session_state.history.append(result.content)
                st.write("### Summary:")
                st.write(result.content)

                # Show word count
                word_count = len(result.content.split())
                st.write(f"**Word Count:** {word_count}")

                # Download button
                summary_file = StringIO(result.content)
                st.download_button(
                    label="Download Summary",
                    data=summary_file.getvalue(),
                    file_name="summary.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please fill in all required fields.")

# Display history
if st.session_state.history:
    st.write("### History")
    for i, summary in enumerate(st.session_state.history, 1):
        st.write(f"**Summary {i}:**")
        st.write(summary)

# Clear history button
if st.button('Clear History'):
    st.session_state.history = []
    st.write("History cleared!")

# Feedback mechanism
feedback = st.text_area("Provide Feedback on the Summary (Optional)")
if feedback:
    st.write("Thank you for your feedback!")

# Dark mode toggle
dark_mode = st.checkbox("Enable Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )