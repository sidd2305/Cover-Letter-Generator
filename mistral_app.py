import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2

client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

def extract_text_from_pdf(pdf_file):
    pdfReader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

def save_to_file(output_text, filename='cover_letter.txt'):
    with open(filename, 'w') as f:
        f.write(output_text)

def format_prompt(message, history):
    prompt = ""
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt


def generate(prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False
    )
    output = ""

    for response in stream:
        output += response.token.text
    
    return output


def main():
   
    st.title("Cover Letter Generator✉️")
    st.write("Upload your resume PDF and enter the Job Description to generate a cover letter using Mistral 7B.📩✨")
    st.write("⚠️Do Remember to upload a resume to avoid any errors⚠️")
    st.write("Tip😉-Do scroll down to also download your cover letter as a file.")
    # Upload PDF file
    uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
    job_description = st.text_area("Enter the target job description")
    
    history = []  # You may add functionality to maintain history here

    # temperature = st.slider("Temperature", 0.0, 1.0, 0.9, step=0.05)
    max_new_tokens = st.slider("Max number of tokens(768 is optimal)", 0, 1048, 768, step=64)
    # top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, step=0.05)
    # repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.2, step=0.05)
    prompt = "Create a cover letter for a job application for this job description: " + job_description + ".Extract neccessary skills from the resume and frame it according the job description. Resume  of the applicant are as follows:" + resume_text

    if st.button("Generate"):
        st.write("Generating A Cover Letter For you...")
        output_text = generate(prompt, history, 0.90, max_new_tokens, 0.90, 1.20)
        st.write(output_text[:-4])
        save_to_file(output_text)
        # save_to_file(output_text)
        
        st.subheader("Download the Cover Letter")
        st.download_button(label="Download Cover Letter", data=open('cover_letter.txt', 'rb'), file_name='cover_letter.txt')


if __name__ == "__main__":
    main()
