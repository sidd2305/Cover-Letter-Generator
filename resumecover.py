# -*- coding: utf-8 -*-
"""resumecover

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WBKt_HPhm0HdojRF6vRPPzQpQSjIIrLI
"""

# importing required modules
import PyPDF2
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdfReader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdfReader.pages:
        text += page.extract_text()
    return text

# Function to display chat history
def display_chat_history(messages):
    for message in messages:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

# Function to get assistant's response
def get_assistant_response(messages):
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    )
    response = r.choices[0].message.content
    return response

# Streamlit app
def main():
    st.title("Cover Letter Generator")
    st.write("Upload your resume PDF and enter the job description to generate a cover letter.")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload your resume PDF", type=["pdf"])
    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)

        # Input job description
        job_description = st.text_area("Enter the target job description")

        # Initialize chat history
        messages = [{"role": "assistant", "content": "How can I help?"}]

        # Generate cover letter button
        if st.button("Generate Cover Letter"):
            if job_description:
                # Construct assistant message
                assistant_message = "Create a cover letter for a job application for this job description: " + job_description + ". Details of the applicant are as follows: " + resume_text

                # Display chat history
                # display_chat_history(messages)

                # Add assistant message to messages
                messages.append({"role": "assistant", "content": assistant_message})

                # Get assistant's response
                assistant_response = get_assistant_response(messages)
                messages.append({"role": "assistant", "content": assistant_response})

                # Display updated chat history
                # display_chat_history(messages)

                # Display cover letter
                st.write("Cover Letter:")
                st.write(assistant_response)
            else:
                st.warning("Please enter the job description.")

if __name__ == "__main__":
    main()
