import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw
import PyPDF2
import tempfile
import os
from google.api_core import exceptions
from dotenv import load_dotenv
import time
from flask import Flask, render_template, request
import markdown

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure the Gemini AI model
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def analyze_medical_report(content, content_type):
    prompt = "Analyze this medical report in detail. Identify and summarize the key findings, diagnoses, and observations concisely. Provide actionable recommendations, recovery tips, and potential home remedies where applicable. Additionally, suggest any lifestyle adjustments or further diagnostic tests required to ensure a thorough understanding of the patient's condition. If possible, explain complex medical terms in simpler language for better clarity."
    
    for attempt in range(MAX_RETRIES):
        try:
            if content_type == "image":
                response = model.generate_content([prompt, content])
            else:  # text
                # Gemini 1.5 Flash can handle larger inputs, so we'll send the full text
                response = model.generate_content(f"{prompt}\n\n{content}")
            
            return response.text
        except exceptions.GoogleAPIError as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"An error occurred. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                st.error(f"Failed to analyze the report after {MAX_RETRIES} attempts. Error: {str(e)}")
                return fallback_analysis(content, content_type)

def fallback_analysis(content, content_type):
    st.warning("Using fallback analysis method due to API issues.")
    if content_type == "image":
        return "Unable to analyze the image due to API issues. Please try again later or consult a medical professional for accurate interpretation."
    else:  # text
        word_count = len(content.split())
        return f"""
        Fallback Analysis:
        1. Document Type: Text-based medical report
        2. Word Count: Approximately {word_count} words
        3. Content: The document appears to contain medical information, but detailed analysis is unavailable due to technical issues.
        4. Recommendation: Please review the document manually or consult with a healthcare professional for accurate interpretation.
        5. Note: This is a simplified analysis due to temporary unavailability of the AI service. For a comprehensive analysis, please try again later.
        """

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_report():
    # Get file data
    file = request.files['file']
    file_type = request.form['file_type']

    image_path = None  # Initialize image_path to avoid UnboundLocalError

    # Process the file accordingly
    if file_type == "image":
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        image = Image.open(tmp_file_path)
        
        # Optional: Process the image (add analysis, text, etc.)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Analysis: Example text", fill="red")
        
        # Define a valid image path
        image_path = r"C:\Users\Komal\Desktop\KKCODINGMAIN\Data Analytics\Medical Api\upload\analyzed_image.png"
        image.save(image_path)
        
        analysis = analyze_medical_report(image, "image")
        os.unlink(tmp_file_path)
    else:  # PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        with open(tmp_file_path, 'rb') as pdf_file:
            pdf_text = extract_text_from_pdf(pdf_file)
        analysis = analyze_medical_report(pdf_text, "text")
        os.unlink(tmp_file_path)

    # Convert Markdown to HTML
    analysis_html = markdown.markdown(analysis)

    # Ensure image_path is None for PDFs
    return render_template('index.html', analysis=analysis_html, image_path=image_path if image_path else "")

if __name__ == '__main__':
    app.run(debug=True)
