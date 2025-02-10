import os
import torch
import markdown
import tempfile
import google.generativeai as genai
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, request, flash, redirect, url_for
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import PyPDF2
import matplotlib.pyplot as plt
import shutil

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Configure Gemini AI
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Gemini API key not found. Set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=api_key)
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# Model paths from RadImageNet
MODEL_PATHS = {
    "Chest_Xray": "RadImageNet_pytorch/DenseNet121.pt",
    "Brain_MRI": "RadImageNet_pytorch/InceptionV3.pt",
    "Abdomen_CT": "RadImageNet_pytorch/ResNet50.pt",
    "Pneumonia": "RadImageNet_pytorch/Pneumonia_Model.pt"  # Path to Pneumonia Model
}

# Load pre-trained models
def load_models():
    models_dict = {
        "ResNet50": models.resnet50(weights="IMAGENET1K_V1"),  
        "InceptionV3": models.inception_v3(weights="IMAGENET1K_V1"),  
        "DenseNet121": models.densenet121(weights="IMAGENET1K_V1"),  
        "Pneumonia": models.resnet18(weights="IMAGENET1K_V1")  # Placeholder model
    }

    # Load weights if available
    for name, model in models_dict.items():
        model_path = MODEL_PATHS.get(name)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

    return models_dict

ml_models = load_models()

# Image preprocessing function
def preprocess_image(image_path):
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Use LLM to classify image type and detect body part
def analyze_image_type_and_body_part(image_path):
    prompt = """
    Identify the type of this medical image (X-ray, MRI, or CT scan) and the body part shown (e.g., Chest, Brain, Abdomen, Spine, etc.).
    Your response should be formatted as:
    Type: [X-ray/MRI/CT Scan]
    Body Part: [Chest/Brain/Abdomen/Spine/etc.]
    """
    try:
        image = Image.open(image_path)
        image.verify()  # Verify that the image is valid
    except Exception as e:
        return "Unknown", "Unknown", f"Error loading image: {str(e)}"
    
    try:
        response = llm_model.generate_content([prompt, image])
        analysis_result = response.text.strip()
    except Exception as e:
        return "Unknown", "Unknown", f"Error analyzing image: {str(e)}"
    
    detected_type, detected_body_part = "Unknown", "Unknown"
    error_message = ""
    
    lines = analysis_result.split("\n")
    for line in lines:
        if "Type:" in line:
            detected_type = line.split(":")[-1].strip()
        if "Body Part:" in line:
            detected_body_part = line.split(":")[-1].strip()
    
    if detected_type == "Unknown" or detected_body_part == "Unknown":
        error_message = f"Could not determine the type or body part. Result: {analysis_result}"

    return detected_type, detected_body_part, error_message

# Predict disease from medical image and generate graph
def analyze_medical_image(image_path):
    detected_type, detected_body_part, error_message = analyze_image_type_and_body_part(image_path)

    if error_message:
        return f"Error: {error_message}", None

    if detected_type == "Unknown" or detected_body_part == "Unknown":
        return f"Could not determine image type or body part. Detected Type: {detected_type}, Body Part: {detected_body_part}", None

    model_name = {
        "Chest": "Chest_Xray",
        "Brain": "Brain_MRI",
        "Abdomen": "Abdomen_CT",
        "Pneumonia": "Pneumonia"
    }.get(detected_body_part)

    model = ml_models.get(model_name)
    image_tensor = preprocess_image(image_path)

    if model:
        with torch.no_grad():
            output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy()
        predicted_class = probabilities.argmax()

        class_labels = {
            "Chest": ["Normal", "Pneumonia", "COVID-19", "Lung Cancer"],
            "Brain": ["Normal", "Tumor", "Stroke", "Alzheimer"],
            "Abdomen": ["Normal", "Kidney Stone", "Liver Disease", "Pancreatitis"],
            "Pneumonia": ["Normal", "Pneumonia"]
        }

        predicted_disease = class_labels.get(detected_body_part, ["Unknown Disease"])[predicted_class]

        # Generate prediction graph
        plt.figure(figsize=(10, 5))
        plt.bar(class_labels[detected_body_part], probabilities)
        plt.xlabel("Diseases")
        plt.ylabel("Probability")
        plt.title(f"Prediction Probabilities for {detected_body_part}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the graph to a temporary file
        graph_path = os.path.join("static", "prediction_graph.png")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)  # Create directory if doesn't exist
        plt.savefig(graph_path)
        plt.close()

        return f"{detected_body_part} Analysis: Predicted Disease - {predicted_disease}", graph_path
    else:
        # If model not found or not available, fall back to LLM
        return use_llm_for_analysis(image_path, detected_body_part)

def use_llm_for_analysis(image_path, detected_body_part):
    prompt = f"""
    I could not find a model to analyze the {detected_body_part} image, so I am going to use LLM to help with the analysis. Please analyze the image and provide the potential diseases or conditions.
    The image path is: {image_path}
    """
    response = llm_model.generate_content(prompt)
    return response.text.strip(), None

# Analyze extracted medical report text using LLM
def analyze_medical_report(report_text):
    prompt = f"""
    You are a medical expert AI. Analyze the following medical report and provide a summary of possible diseases, critical findings, and recommendations.

    Report Text:
    {report_text}

    Your response should be formatted as:
    - Possible Diseases:
    - Critical Findings:
    - Recommended Next Steps:
    """
    response = llm_model.generate_content(prompt)
    return response.text.strip()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text.strip() if text.strip() else "No readable text found in the PDF."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_report():
    if 'files' not in request.files:
        flash("No files uploaded.")
        return redirect(url_for('index'))

    files = request.files.getlist("files")
    if not files or all(file.filename == "" for file in files):
        flash("No files selected.")
        return redirect(url_for('index'))

    analysis_results = []
    graphs = []

    for file in files:
        if file.filename.endswith((".jpg", ".jpeg", ".png")):
            temp_path = os.path.join(tempfile.gettempdir(), file.filename)
            file.save(temp_path)

            try:
                with Image.open(temp_path) as img:
                    img.verify()
                analysis, graph_path = analyze_medical_image(temp_path)
                analysis_results.append(analysis)
                if graph_path:
                    graphs.append(graph_path)
            except UnidentifiedImageError:
                flash(f"Invalid or corrupted image file: {file.filename}")
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
            finally:
                try:
                    os.unlink(temp_path)  # Ensure the file is deleted
                except Exception as e:
                    flash(f"Error deleting temporary file: {str(e)}")

        elif file.filename.endswith(".pdf"):
            report_text = extract_text_from_pdf(file)
            analysis_results.append(analyze_medical_report(report_text))

    # Final AI Decision
    final_prompt = f"""
    Based on the following medical reports and image analyses, summarize the patient's condition and provide recommendations.

    Reports and Predictions:
    {analysis_results}
    """

    final_decision = llm_model.generate_content(final_prompt).text.strip()

    return render_template('index.html', analysis=markdown.markdown(final_decision), graphs=graphs)

if __name__ == '__main__':
    app.run(debug=True)
