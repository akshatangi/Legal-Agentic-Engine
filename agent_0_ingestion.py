import os
import pdfplumber
import re
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the REAL OpenNyAI / InLegalBERT model from Hugging Face
# Note: The first time you run this, it will download ~1.5GB to your local cache.
print("🧠 Loading InLegalBERT Model (This may take a minute on first run)...")
MODEL_NAME = "law-ai/InLegalBERT" 

# We set up a Hugging Face pipeline for text classification
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7) 
# Note: In a full production app, you'd load the specifically fine-tuned weights for rhetorical roles, 
# but this base structure proves the architecture works locally.

# Create the classifier pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)
print("✅ Model loaded successfully!")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts raw text from a PDF judgment."""
    print(f"📄 Opening PDF: {pdf_path}...")
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        print(f"✅ Extracted {len(text)} characters.")
        return text
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""

def semantic_chunker(text: str) -> list:
    """Chops the massive text wall into logical paragraphs using LangChain."""
    print("✂️ Chunking text using RecursiveCharacterTextSplitter...")
    
    # InLegalBERT has a 512 token limit (roughly 2000 characters). 
    # We will chunk at 1000 characters to be safe and give it plenty of context.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, # A little overlap so sentences aren't cut in half abruptly
        length_function=len,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    raw_chunks = text_splitter.split_text(text)
    
    # Filter out any tiny useless chunks
    valid_chunks = [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > 50]
    print(f"✅ Created {len(valid_chunks)} valid chunks.")
    
    return valid_chunks

def opennyai_role_classifier(chunks: list) -> dict:
    """
    Uses the local InLegalBERT neural network to classify paragraphs.
    """
    print("🏷️ Running neural classification on chunks...")
    
    labeled_chunks = {
        "FACT": [], "PETITIONER": [], "RESPONDENT": [], 
        "STATUTE": [], "PRECEDENT": [], "REASONING": [], "ORDER": []
    }
    
    # Map the model's output labels to our specific architecture buckets
    label_map = {
        "LABEL_0": "FACT", "LABEL_1": "PETITIONER", "LABEL_2": "RESPONDENT",
        "LABEL_3": "STATUTE", "LABEL_4": "PRECEDENT", "LABEL_5": "REASONING", "LABEL_6": "ORDER"
    }
    
    for chunk in chunks:
        chunk_id = f"chunk_{str(uuid.uuid4())[:8]}"
        chunk_obj = {"id": chunk_id, "text": chunk}
        
        # --- THE AI MAGIC HAPPENS HERE ---
        # The model reads the text and predicts its rhetorical role
        prediction = classifier(chunk)[0] 
        predicted_label = label_map.get(prediction['label'], "FACT") # Default to FACT if unsure
        
        labeled_chunks[predicted_label].append(chunk_obj)
            
    return labeled_chunks

if __name__ == "__main__":
    # 1. BULLETPROOF PATH: This forces Python to look in the exact same folder where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_folder = os.path.join(script_dir, "input_pdfs")
    
    print(f"\n🔍 Looking for PDFs in EXACT path: {pdf_folder}")
    
    # Check if the folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ Error: Could not find the folder. Please ensure 'input_pdfs' is in that exact path.")
    else:
        # 2. BULLETPROOF EXTENSION: Checks for .pdf, .PDF, .Pdf, etc.
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"⚠️ Found the folder, but no PDFs inside. Are you sure they have a .pdf extension?")
            # This prints everything in the folder so you can see what Python sees
            print(f"📄 Files currently in that folder: {os.listdir(pdf_folder)}") 
        else:
            print(f"📂 Found {len(pdf_files)} PDFs to process.\n")
            
            # We use a FOR loop to go through every single file
            for test_pdf in pdf_files:
                pdf_path = os.path.join(pdf_folder, test_pdf)
                
                print(f"\n=======================================================")
                print(f"🚀 NOW PROCESSING: {test_pdf}")
                print(f"=======================================================")
                
                # 1. Extract the raw text from the PDF
                raw_text = extract_text_from_pdf(pdf_path)
                
                if raw_text:
                    # 2. Chop the text into paragraphs
                    chunks = semantic_chunker(raw_text)
                    
                    # 3. Run the local AI to color-code the paragraphs
                    tagged_data = opennyai_role_classifier(chunks)
                    
                    # 4. Print a quick summary so you know it finished this PDF
                    print(f"✅ Finished sorting {test_pdf} into buckets.")