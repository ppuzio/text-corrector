import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import shutil

# Create local_model directory if it doesn't exist
if not os.path.exists("local_model"):
    os.makedirs("local_model")
else:
    # Clear existing model files to avoid conflicts
    print("Clearing existing model files...")
    for file in os.listdir("local_model"):
        file_path = os.path.join("local_model", file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Download model and tokenizer
model_name = "Grammarly/coedit-large"
print(f"Downloading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save them to local_model directory
print("Saving model and tokenizer to local_model directory")
tokenizer.save_pretrained("local_model")
model.save_pretrained("local_model")

print("Model download complete. Files saved to local_model directory.")
print("Contents of local_model directory:")
for file in os.listdir("local_model"):
    file_path = os.path.join("local_model", file)
    size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"  - {file} ({size:.2f} MB)") 