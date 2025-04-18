import sys
sys.setrecursionlimit(3000)

from setuptools import setup
import os

# List all model files in the source directory
model_dir = "./local_model"
model_files = []
if os.path.exists(model_dir):
    for root, _, files in os.walk(model_dir):
        for file in files:
            source_path = os.path.join(root, file)
            # Get the relative path from the model_dir
            rel_path = os.path.relpath(os.path.dirname(source_path), model_dir)
            if rel_path == ".":  # Files in the root of local_model
                dest_dir = "local_model"
            else:  # Files in subdirectories
                dest_dir = os.path.join("local_model", rel_path)
            
            # Add each file to be placed in Resources/local_model
            model_files.append((dest_dir, [source_path]))
else:
    print("WARNING: local_model directory not found. Please run download_model.py first.")

APP = ['app.py']
DATA_FILES = model_files  # Include all model files

OPTIONS = {
    'argv_emulation': False,
    'plist': {
        'CFBundleName': 'TextCorrector',
        'CFBundleDisplayName': 'Text Corrector',
        'LSUIElement': True,
    },
    'packages': [
        'rumps', 
        'pyperclip', 
        'pynput', 
        'torch', 
        'transformers',
        'safetensors',
        'accelerate',
        'sentencepiece',
    ],
    'includes': ['torch.backends.mps'],
}

setup(
    app=APP,
    name='TextCorrector',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
