import rumps
import pyperclip
from pynput import keyboard
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import threading
import os
import sys
import json
import subprocess
import time

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TextCorrectorApp(rumps.App):
    def __init__(self):
        super(TextCorrectorApp, self).__init__("✓", quit_button=rumps.MenuItem("Quit"))
        self.menu = [
            "Correct Clipboard (⌘+`)",
            None,  # Separator
            "Minimal Corrections",
            None,  # Separator
            "Simple Mode",
            "Settings"
        ]
        
        # Initialize model to None
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Model configuration - using Grammarly's professional grammar correction model
        self.model_name = "Grammarly/coedit-large"
        self.prefix = ""  # Grammarly's model doesn't need a special prefix
        
        # Flag for minimal corrections
        self.minimal_corrections = True
        
        # Flag for simple mode (doesn't preserve clipboard)
        self.simple_mode = False
        
        # Set up notifier
        self.setup_notifier()
        
        # Start loading model in background
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Try a different approach to setting up the keyboard listener (safer)
        try:
            self.setup_keyboard_listener()
            print("Keyboard listener set up successfully")
        except Exception as e:
            print(f"Failed to set up keyboard listener: {e}")
            self.show_notification("Warning", "Keyboard shortcut may not work. Use the menu item instead.")
        
        print("App initialized")
        self.show_notification("Started", "Application is starting up...")
        self.show_notification("Usage", "Copy text (⌘+C), then press ⌘+` to correct")
    
    def setup_keyboard_listener(self):
        """Set up the keyboard listener with error handling"""
        try:
            # Try using the HotKey approach instead of GlobalHotKeys
            self.hotkey = keyboard.HotKey(
                keyboard.HotKey.parse('<cmd>+`'), 
                self.on_hotkey_activated
            )
            
            # Set up a standard listener for key press
            def on_press(key):
                try:
                    self.hotkey.press(key)
                except Exception as e:
                    print(f"Error in hotkey press: {e}")
            
            def on_release(key):
                try:
                    self.hotkey.release(key)
                except Exception as e:
                    print(f"Error in hotkey release: {e}")
            
            # Start the listener in a non-blocking way
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.start()
            
            # Fallback option in case the hotkey doesn't work
            print("Keyboard listener started with HotKey approach")
            
        except Exception as e:
            print(f"Error setting up HotKey: {e}")
            # Try the GlobalHotKeys as fallback but with try-except
            try:
                self.listener = keyboard.GlobalHotKeys({
                    '<cmd>+`': lambda: self.on_hotkey_activated()
                })
                self.listener.start()
                print("Keyboard listener started with GlobalHotKeys approach")
            except Exception as e:
                print(f"Failed to set up GlobalHotKeys too: {e}")
                raise
    
    def on_hotkey_activated(self):
        """Handler for hotkey activation - safer than direct callback"""
        try:
            # Use a separate thread to avoid blocking the listener
            threading.Thread(target=self.correct_clipboard, daemon=True).start()
        except Exception as e:
            print(f"Error handling hotkey: {e}")
            self.show_notification("Error", "Failed to process hotkey command")
    
    def setup_notifier(self):
        """Setup terminal-notifier if available, otherwise fall back to other methods"""
        # Check if terminal-notifier is installed
        try:
            result = subprocess.run(["which", "terminal-notifier"], capture_output=True, text=True)
            self.terminal_notifier_path = result.stdout.strip() if result.returncode == 0 else None
            
            if self.terminal_notifier_path:
                print(f"Using terminal-notifier from: {self.terminal_notifier_path}")
            else:
                print("terminal-notifier not found, attempting to install it")
                try:
                    # Try to install terminal-notifier using brew
                    subprocess.run(["brew", "install", "terminal-notifier"], check=True)
                    print("terminal-notifier installed successfully")
                    self.terminal_notifier_path = "terminal-notifier"
                except:
                    print("Could not install terminal-notifier")
                    self.terminal_notifier_path = None
        except Exception as e:
            print(f"Error setting up notifier: {e}")
            self.terminal_notifier_path = None
        
    def show_notification(self, title, message):
        """Show a notification using the best available method"""
        print(f"NOTIFICATION: {title} - {message}")
        
        # Replace newlines with spaces for better compatibility
        message = message.replace("\n", " - ")
        
        # Method 1: terminal-notifier (most reliable)
        if self.terminal_notifier_path:
            try:
                subprocess.run([
                    self.terminal_notifier_path,
                    "-title", "Text Corrector",
                    "-subtitle", title,
                    "-message", message,
                    "-sound", "Glass"
                ], check=False)
                return
            except Exception as e:
                print(f"terminal-notifier failed: {e}")
        
        # Method 2: native macOS notification via osascript
        try:
            # Escape all quotes properly for shell command
            safe_message = message.replace("'", "'\\''").replace('"', '\\"')
            safe_title = title.replace("'", "'\\''").replace('"', '\\"')
            
            # Use a simpler AppleScript command to avoid syntax issues
            cmd = ["osascript", "-e", f'display notification "{safe_message}" with title "Text Corrector" subtitle "{safe_title}"']
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"osascript notification failed: {e}")
        
        # Method 3: rumps notification as last resort
        try:
            rumps.notification("Text Corrector", title, message)
        except Exception as e:
            print(f"rumps notification failed: {e}")
            
        # Method 4: Always play a sound regardless of notification success
        try:
            subprocess.run(["afplay", "/System/Library/Sounds/Glass.aiff"], check=False)
        except Exception as e:
            print(f"Sound failed: {e}")

    def process_text(self, text):
        """Process text through the model and return corrected version"""
        # Get MPS device reference
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Grammarly's model doesn't need a special prefix
        formatted_input = text
            
        print(f"Processing input: {formatted_input[:50]}...")
        
        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt", max_length=512, truncation=True).to(device)
        print("Text tokenized")
        
        # Generate corrected text
        print("Generating corrected text...")
        
        # Generation parameters optimized for Grammarly's model
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            min_length=max(10, len(text) // 2),  # Don't make it too short
            num_beams=5,
            length_penalty=1.0,  # Discourage length changes
            early_stopping=True,
            do_sample=False,     # Deterministic output is better for grammar
            repetition_penalty=1.2,
            num_return_sequences=1
        )
        
        # Decode the output tokens
        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Raw model output: {corrected[:50]}... ({len(corrected)} chars)")
        
        # Check if the output is too short or empty
        if len(corrected.strip()) < min(5, len(text)):
            print("Model output too short, using original text")
            corrected = text
        
        # Make sure first letter is capitalized for sentences
        if corrected and corrected[0].islower() and not corrected.startswith(('i ', 'i\'', 'iPhone')):
            corrected = corrected[0].upper() + corrected[1:]
        
        # Make sure "I" is always capitalized
        for i_pattern in [' i ', ' i\'m', ' i\'ll', ' i\'ve', ' i\'d', ' i think', ' i believe']:
            if i_pattern in corrected.lower():
                corrected = corrected.replace(i_pattern.lower(), i_pattern.replace(' i', ' I'))
        
        print(f"Final corrected text: {corrected[:50]}... ({len(corrected)} chars)")
        
        return corrected

    def correct_clipboard(self, *args):
        """Simple and reliable clipboard-based correction method"""
        print("Clipboard correction requested")
        if not self.model_loaded:
            print("Model not loaded yet")
            self.show_notification("Please Wait", "Model is still loading")
            return
        
        # Get text from clipboard
        try:
            text = pyperclip.paste()
        except Exception as e:
            print(f"Error accessing clipboard: {e}")
            self.show_notification("Error", "Could not access clipboard")
            return
            
        if not text or len(text.strip()) == 0:
            print("No text in clipboard")
            self.show_notification("Error", "No text in clipboard - please copy (⌘+C) text first")
            return
        
        print(f"Processing text from clipboard: {text[:50]}... ({len(text)} chars)")
        
        # Store original text for comparison
        original_text = text.strip()
        
        try:
            # Process the text using the processing function
            corrected = self.process_text(text)
            
            # In simple mode, just replace clipboard without trying to restore
            if self.simple_mode:
                # Simply put the corrected text on clipboard
                pyperclip.copy(corrected)
                print("Corrected text copied to clipboard (simple mode)")
                self.show_notification("Text Corrected ✓", "Corrected text is now on clipboard. Press ⌘+V to paste.")
                return
            
            # In standard mode, check for differences and notify
            if original_text != corrected.strip():
                # Text was corrected
                pyperclip.copy(corrected)
                print("Corrected text copied to clipboard")
                
                # Create a compact difference preview
                orig_preview = original_text[:40] + "..." if len(original_text) > 40 else original_text
                corr_preview = corrected.strip()[:40] + "..." if len(corrected.strip()) > 40 else corrected.strip()
                
                self.show_notification("Text Corrected ✓", 
                                     f"From: {orig_preview}\nTo: {corr_preview}\n\nCorrected text is on clipboard. Press ⌘+V to paste.")
            else:
                # No changes needed
                self.show_notification("No Changes Needed", "Text was already correct")
        except Exception as e:
            error_msg = f"Failed to correct text: {str(e)}"
            print(f"Error correcting text: {e}")
            self.show_notification("Error", error_msg)
    
    def load_model(self):
        print("------- MODEL LOADING STARTED -------")
        self.show_notification("Loading Model", "Please wait...")
        
        # Always use the local model directory
        if getattr(sys, 'frozen', False):
            print("Running as bundled app")
            app_path = os.path.dirname(sys.executable)
            resources_dir = os.path.abspath(os.path.join(app_path, '..', 'Resources'))
            self.model_path = os.path.join(resources_dir, 'local_model')
            print(f"Using bundled model from: {self.model_path}")
        else:
            print("Running as script")
            self.model_path = os.path.abspath("./local_model")
            print(f"Using local model from: {self.model_path}")
            
        # Check if the local model exists - look for either pytorch_model.bin or model.safetensors
        model_files = [
            os.path.join(self.model_path, "pytorch_model.bin"),
            os.path.join(self.model_path, "model.safetensors")
        ]
        
        if not os.path.exists(self.model_path) or not any(os.path.isfile(f) for f in model_files):
            # List what files are in the directory if it exists
            if os.path.exists(self.model_path):
                print(f"Directory exists but no model files found. Contents of {self.model_path}:")
                for file in os.listdir(self.model_path):
                    print(f"  - {file}")
            
            error_msg = f"Local model not found at: {self.model_path}. Please run download_model.py first."
            print(error_msg)
            self.show_notification("Error", error_msg)
            return
        
        # Utilize M1 Pro's MPS acceleration
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("Tokenizer loaded successfully")
            print(f"Tokenizer type: {type(self.tokenizer).__name__}")
            
            # Check config file to identify which model we're actually loading
            try:
                import json
                config_path = os.path.join(self.model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if '_name_or_path' in config:
                        print(f"Actual model: {config['_name_or_path']}")
                    print(f"Model architecture: {config.get('architectures', ['unknown'])[0]}")
                    print(f"Model size: {config.get('d_model', 'unknown')} hidden dimensions")
            except Exception as e:
                print(f"Error reading model config: {e}")
            
            print("Loading model...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use half precision for better performance
                low_cpu_mem_usage=True
            ).to(device)
            print("Model loaded successfully")
            print(f"Model type: {type(self.model).__name__}")
            
            # Basic test to see if model works
            print("Performing basic model test")
            test_input = "This is a incorrect sentence."
            formatted_input = f"{self.prefix}{test_input}"
            test_encoded = self.tokenizer(formatted_input, return_tensors="pt").to(device)
            test_output = self.model.generate(
                test_encoded.input_ids, 
                max_length=50,
                num_beams=5  # T5 grammar correction models typically use beam search
            )
            test_decoded = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
            print(f"Test input: '{test_input}'")
            print(f"Test output: '{test_decoded}'")
            
            # Check if the model is actually correcting grammar
            if test_decoded == test_input or test_decoded == formatted_input:
                print("WARNING: Model is not correcting grammar, just echoing input.")
                print("Expected output: 'This is an incorrect sentence.'")
            else:
                print(f"Grammar correction test was successful.")
            
            self.model_loaded = True
            print("------- MODEL LOADING COMPLETED -------")
            self.show_notification("Ready", "Model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            print(f"Error loading model: {e}")
            print("------- MODEL LOADING FAILED -------")
            self.show_notification("Error", error_msg)
    
    @rumps.clicked("Correct Clipboard (⌘+`)")
    def manual_correct(self, _):
        """Menu item to correct clipboard"""
        print("Menu item 'Correct Clipboard' clicked")
        self.correct_clipboard()
    
    @rumps.clicked("Simple Mode")
    def toggle_simple_mode(self, sender):
        """Toggle simple mode which doesn't preserve clipboard"""
        self.simple_mode = not self.simple_mode
        sender.state = self.simple_mode
        mode = "Simple mode" if self.simple_mode else "Standard mode"
        print(f"Switched to {mode}")
        
        if self.simple_mode:
            self.show_notification("Mode Changed", "Simple mode enabled (less error-prone but won't preserve clipboard)")
        else:
            self.show_notification("Mode Changed", "Standard mode enabled")
    
    @rumps.clicked("Minimal Corrections")
    def toggle_minimal_corrections(self, sender):
        self.minimal_corrections = not self.minimal_corrections
        sender.state = self.minimal_corrections
        mode = "Minimal corrections" if self.minimal_corrections else "Standard corrections"
        print(f"Switched to {mode} mode")
        self.show_notification("Mode Changed", f"Now using {mode} mode")
    
    @rumps.clicked("Settings")
    def settings(self, _):
        print("Settings menu item clicked")
        # Settings window with more options
        window = rumps.Window(
            message="Configure correction behavior:",
            title="Text Corrector Settings",
            dimensions=(400, 100)
        )
        
        # Prepopulate with current prefix
        window.default_text = self.prefix
        
        # Add instructions
        window.add_button("Cancel")
        window.add_button("Save")
        
        response = window.run()
        if response.clicked and response.clicked != "Cancel":
            self.prefix = response.text.strip()
            print(f"Custom prefix updated: {self.prefix}")
            
            # Add instruction options via menu
            correction_style = rumps.Window(
                message="Choose your correction style:",
                title="Correction Style",
                ok="Apply",
                cancel="Skip"
            ).run()
            
            if correction_style.clicked == "Apply":
                self.show_notification("Settings Updated", "Your custom correction settings have been applied")
        
if __name__ == "__main__":
    print("Starting Text Corrector Application")
    app = TextCorrectorApp()
    print("App created, starting main loop...")
    app.run()
