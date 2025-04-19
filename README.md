# Lessons learned developing this

- Tried to implement it with a shortcut (Cmd + Control + C) - the Python way of working with MacOS keyboard permissions was less than ideal and I'll probably rewrite it to Swift one day, since it appears to have much more straightforward access policies.
- Choosing the model took some time - the Grammarly's was the first one that was actually reliable, even though it needed fine-tuning (it kept adding a lot of stuff like emoticons)
- The .cursorrules created here will probably be a good baseline for future ML projects
- It also required some polishing of the prompt - too harsh and the answers were empty, too lax and there were no corrections
- Paid models like Gemini Pro Max (0.05$ per prompt) actually weren't much more helpful, and they actually generated "test" scripts that didn't even compile :) a dollar terribly spent

# Text Corrector

A macOS menubar application that corrects English text grammar on the fly using AI language models. Simply copy text, click the menubar icon to correct it, and paste the corrected version.

## Features

- Grammar correction using Grammarly's coedit-large model
- Accessible from the menu bar
- No internet required - runs completely offline
- Preserves clipboard content
- Simple mode for straightforward corrections
- Works across all macOS applications
- Minimal RAM and CPU usage when idle
- Runs natively on Apple Silicon (M1/M2/M3)

## Usage

1. Copy text from any application
2. Click the checkmark icon (✓) in the menu bar and select "Correct Clipboard"
3. Paste the corrected text where needed
4. Receive notifications indicating if text was corrected

### Menu Options

- **Correct Clipboard**: Manually trigger correction
- **Minimal Corrections**: Toggle between minimal and standard correction modes
- **Simple Mode**: Toggle clipboard preservation mode
- **Settings**: Configure the application behavior

## How It Works

The application uses a local language model to correct grammar and improve text readability. The model is loaded into memory when the app starts and runs locally on your machine without sending any data to external servers.

## Troubleshooting

- **Model Loading Issues**: If the model fails to load, try running `python3 download_model.py` again to re-download it
- **Notifications Not Showing**: Check macOS notification settings for the app

## Installation

### Option 1: Download the pre-built app

1. Download the latest release from the [Releases](https://github.com/ppuzio/text-corrector/releases) page
2. Unzip the downloaded file
3. Move Text Corrector.app to your Applications folder
4. Open the app (right-click and select "Open" the first time to bypass Gatekeeper)
5. Grant the necessary permissions when prompted

### Option 2: Build from source

1. Clone this repository:

   ```
   git clone https://github.com/ppuzio/text-corrector.git
   cd text-corrector
   ```

2. Create and activate a virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Download the model:

   ```
   python3 download_model.py
   ```

5. Run the application:

   ```
   python3 app.py
   ```

6. (Optional) Build a standalone app:
   ```
   python3 setup.py py2app
   ```
   and run it using
   ```
   ./dist/TextCorrector.app/Contents/MacOS/TextCorrector
   ```
   The app will be available in the `dist` folder.

## Development

- Python 3.9+
- Dependencies listed in requirements.txt
- macOS 10.15+ supported

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Grammarly for their open-source language models
- The Hugging Face team for transformers library
- Apple for MPS acceleration on Apple Silicon
