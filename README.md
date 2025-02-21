# Multilingual On-Device Translator

**Welcome to the future of translation!**  
This project harnesses the power of on-device AI to provide real-time, robust translations between English, Japanese, Korean, and Chinese. Our solution not only meets the hackathon criteria but exceeds them by offering state-of-the-art fallback mechanisms, on-device optimization for Snapdragon devices, and a user-friendly interface.

---

## Problem Statement

In today’s global business environment, effective communication across multiple languages is crucial. Traditional translation systems often rely on cloud connectivity, which can lead to latency, privacy issues, and dependency on a stable network connection. Our goal was to develop an on-device translator that:
- **Delivers real-time translations** without relying on external servers.
- **Optimizes performance on Snapdragon/Qualcomm devices** for low latency and efficient memory usage.
- **Handles edge cases gracefully** using robust fallback strategies.
- **Provides a multilingual user interface** to serve users worldwide.

---

## Technical Overview

Our translation engine supports 12 inter-language pairs with robust fallback logic:
- **Primary Translation:**  
  Uses exported ONNX models derived from MarianMTModel (optimized with int8 quantization for on-device efficiency).
- **Fallback Mechanism:**  
  If an ONNX model for a specific translation pair is missing or produces unstable output, our system automatically falls back to using the M2M100 model.
- **Optimizations:**  
  - **Caching:** Heavy resources (e.g., ONNX models and tokenizers) are cached to reduce load times.
  - **Parallel Processing:** Multi-sentence inputs are processed concurrently to improve efficiency.
  - **Safety Nets:** Extensive post-processing is implemented to clean up decoding errors (e.g., excessive token repetition).

These strategies ensure our system is both innovative and resilient, meeting the highest standards expected by the judges.

---

## System Dependencies

Before installing Python dependencies, please ensure the following system packages are installed. This is required to build certain components (e.g., SentencePiece):

```bash
sudo apt-get update
sudo apt-get install cmake pkg-config libsentencepiece-dev
```
These commands install CMake, pkg-config, and the SentencePiece development files (including sentencepiece.pc), ensuring a smooth build process during pip install -r requirements.txt.

---

## ONNX Model Export and Fallback Setup

Our translation engine uses exported ONNX models for primary translation routes. **Before running the application, please complete the following steps:**

1. **Export the ONNX Models:**  
   Run the provided MarianMTModel export script (e.g., `export_marian_models.py`) to generate the ONNX files (such as `marian_ja-en.onnx`, `marian_zh-en.onnx`, etc.) in the `models/` directory.
   ```bash
   python export_marian_models.py

2. **Fallback Behavior:**  
If a specific ONNX model is not available (for example, if the export has not been run), the system will automatically fall back to using the M2M100 model.
**Note:** The M2M100 model will be downloaded on the first run, which might also take some time.

---

## Installation

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/mneang/snapdragon-ai.git
   cd snapdragon-ai
  
2. **Install Python Dependencies:**  
  ```bash
  pip install -r requirements.txt
  ```
**Note:** The M2M100 model will be downloaded on the first run, which might also take some time.

3. **Export the ONNX Models:**  
Run the model export script as described above.

---

## Running the Application

You can run the application using Streamlit. In your terminal or GitHub Codespaces, execute:
```bash
streamlit run app.py
```
**Note**: For demo purposes, we recommend running this in GitHub Codespaces on the virtual Snapdragon device (1000 minutes of access). 

---

## Usage Instructions

### Language Options:
- The app supports **English, Japanese, Korean, and Chinese**.  
  Select your source language and target language from the sidebar.
  - **English** is used for clear instructions.
  - **Japanese, Korean, and Chinese** options ensure the app is accessible for users in those regions.

### Input Modes:
- **Text Input:**  
  Simply type your text into the provided text area.
- **Voice Input (Beta):**  
  If you select voice input, a sample audio file (`output.mp3`) will play, and the app will automatically transcribe it using the Faster-Whisper model. You can then edit the transcription if needed before translating.

### Translation Process:
After entering your text, click the **"▶️ Translate"** button. The app will display processing time and memory usage, demonstrating the on-device efficiency.  
**Note:** Longer sentences will take more time to process translations.

### Model Loading Note:
Please be aware that the fallback translator (using M2M100) and the ONNX model loading may take extra time on the first run. Refreshing the page after the initial load can help ensure smoother performance.

---

## Deployment

Due to the large size of our ONNX models, deploying to services like Heroku or Streamlit Community Cloud can be challenging. We recommend:
- **Running the App via GitHub Codespaces:**  
  This allows you to access the app on a virtual Snapdragon device without encountering file size limits.
- **Alternatively:**  
  Host the ONNX models externally (e.g., on the Hugging Face Hub) and modify the application to download them at runtime.

---

## Future Improvements and Caveats/Assumptions

While our Multilingual On-Device Translator is a robust and innovative solution, we recognize that there are areas for further refinement. Our commitment to continuous improvement is reflected in the following points:

- **Enhanced Model Optimization:**  
  We plan to further optimize our ONNX models by exploring advanced quantization techniques and leveraging Qualcomm’s SNPE or TensorRT, reducing latency and memory usage even more on Snapdragon devices.

- **Expanded Language Support:**  
  Future updates will include support for additional languages and dialects, broadening the accessibility of our translation engine to a wider global audience.

- **Improved Fallback Mechanisms:**  
  We are actively refining our fallback translators (e.g., M2M100) to minimize edge cases such as repetitive token output. Ongoing adjustments to decoding parameters and post-processing routines will enhance overall translation quality.

- **Voice Input Integration:**  
  Although currently in beta, we intend to fully integrate live voice transcription using the Faster-Whisper model, ensuring seamless end-to-end voice-to-translation functionality.

- **User Experience Enhancements:**  
  Further refinements to the UI/UX—including more intuitive multilingual prompts, mobile optimization, and detailed status feedback—are planned to improve usability and overall user satisfaction.

- **Assumptions and Limitations:**  
  - Our current system assumes a stable network connection during the initial model downloads and fallback downloads.  
  - The fallback mechanism is designed to handle rare edge cases; however, we acknowledge that certain phrasings may still produce minor decoding artifacts, which will be addressed in future updates.

We believe these enhancements will not only solidify our competitive edge but also demonstrate our commitment to ongoing innovation and impact in the field of on-device AI translation.
