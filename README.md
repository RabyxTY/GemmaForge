# 🎬 GemmaForge: AI Re-Voicer & Video Generator

GemmaForge is an interactive generative media pipeline that allows you to automate video narration, parodies, and complete story-to-video generation using **Google Gemma-2-9b-it** and **Fireworks AI** platform.

This project was built for the **AMD Developer Hackathon: ACT II** (Track 2: Video Captioning / Open track).

---

## ✨ Features
1.  **🔄 Video Re-Voicing (Ре-озвучка)**: 
    Upload an existing video, automatically extract its audio transcript (via Fireworks Whisper API) and analyze visual cues (via Fireworks Multimodal Qwen-VL API). Gemma then drafts a brand new script based on your custom style sliders (Sarcasm, Humor, IT slang, Formal tone). The script is voiced using local high-quality TTS (`edge-tts`) and compiled back to MP4 overlaying the original video track using FFmpeg.
2.  **✨ Story-to-Video (Генерация с нуля)**:
    Input a simple story prompt (e.g., *"A cat trying to deploy to production on Friday night"*), select a format (**9:16 vertical** for TikTok/Reels or **16:9 horizontal**), adjust style weights, and watch Gemma act as a director: creating narration, generating images with `FLUX.1 [schnell]`, and stitching it all into a smooth, voiced MP4 video.

---

## 🛠️ System Prerequisites
Since the project manipulates media files, you must have **FFmpeg** installed on your system.

*   **macOS**: `brew install ffmpeg`
*   **Linux**: `sudo apt-get install ffmpeg`
*   **Windows**: Download via [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

---

## 🚀 Setup & Installation (Local Execution)

1.  **Clone the repository and go to the project directory:**
    ```bash
    cd hackathon-project
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment:**
    Copy `.env.example` to `.env` and fill in your Fireworks API key:
    ```bash
    FIREWORKS_API_KEY=fw_your_api_key_here
    ```
5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 🐳 Docker Deployment (Containerized)
The hackathon submission requires containerization. You can easily build and run the project inside Docker:

1.  **Build the Docker image:**
    ```bash
    docker build -t gemmaforge .
    ```
2.  **Run the container:**
    ```bash
    docker run -p 8501:8501 --env FIREWORKS_API_KEY=fw_your_api_key_here gemmaforge
    ```
3.  Access the application at `http://localhost:8501`.

---

## 📄 License
This project is licensed under the **Apache License 2.0** (required by Lablab.ai hackathon rules).
