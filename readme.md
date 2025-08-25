# Voice Matcher - Audio Forensics 🎙️🔊

An AI-powered voice comparison app built with **Streamlit** and **SpeechBrain**.
Deployed version: https://voice-matcher.streamlit.app/

## 🚀 Features

- Upload suspect & evidence voice samples
- AI-based similarity analysis (ECAPA-TDNN)
- Confidence scoring (Strong Match / Possible / No Match)
- Visualization with Plotly
- Downloadable analysis report (In testing)

## 🛠 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/voice-matcher.git
cd voice-matcher
pip install -r requirements.txt
```

Note: This project requires **FFmpeg** for audio processing if you're running it locally.

### Windows

1. Download FFmpeg from [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract and add the `bin` folder (with ffmpeg.exe & ffprobe.exe) to your PATH.

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install ffmpeg
```
