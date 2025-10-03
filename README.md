<div align="center">

# 🎓 AI Certificate Caption Generator

### Transform certificates into professional social media posts in **6-8 seconds**

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Mistral](https://img.shields.io/badge/Mistral_7B-AI_Powered-7C3AED?style=for-the-badge)](https://mistral.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Powered by Mistral 7B AI • Smart Dual OCR • 15+ Customization Options**

[🚀 Quick Start](#-quick-start) • [✨ Features](#-features) • [📸 Demo](#-demo) • [🔧 Troubleshooting](#-troubleshooting)

---

</div>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 **AI-Powered**

- Mistral 7B Q4 (6-8 sec generation)
- Natural, human-like captions
- GPU-accelerated processing
- 15+ customization options

### 🎨 **Full Control**

- 5 Tone styles (Professional → Casual)
- 4 Platform formats (LinkedIn/Twitter/etc)
- 3 Length options (100/150/200 words)
- Emoji control (None → Enthusiastic)

</td>
<td width="50%">

### 🔍 **Smart OCR**

- Dual engine (PyTesseract + EasyOCR)
- PDF support (text + image-based)
- Auto data extraction
- 25+ error scenarios handled

### 💎 **Beautiful UI**

- Futuristic dark theme
- Glass-morphism design
- One-click copy/download
- Real-time progress

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
# 1. Clone & Install
git clone https://github.com/thesoham2203/Caption-Generator.git
cd Caption-Generator
pip install -r requirements.txt

# 2. Setup AI Model (one-time)
# Download Ollama: https://ollama.ai/download
ollama pull mistral:7b-instruct-q4_K_M

# 3. Run
streamlit run streamlit_app.py
```

**That's it!** App opens at `http://localhost:8501` 🎉

> **Note:** Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) for image/PDF processing

---

## 📸 Demo

### Input

Upload certificate → Choose settings → Click "Generate"

### Output

```
I'm pleased to announce the successful completion of my Python Data Science
Specialization from Coursera. This comprehensive program enhanced my analytical
capabilities and technical proficiency in data manipulation, statistical analysis,
machine learning algorithms, and visualization.

These skills will be instrumental in driving data-informed decisions and creating
impactful analytical solutions in my professional work.

#DataScience #Python #MachineLearning #Analytics #ProfessionalDevelopment
```

**Generation Time:** 6-8 seconds ⚡

---

## ⚙️ How It Works

```
📤 Upload Certificate (PNG/JPG/PDF)
          ↓
🔍 Smart OCR (Dual Engine: PyTesseract + EasyOCR)
          ↓
📊 Data Extraction (Title, Org, Skills, Industry)
          ↓
🎨 Customize (Tone, Platform, Length, Emojis)
          ↓
🤖 Mistral 7B AI (Generate Caption)
          ↓
✅ Ready Caption (Copy/Download)
```

**Architecture:** Streamlit Frontend ↔ Ollama API (Port 11434) ↔ Mistral 7B Q4

---

## 💻 System Requirements

| Component           | Minimum          | Recommended   |
| ------------------- | ---------------- | ------------- |
| **Python**          | 3.8+             | 3.10+         |
| **RAM**             | 8GB              | 16GB          |
| **Storage**         | 5GB              | 10GB          |
| **GPU**             | None (CPU works) | RTX 3050 6GB+ |
| **Generation Time** | 15-20s (CPU)     | 6-8s (GPU)    |

---

## 🎨 Customization

<details>
<summary><b>📝 5 Tone Styles</b></summary>

- **Professional** - Formal, business-appropriate
- **Enthusiastic** - Energetic with celebration
- **Humble** - Modest and grateful
- **Confident** - Achievement-focused
- **Casual** - Friendly and conversational

</details>

<details>
<summary><b>📱 4 Platform Formats</b></summary>

- **LinkedIn** - Professional, career growth focus
- **Twitter** - Concise, engaging, thread-ready
- **Instagram** - Visual storytelling, aspirational
- **Facebook** - Personal achievement sharing

</details>

<details>
<summary><b>📏 3 Length Options</b></summary>

- **Short** (~100 words) - Quick announcements
- **Medium** (~150 words) - Balanced detail
- **Long** (~200 words) - Comprehensive story

</details>

---

## � Troubleshooting

<details>
<summary><b>❌ "Mistral model not available"</b></summary>

```bash
ollama pull mistral:7b-instruct-q4_K_M
ollama list  # Verify installation
```

</details>

<details>
<summary><b>❌ "OCR extraction failed"</b></summary>

1. Upload higher quality image (300+ DPI)
2. Use manual input mode (fallback option)
3. Check Tesseract: `tesseract --version`
4. See ERROR_HANDLING_GUIDE.md for detailed solutions

</details>

<details>
<summary><b>⏱️ Slow generation (>20s)</b></summary>

1. Check GPU usage: `nvidia-smi`
2. Close other GPU apps (games, video editors)
3. Verify model downloaded: `ollama list`
4. Restart Ollama: `ollama serve`

</details>

<details>
<summary><b>🔌 "Could not connect to Ollama"</b></summary>

```bash
# Start Ollama service
ollama serve

# Test connection
ollama list
```

</details>

> **💡 Tip:** The app includes 25+ error scenarios with detailed solutions. Check the sidebar's troubleshooting section!

---

## 📁 Project Structure

```
Caption-Generator/
├── streamlit_app.py              # Main application
├── requirements.txt              # Dependencies (7 packages)
├── README.md                     # This file
├── FINAL_GUIDE.md               # Complete usage guide
├── ERROR_HANDLING_GUIDE.md      # Troubleshooting reference
└── .gitignore                   # Git exclusions
```

---

## � Use Cases

| User Type            | Benefits                                               |
| -------------------- | ------------------------------------------------------ |
| 🎓 **Students**      | Share completions, build presence, showcase learning   |
| 💼 **Professionals** | Announce certifications, demonstrate growth, network   |
| 🔍 **Job Seekers**   | Highlight qualifications, stand out, build credibility |

---

## 🤝 Contributing

Contributions welcome! Fork → Create branch → Commit → Push → Open PR

---

## Author

**Soham Penshanwar**  
[![GitHub](https://img.shields.io/badge/GitHub-thesoham2203-181717?style=flat&logo=github)](https://github.com/thesoham2203)

---

## 🙏 Credits

Built with: [Mistral AI](https://mistral.ai/) • [Ollama](https://ollama.ai/) • [Streamlit](https://streamlit.io/) • [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## � License

MIT License - see [LICENSE](LICENSE) file

---

<div align="center">

### ⭐ Star this repo if you find it useful!

**Built with ❤️ using Streamlit & Mistral 7B AI**

[🚀 Get Started](#-quick-start) • [📖 Documentation](FINAL_GUIDE.md) • [🔧 Troubleshooting](#-troubleshooting) • [❓ Issues](https://github.com/thesoham2203/Caption-Generator/issues)

</div>
