# 🎓 AI Certificate Caption Generator

> Transform certificates into professional LinkedIn posts using **Mistral 7B AI**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![AI](https://img.shields.io/badge/AI-Mistral_7B-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

### 🤖 **AI-Powered Generation**
- **Mistral 7B Q4** integration via Ollama API
- Generation time: **6-8 seconds** (GPU-accelerated)
- High-quality, human-like captions
- Microservices architecture (Frontend ↔ Backend)

### 🔍 **Smart OCR Engine**
- **Dual OCR**: PyTesseract + EasyOCR
- Intelligent data extraction:
  - Certificate title & organization
  - Issue date & skills
  - Industry classification
  - Certificate type detection

### 🎨 **Customization Options**
- **5 Tone Styles**: Professional, Enthusiastic, Humble, Confident, Casual
- **4 Platforms**: LinkedIn, Twitter, Instagram, Facebook
- **3 Length Options**: Short (100w), Medium (150w), Long (200w)
- **Emoji Control**: None → Minimal → Moderate → Enthusiastic
- **Custom Messages**: Add personal notes
- **Smart Hashtags**: Auto-generated relevant tags

### 💎 **Beautiful UI**
- Futuristic dark theme with glass-morphism
- Real-time generation progress
- Responsive design
- One-click copy & download

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** ([Installation Guide](#tesseract-installation))
3. **Ollama** with Mistral 7B ([Installation Guide](#ollama-installation))

### Installation

```bash
# 1. Clone repository
git clone https://github.com/thesoham2203/Caption-Generator.git
cd Caption-Generator

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# 4. Pull Mistral 7B model
ollama pull mistral:7b-instruct-q4_K_M

# 5. Run the app
streamlit run streamlit_app_new.py
```

The app will open at `http://localhost:8501`

---

## 📖 How to Use

### Method 1: Upload Certificate

1. **Upload** your certificate (PNG, JPG, or PDF)
2. **Customize** caption settings in sidebar:
   - Select tone (Professional/Enthusiastic/etc.)
   - Choose platform (LinkedIn/Twitter/etc.)
   - Set length and emoji style
   - Add custom message (optional)
3. **Click** "Generate AI Caption"
4. **Copy** or **Download** your caption!

### Method 2: Manual Input

1. Expand "📝 Or Enter Details Manually"
2. Enter:
   - Certificate title
   - Organization name
   - Skills (comma-separated)
   - Date (optional)
3. Check "Use manual input instead of OCR"
4. Generate!

---

## 🎯 Example Output

**Input:** Python Data Science Certification from Coursera

**Generated Caption (Professional, Medium):**
```
I'm pleased to announce the successful completion of my Python Data Science
Specialization from Coursera. This comprehensive program has significantly
enhanced my analytical capabilities and technical proficiency.

Throughout this certification, I developed expertise in data manipulation
with Pandas and NumPy, statistical analysis, machine learning algorithms,
and data visualization using Matplotlib and Seaborn. The hands-on projects
provided valuable experience in solving real-world data challenges.

These skills will be instrumental in driving data-informed decisions and
creating impactful analytical solutions in my professional work.

#DataScience #Python #MachineLearning #Analytics #ProfessionalDevelopment
#LifelongLearning #CareerGrowth
```

---

## ⚙️ Technical Architecture

```
┌─────────────────────────────────────┐
│   STREAMLIT FRONTEND (Port 8501)    │
│                                     │
│  ┌──────────────┐  ┌──────────────┐│
│  │  OCR Engine  │  │ Data Extract ││
│  │  Dual-mode   │  │  Smart NLP   ││
│  └──────────────┘  └──────────────┘│
│           │              │          │
│           └──────┬───────┘          │
│                  ↓                  │
│        User Customization           │
│        (Tone, Length, Style)        │
│                  ↓                  │
│          API Call (POST)            │
└──────────────────┼──────────────────┘
                   ↓
         ┌──────────────────┐
         │   OLLAMA API     │
         │   (Port 11434)   │
         │                  │
         │  Mistral 7B Q4   │
         │  (RTX 3050 GPU)  │
         └──────────────────┘
                   ↓
           LinkedIn Caption
```

**Key Design Decisions:**
- **Separation of Concerns**: Frontend (Streamlit) ↔ Backend (Ollama)
- **GPU Optimization**: Mistral 7B Q4 quantized for 6GB VRAM
- **Fallback Logic**: Manual input if OCR fails
- **Caching**: Ollama keeps model loaded for fast subsequent requests

---

## 🔧 Configuration

### Tesseract Installation

**Windows:**
1. Download: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to `C:\Program Files\Tesseract-OCR`
3. Add to PATH (usually automatic)

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Ollama Installation

**Windows:**
1. Download: [Ollama for Windows](https://ollama.ai/download/windows)
2. Run installer
3. Verify: `ollama list`

**Linux/macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Pull Mistral Model:**
```bash
ollama pull mistral:7b-instruct-q4_K_M
```

**Verify Installation:**
```bash
ollama list
# Should show: mistral:7b-instruct-q4_K_M
```

---

## 💻 Hardware Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB free
- **GPU**: Optional (runs on CPU)

### Recommended (for best performance)
- **CPU**: 6+ cores
- **RAM**: 16GB
- **Storage**: 10GB free
- **GPU**: NVIDIA RTX 3050 (6GB) or better
  - With GPU: ~6-8 seconds per caption
  - With CPU only: ~15-20 seconds per caption

---

## 📊 Performance Benchmarks

| Hardware | Generation Time | VRAM Usage |
|----------|----------------|------------|
| RTX 3050 6GB | 6-8 seconds | ~4GB |
| RTX 3060 12GB | 4-6 seconds | ~4GB |
| CPU only (i7) | 15-20 seconds | N/A |
| M1 Mac | 8-10 seconds | ~4GB |

---

## 🎨 Customization Options

### Tone Styles
- **Professional**: Formal, business-appropriate language
- **Enthusiastic**: Energetic with celebration
- **Humble**: Modest and grateful tone
- **Confident**: Achievement-focused and assertive
- **Casual**: Friendly and conversational

### Platform Optimization
- **LinkedIn**: Professional focus, career growth
- **Twitter**: Concise, engaging, thread-ready
- **Instagram**: Visual storytelling, aspirational
- **Facebook**: Personal achievement sharing

### Caption Length
- **Short** (~100 words): Quick announcements
- **Medium** (~150 words): Balanced detail
- **Long** (~200 words): Comprehensive story

---

## 🛠️ Troubleshooting

### "Mistral model not available"
**Solution:**
```bash
ollama pull mistral:7b-instruct-q4_K_M
```

### OCR extraction fails
**Solutions:**
1. Try uploading a higher quality image
2. Use manual input mode
3. Ensure Tesseract is installed: `tesseract --version`

### Slow generation (>20 seconds)
**Solutions:**
1. Check if Ollama is using GPU: `nvidia-smi` (Windows/Linux)
2. Close other GPU-intensive applications
3. Ensure model is fully downloaded: `ollama list`

### "Could not connect to Ollama"
**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not, start it
ollama serve
```

---

## 📁 Project Structure

```
Caption-Generator/
├── streamlit_app_new.py      # Main application (clean refactored version)
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This file
├── sample_certificates/      # Test certificate images
│   ├── tech_certificate.png
│   ├── datascience_certificate.png
│   └── ...
└── myenv/                    # Virtual environment (not in git)
```

---

## 🎓 Use Cases

### Students
- Share course completions
- Build professional online presence
- Showcase continuous learning

### Professionals
- Announce certifications
- Demonstrate skill development
- Network with industry peers

### Job Seekers
- Highlight qualifications
- Stand out to recruiters
- Build credibility

---

## 🚀 Roadmap

- [ ] Batch processing (multiple certificates)
- [ ] Caption templates library
- [ ] Multi-language support
- [ ] Browser extension
- [ ] Mobile app
- [ ] API endpoint for integration

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Soham Penshanwar**
- GitHub: [@thesoham2203](https://github.com/thesoham2203)
- LinkedIn: [Your LinkedIn Profile]

---

## 🙏 Acknowledgments

- **Mistral AI** for the excellent 7B model
- **Ollama** for local LLM infrastructure
- **Streamlit** for the amazing framework
- **Tesseract** for OCR capabilities

---

## 📞 Support

Having issues? 

1. Check the [Troubleshooting](#troubleshooting) section
2. Open an [Issue](https://github.com/thesoham2203/Caption-Generator/issues)
3. Star ⭐ the repo if you find it useful!

---

<p align="center">
  <strong>Built with ❤️ using Streamlit & Mistral 7B AI</strong>
</p>

<p align="center">
  <a href="#-quick-start">Get Started</a> •
  <a href="#-how-to-use">Usage Guide</a> •
  <a href="#-troubleshooting">Help</a>
</p>
