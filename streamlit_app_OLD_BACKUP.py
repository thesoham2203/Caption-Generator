"""
Enhanced Certificate Caption Generator - Streamlit App
Production-ready web interface with beautiful UI
"""

import streamlit as st
import os
import re
import io
import json
import tempfile
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging

# Core libraries
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# AI MODEL IMPORTS (Conditional - will check availability)
# ============================================================
BLIP_AVAILABLE = False
OLLAMA_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_AVAILABLE = True
    logger.info("✅ BLIP-2 models available")
except ImportError:
    logger.warning("⚠️ BLIP models not available. Install: pip install transformers torch")

try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("✅ Ollama client available")
except ImportError:
    logger.warning("⚠️ Ollama not available. Install: pip install ollama")

# ============================================================
# POPPLER CONFIGURATION
# ============================================================
# If you have Poppler installed, set the path to the bin folder here
# Example for Windows: r"C:\Program Files\poppler-xx.xx.x\Library\bin"
# Leave as None to use system PATH
POPPLER_PATH = None  # Set this to your Poppler bin path if needed

# Check if Poppler is available
POPPLER_AVAILABLE = False
try:
    from pdf2image.exceptions import PDFInfoNotInstalledError
    if POPPLER_PATH:
        os.environ['PATH'] = POPPLER_PATH + os.pathsep + os.environ.get('PATH', '')
    # Test if poppler works
    import subprocess
    subprocess.run(['pdftoppm', '-v'], capture_output=True, check=False)
    POPPLER_AVAILABLE = True
except:
    pass

# Try EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Certificate Caption Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic, professional UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - Dark futuristic theme */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }
    
    /* Animated gradient background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass-morphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        text-align: center;
        padding: 2rem 0 1rem 0;
        font-size: 3.5rem !important;
        letter-spacing: -2px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
    }
    
    h2 {
        color: #b8b8ff;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #a0a0ff;
        font-weight: 500;
        font-size: 1.3rem !important;
    }
    
    /* Futuristic buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        width: 100%;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        padding: 3rem 2rem;
        border-radius: 20px;
        border: 2px dashed rgba(102, 126, 234, 0.5);
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s;
    }
    
    .upload-section:hover {
        border-color: rgba(102, 126, 234, 0.8);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Caption box - Neon glow effect */
    .caption-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2),
                    inset 0 0 20px rgba(102, 126, 234, 0.1);
        transition: all 0.3s;
    }
    
    .caption-box:hover {
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4),
                    inset 0 0 30px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
    }
    
    /* Metrics cards - Futuristic stats */
    .metrics-card {
        background: rgba(102, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        transition: all 0.3s;
    }
    
    .metrics-card:hover {
        background: rgba(102, 126, 234, 0.15);
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success/Info messages */
    .success-message {
        background: rgba(76, 175, 80, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #90ee90;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        backdrop-filter: blur(15px);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #b8b8ff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #e0e0ff !important;
        font-weight: 500;
    }
    
    /* Input fields - Futuristic glow */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 0.8rem !important;
        transition: all 0.3s !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>select:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.4) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.7);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Expander - Glass effect */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        color: #e0e0ff !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 0 0 10px 10px !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background: rgba(102, 126, 234, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border-left: 4px solid #667eea !important;
        color: #e0e0ff !important;
    }
    
    /* Skill badges */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 1px solid rgba(102, 126, 234, 0.4);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        color: #b8b8ff;
        font-weight: 500;
        font-size: 0.9rem;
        backdrop-filter: blur(5px);
        transition: all 0.3s;
    }
    
    .skill-badge:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .stSpinner > div {
        border-color: #667eea !important;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Neon glow text effect */
    .neon-text {
        color: #fff;
        text-shadow: 0 0 10px #667eea,
                     0 0 20px #667eea,
                     0 0 30px #667eea,
                     0 0 40px #764ba2;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Column divider */
    .st-emotion-cache-1r4qj8v {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 20px;
        padding: 2rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Data Classes
@dataclass
class CertificateAnalysis:
    """Data class for certificate analysis results"""
    title: str = ""
    organization: str = ""
    recipient_name: str = ""
    completion_status: str = ""
    skills_covered: List[str] = None
    duration: str = ""
    date_issued: str = ""
    certificate_type: str = ""
    confidence_score: float = 0.0
    industry: str = ""
    
    def __post_init__(self):
        if self.skills_covered is None:
            self.skills_covered = []

@dataclass
class CaptionTemplate:
    """Data class for caption templates"""
    name: str
    style: str
    opening: List[str]
    achievement_templates: Dict[str, str]
    value_statements: List[str]
    call_to_actions: List[str]
    hashtag_style: str

# ============================================================
# AI MODEL CLASSES - Multi-Modal Pipeline
# ============================================================

class BLIPCaptionGenerator:
    """BLIP-2 Vision-Language Model for Image Understanding"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if BLIP_AVAILABLE and torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load BLIP model (lazy loading)"""
        if not BLIP_AVAILABLE:
            logger.warning("BLIP not available. Install transformers and torch.")
            return
        
        try:
            logger.info("🔄 Loading BLIP-2 model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            logger.info(f"✅ BLIP-2 loaded on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load BLIP: {e}")
            self.model = None
    
    def generate_caption(self, image: Image.Image, max_length: int = 100) -> Optional[str]:
        """Generate caption from certificate image"""
        if self.model is None:
            return None
        
        try:
            # Preprocess image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,
                    temperature=0.7
                )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"BLIP caption: {caption}")
            return caption
            
        except Exception as e:
            logger.error(f"BLIP generation error: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if BLIP is available"""
        return self.model is not None


class OllamaLLMGenerator:
    """LLaMA 3.2 via Ollama for Caption Refinement"""
    
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Ollama server is running"""
        if not OLLAMA_AVAILABLE:
            return False
        
        try:
            # Test connection
            response = ollama.list()
            logger.info(f"✅ Ollama connected. Available models: {len(response.get('models', []))}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Ollama not running: {e}")
            return False
    
    def generate_caption(self, prompt: str, context: Dict = None) -> Optional[str]:
        """Generate refined caption using LLaMA"""
        if not self.available:
            return None
        
        try:
            # Build prompt with context
            full_prompt = self._build_prompt(prompt, context)
            
            # Call Ollama API
            response = ollama.generate(
                model=self.model_name,
                prompt=full_prompt,
                options={
                    'temperature': 0.7,
                    'max_tokens': 300,
                    'top_p': 0.9
                }
            )
            
            caption = response.get('response', '').strip()
            logger.info(f"LLaMA caption generated: {len(caption)} chars")
            return caption
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return None
    
    def _build_prompt(self, base_prompt: str, context: Dict = None) -> str:
        """Build enriched prompt for LLaMA"""
        prompt_parts = [
            "You are a professional LinkedIn caption writer.",
            "Generate an engaging, authentic LinkedIn post for this certificate achievement.",
            "",
            f"Context: {base_prompt}",
        ]
        
        if context:
            if context.get('title'):
                prompt_parts.append(f"Certificate: {context['title']}")
            if context.get('organization'):
                prompt_parts.append(f"From: {context['organization']}")
            if context.get('skills'):
                prompt_parts.append(f"Skills: {', '.join(context['skills'][:5])}")
            if context.get('industry'):
                prompt_parts.append(f"Industry: {context['industry']}")
        
        prompt_parts.extend([
            "",
            "Requirements:",
            "- Professional but authentic tone",
            "- 2-3 paragraphs",
            "- Highlight the achievement and skills",
            "- End with relevant hashtags (5-7)",
            "- Make it personal and engaging",
            "",
            "Caption:"
        ])
        
        return "\n".join(prompt_parts)
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        return self.available


class HybridCaptionGenerator:
    """Multi-Modal Pipeline: BLIP + OCR + LLaMA"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.blip = BLIPCaptionGenerator() if BLIP_AVAILABLE else None
        self.llama = OllamaLLMGenerator() if OLLAMA_AVAILABLE else None
        
        # Log available modes
        modes = []
        if self.blip and self.blip.is_available():
            modes.append("BLIP-2")
        if self.llama and self.llama.is_available():
            modes.append("LLaMA")
        
        logger.info(f"🤖 AI Modes available: {', '.join(modes) if modes else 'None (Template-only)'}")
    
    def generate(self, image: Image.Image, analysis: CertificateAnalysis, 
                 mode: str = "hybrid") -> Tuple[str, str]:
        """
        Generate caption using selected mode
        
        Modes:
        - template: Fast, rule-based (original)
        - blip: BLIP-2 visual understanding
        - llama: LLaMA text generation
        - hybrid: BLIP + OCR + LLaMA (full pipeline)
        
        Returns: (caption, mode_used)
        """
        if mode == "template":
            return self._template_generation(analysis), "template"
        
        elif mode == "blip" and self.blip and self.blip.is_available():
            caption = self._blip_generation(image, analysis)
            return caption, "blip"
        
        elif mode == "llama" and self.llama and self.llama.is_available():
            caption = self._llama_generation(analysis)
            return caption, "llama"
        
        elif mode == "hybrid":
            # Full Multi-Modal Pipeline
            caption = self._hybrid_generation(image, analysis)
            return caption, "hybrid"
        
        else:
            # Fallback to template
            logger.warning(f"Mode '{mode}' not available, falling back to template")
            return self._template_generation(analysis), "template (fallback)"
    
    def _template_generation(self, analysis: CertificateAnalysis) -> str:
        """Original template-based generation"""
        # Use existing CaptionGenerator logic
        parts = []
        parts.append("🎓 Excited to share my latest achievement!\n\n")
        
        if analysis.title:
            parts.append(f"I've successfully completed the {analysis.title}")
            if analysis.organization:
                parts.append(f" from {analysis.organization}!")
            else:
                parts.append("!")
        
        if analysis.skills_covered:
            parts.append(f"\n\n💡 Key skills gained: {', '.join(analysis.skills_covered[:5])}")
        
        parts.append("\n\n#ContinuousLearning #ProfessionalDevelopment")
        
        return "".join(parts)
    
    def _blip_generation(self, image: Image.Image, analysis: CertificateAnalysis) -> str:
        """BLIP-only generation"""
        blip_caption = self.blip.generate_caption(image)
        
        if not blip_caption:
            return self._template_generation(analysis)
        
        # Enhance with structured data
        parts = [f"🎓 {blip_caption}\n"]
        
        if analysis.organization:
            parts.append(f"\nIssued by: {analysis.organization}")
        
        if analysis.skills_covered:
            parts.append(f"\nKey skills: {', '.join(analysis.skills_covered[:5])}")
        
        parts.append("\n\n#Achievement #Learning")
        
        return "".join(parts)
    
    def _llama_generation(self, analysis: CertificateAnalysis) -> str:
        """LLaMA-only generation"""
        context = {
            'title': analysis.title,
            'organization': analysis.organization,
            'skills': analysis.skills_covered,
            'industry': analysis.industry
        }
        
        prompt = f"Certificate achievement: {analysis.title}"
        caption = self.llama.generate_caption(prompt, context)
        
        return caption if caption else self._template_generation(analysis)
    
    def _hybrid_generation(self, image: Image.Image, analysis: CertificateAnalysis) -> str:
        """Full Multi-Modal Pipeline: BLIP + OCR + LLaMA"""
        
        # Step 1: Visual understanding (BLIP)
        blip_caption = None
        if self.blip and self.blip.is_available():
            blip_caption = self.blip.generate_caption(image)
        
        # Step 2: Build enriched context
        context = {
            'title': analysis.title,
            'organization': analysis.organization,
            'skills': analysis.skills_covered,
            'industry': analysis.industry,
            'visual_description': blip_caption
        }
        
        # Step 3: LLaMA refinement
        if self.llama and self.llama.is_available():
            prompt = f"Certificate: {analysis.title} from {analysis.organization}"
            if blip_caption:
                prompt += f". Visual context: {blip_caption}"
            
            caption = self.llama.generate_caption(prompt, context)
            if caption:
                return caption
        
        # Fallback: BLIP or template
        if blip_caption:
            return self._blip_generation(image, analysis)
        
        return self._template_generation(analysis)
    
    def get_available_modes(self) -> List[str]:
        """Get list of available generation modes"""
        modes = ["template"]  # Always available
        
        if self.blip and self.blip.is_available():
            modes.append("blip")
        if self.llama and self.llama.is_available():
            modes.append("llama")
        if self.blip and self.blip.is_available() and self.llama and self.llama.is_available():
            modes.append("hybrid")
        
        return modes

# Core Analyzer Class
class CertificateAnalyzer:
    """Enhanced certificate analyzer with OCR and NLP"""
    
    def __init__(self):
        self.pytesseract_config = r'--oem 3 --psm 6 -l eng'
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], verbose=False)
            except:
                pass
        
        self.certificate_keywords = {
            'completion': ['completed', 'finished', 'successfully completed', 'accomplished', 
                          'certified', 'graduated', 'achieved', 'earned', 'obtained'],
            'course_types': ['course', 'training', 'program', 'workshop', 'certification',
                           'bootcamp', 'seminar', 'webinar', 'masterclass'],
            'organization_indicators': ['organized by', 'conducted by', 'hosted by', 
                                       'presented by', 'issued by', 'from', 'by', 'offered by'],
            'skill_indicators': ['skills', 'learned', 'covered', 'topics', 'subjects', 
                               'modules', 'curriculum', 'competencies'],
            'duration_patterns': [
                r'(\d+)\s*(hour|hr|hours|hrs|week|weeks|wk|wks|month|months|mon|mos|day|days|year|years|yr|yrs)'
            ]
        }
        
        self.industry_hashtags = {
            'technology': ['#TechSkills', '#Programming', '#SoftwareDevelopment', '#Innovation', '#TechCareers'],
            'data_science': ['#DataScience', '#MachineLearning', '#Analytics', '#AI', '#DataAnalytics'],
            'design': ['#Design', '#UXDesign', '#CreativeSkills', '#UserExperience', '#DesignThinking'],
            'business': ['#ProfessionalDevelopment', '#CareerGrowth', '#SkillDevelopment', '#Leadership', '#BusinessSkills', '#EmployabilitySkills'],
            'marketing': ['#DigitalMarketing', '#MarketingStrategy', '#SocialMedia', '#ContentMarketing'],
            'finance': ['#Finance', '#FinTech', '#Investment', '#Accounting', '#FinancialLiteracy'],
            'healthcare': ['#Healthcare', '#MedicalTraining', '#HealthTech', '#HealthcareInnovation'],
            'education': ['#Education', '#Teaching', '#LearningAndDevelopment', '#EdTech', '#ContinuousLearning'],
            'general': ['#ProfessionalDevelopment', '#SkillBuilding', '#CareerGrowth', '#Upskilling']
        }
        
        self._load_caption_templates()
    
    def _load_caption_templates(self):
        """Load caption templates"""
        self.caption_templates = {
            'professional': CaptionTemplate(
                name="Professional",
                style="formal",
                opening=["I'm pleased to share", "I'm proud to announce", "Excited to share"],
                achievement_templates={
                    'course': "I have successfully completed the {title} course{organization_text}.",
                    'workshop': "I participated in the {title} workshop{organization_text}.",
                    'certification': "I have earned certification in {title}{organization_text}.",
                    'program': "I have successfully completed the {title}{organization_text}."
                },
                value_statements=[
                    "This achievement strengthens my professional capabilities and expertise.",
                    "The knowledge gained will be valuable in delivering exceptional results.",
                    "This learning experience enhances my ability to contribute effectively."
                ],
                call_to_actions=[
                    "Looking forward to applying these skills professionally.",
                    "Ready to contribute with enhanced expertise.",
                    "Excited to leverage this knowledge in future endeavors."
                ],
                hashtag_style="professional"
            ),
            'enthusiastic': CaptionTemplate(
                name="Enthusiastic",
                style="casual",
                opening=["Hey LinkedIn! 🎉", "Thrilled to share! 🚀", "Amazing news! ✨"],
                achievement_templates={
                    'course': "🎓 Just crushed the {title} course{organization_text}! 💪",
                    'workshop': "🎯 Had an incredible time at the {title} workshop{organization_text}! 🔥",
                    'certification': "🏆 Officially certified in {title}{organization_text}! 🎊",
                    'program': "🎓 Just completed the {title}{organization_text}! 💪"
                },
                value_statements=[
                    "This journey has been absolutely transformative! 🌟",
                    "Can't wait to put these amazing skills to work! 💡",
                    "Feeling more confident and ready to tackle new challenges! 💪"
                ],
                call_to_actions=[
                    "Bring on the exciting projects! 🚀",
                    "Ready to make some magic happen! ✨",
                    "Let's connect and create something awesome! 🤝"
                ],
                hashtag_style="enthusiastic"
            ),
            'technical': CaptionTemplate(
                name="Technical",
                style="detailed",
                opening=["Technical milestone achieved", "Completed advanced training in"],
                achievement_templates={
                    'course': "Successfully completed comprehensive training in {title}{organization_text}.",
                    'workshop': "Participated in intensive {title} workshop{organization_text}.",
                    'certification': "Achieved professional certification in {title}{organization_text}.",
                    'program': "Successfully completed the {title}{organization_text}."
                },
                value_statements=[
                    "This training provides practical skills directly applicable to complex technical challenges.",
                    "The curriculum covered industry best practices and cutting-edge methodologies.",
                    "Gained hands-on experience with essential tools and frameworks."
                ],
                call_to_actions=[
                    "Ready to implement these methodologies in real-world projects.",
                    "Excited to contribute to technically challenging initiatives.",
                    "Looking forward to collaborating on innovative solutions."
                ],
                hashtag_style="technical"
            )
        }
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            denoised = cv2.fastNlMeansDenoising(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return Image.fromarray(sharpened)
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def extract_text_from_image(self, image_path: str) -> Dict:
        """Extract text using OCR"""
        try:
            image = Image.open(image_path)
            enhanced = self.enhance_image(image)
            
            # PyTesseract
            text = pytesseract.image_to_string(enhanced, config=self.pytesseract_config)
            confidence = 0.75
            
            # EasyOCR if available
            if self.easyocr_reader:
                try:
                    img_array = np.array(enhanced)
                    results = self.easyocr_reader.readtext(img_array, detail=1, paragraph=True)
                    if results:
                        easyocr_text = ' '.join([text for (_, text, conf) in results if conf > 0.3])
                        if len(easyocr_text) > len(text):
                            text = easyocr_text
                            confidence = np.mean([conf for (_, _, conf) in results])
                except:
                    pass
            
            return {
                'text': text,
                'confidence': confidence,
                'engine_used': 'multi-engine'
            }
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'engine_used': 'failed'}
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from PDF"""
        try:
            # Try direct text extraction with PyMuPDF first
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
            
            if text.strip():
                return {'text': text, 'confidence': 0.95, 'method': 'direct'}
            
            # OCR fallback - only if Poppler is available
            if not POPPLER_AVAILABLE:
                logger.warning("Poppler not available. Cannot perform OCR on PDF. Use image files (PNG/JPG) instead.")
                return {
                    'text': '', 
                    'confidence': 0.0, 
                    'method': 'failed',
                    'error': 'Poppler not installed. Please upload an image file (PNG/JPG) or install Poppler for PDF support.'
                }
            
            images = convert_from_path(
                pdf_path, 
                dpi=300,
                poppler_path=POPPLER_PATH
            )
            all_text = ""
            for i, img in enumerate(images):
                temp_path = f"{tempfile.gettempdir()}/page_{i}.png"
                img.save(temp_path)
                result = self.extract_text_from_image(temp_path)
                all_text += result['text'] + "\n"
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return {'text': all_text, 'confidence': 0.8, 'method': 'ocr'}
        except Exception as e:
            error_msg = str(e)
            if 'poppler' in error_msg.lower():
                logger.warning(f"Poppler-related error: {error_msg}")
                return {
                    'text': '', 
                    'confidence': 0.0, 
                    'method': 'failed',
                    'error': 'Poppler not configured correctly. Please use image files (PNG/JPG) instead.'
                }
            logger.error(f"PDF extraction failed: {e}")
            return {'text': '', 'confidence': 0.0, 'method': 'failed', 'error': str(e)}
    
    def detect_industry(self, text: str) -> str:
        """Detect industry from text"""
        text_lower = text.lower()
        industry_keywords = {
            'technology': ['programming', 'coding', 'software', 'python', 'java', 'web', 'app', 'development', 'developer', 'full stack'],
            'data_science': ['data science', 'machine learning', 'analytics', 'ai', 'artificial intelligence', 'data analysis', 'deep learning'],
            'design': ['design', 'ux', 'ui', 'graphic', 'creative', 'figma', 'wireframe'],
            'business': ['business', 'management', 'leadership', 'strategy', 'project management', 'agile', 'scrum', 'employability', 'professional development', 'skill development'],
            'marketing': ['marketing', 'social media', 'seo', 'content', 'digital marketing'],
            'finance': ['finance', 'accounting', 'investment', 'banking', 'fintech'],
            'healthcare': ['healthcare', 'medical', 'nursing', 'health'],
            'education': ['education', 'teaching', 'learning', 'training', 'academic']
        }
        
        scores = {industry: sum(1 for kw in keywords if kw in text_lower) 
                 for industry, keywords in industry_keywords.items()}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
    
    def analyze_certificate(self, text: str, confidence: float) -> CertificateAnalysis:
        """Analyze certificate content"""
        analysis = CertificateAnalysis()
        analysis.confidence_score = confidence
        
        if not text.strip():
            return analysis
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract title
        for line in lines:
            if len(line) > 15 and len(line) < 100:
                if not any(word in line.lower() for word in ['certificate', 'awarded', 'this', 'presented']):
                    analysis.title = line
                    break
        
        # Extract organization
        org_patterns = [
            r'(?:issued by|offered by|from)\s+([A-Za-z\s&,]+(?:University|Institute|College|Academy|Company|Inc))',
        ]
        for pattern in org_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                analysis.organization = matches[0].strip()
                break
        
        # Determine type
        text_lower = text.lower()
        if any(word in text_lower for word in ['workshop', 'seminar', 'webinar']):
            analysis.certificate_type = 'workshop'
        elif any(word in text_lower for word in ['program', 'bootcamp', 'training program']):
            analysis.certificate_type = 'program'
        elif any(word in text_lower for word in ['certification', 'certified', 'professional certificate']):
            analysis.certificate_type = 'certification'
        else:
            analysis.certificate_type = 'course'
        
        # Extract skills with improved logic
        skills = set()
        
        # Look for explicit skill mentions
        skill_patterns = [
            r'skills?:?\s*([^\n]+)',
            r'topics?:?\s*([^\n]+)',
            r'covered:?\s*([^\n]+)',
            r'competencies:?\s*([^\n]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split by commas and common separators
                parts = re.split(r'[,;•|\n]', match)
                for part in parts:
                    part = part.strip()
                    if 3 < len(part) < 40:
                        skills.add(part.title())
        
        # Use TextBlob as fallback
        if len(skills) < 3:
            try:
                blob = TextBlob(text)
                for phrase in blob.noun_phrases:
                    phrase = str(phrase)  # Convert to string
                    if 3 < len(phrase) < 30 and 1 <= len(phrase.split()) <= 3:
                        # Filter out common non-skill phrases
                        if not any(x in phrase.lower() for x in ['certificate', 'this', 'that', 'date', 'chief']):
                            skills.add(phrase.title())
            except:
                pass
        
        analysis.skills_covered = list(skills)[:8]
        
        # Detect industry
        analysis.industry = self.detect_industry(text)
        analysis.completion_status = 'completed'
        
        # Extract duration
        for pattern in self.certificate_keywords['duration_patterns']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                analysis.duration = f"{matches[0][0]} {matches[0][1]}"
                break
        
        return analysis

# Caption Generator
class CaptionGenerator:
    """Generate captions for different platforms"""
    
    def __init__(self, analyzer: CertificateAnalyzer):
        self.analyzer = analyzer
    
    def generate_hashtags(self, analysis: CertificateAnalysis, platform: str = "linkedin") -> List[str]:
        """Generate relevant hashtags"""
        hashtags = set()
        
        # Industry hashtags
        industry_tags = self.analyzer.industry_hashtags.get(analysis.industry, 
                                                            self.analyzer.industry_hashtags['general'])
        hashtags.update(industry_tags[:3])
        
        # Type-based hashtags
        if analysis.certificate_type == 'course':
            hashtags.update(["#OnlineLearning", "#ProfessionalDevelopment"])
        elif analysis.certificate_type == 'workshop':
            hashtags.update(["#Workshop", "#HandsOnLearning"])
        else:
            hashtags.update(["#Certification", "#Achievement"])
        
        return list(hashtags)[:10]
    
    def generate_caption(self, analysis: CertificateAnalysis, style: str = "professional", 
                        platform: str = "linkedin", include_skills: bool = True) -> str:
        """Generate caption"""
        template = self.analyzer.caption_templates.get(style, 
                                                       self.analyzer.caption_templates['professional'])
        parts = []
        
        # Opening
        parts.append(np.random.choice(template.opening))
        parts.append("\n\n")
        
        # Achievement
        org_text = f" by {analysis.organization}" if analysis.organization else ""
        
        # Get the right template, with 'program' as fallback to 'course'
        cert_type = analysis.certificate_type if analysis.certificate_type in template.achievement_templates else 'course'
        achievement = template.achievement_templates.get(cert_type, template.achievement_templates['course'])
        
        parts.append(achievement.format(title=analysis.title, organization_text=org_text))
        parts.append("\n\n")
        
        # Skills - More detailed
        if include_skills and analysis.skills_covered:
            if len(analysis.skills_covered) > 4:
                skills_text = ", ".join(analysis.skills_covered[:4])
                parts.append(f"📚 Key competencies developed:\n{skills_text}, and more.\n\n")
            else:
                skills_text = ", ".join(analysis.skills_covered)
                parts.append(f"📚 Key competencies developed: {skills_text}\n\n")
        
        # Value statement
        parts.append(np.random.choice(template.value_statements))
        parts.append("\n\n")
        
        # CTA
        parts.append(np.random.choice(template.call_to_actions))
        parts.append("\n\n")
        
        # Hashtags
        hashtags = self.generate_hashtags(analysis, platform)
        parts.append(" ".join(hashtags))
        
        return "".join(parts)

# Initialize analyzer and generators
@st.cache_resource
def get_analyzer():
    return CertificateAnalyzer()

@st.cache_resource
def get_generator(_analyzer):
    return CaptionGenerator(_analyzer)

@st.cache_resource
def get_hybrid_generator(_analyzer):
    """Initialize hybrid AI generator (cached for performance)"""
    return HybridCaptionGenerator(_analyzer)

analyzer = get_analyzer()
generator = get_generator(analyzer)
hybrid_generator = get_hybrid_generator(analyzer)

# Process file function
def process_certificate_file(file_bytes, filename):
    """Process uploaded certificate file"""
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    
    with open(temp_path, 'wb') as f:
        f.write(file_bytes)
    
    try:
        if filename.lower().endswith('.pdf'):
            result = analyzer.extract_text_from_pdf(temp_path)
        else:
            result = analyzer.extract_text_from_image(temp_path)
        
        if result['text'].strip():
            analysis = analyzer.analyze_certificate(result['text'], result['confidence'])
            return {
                'success': True,
                'analysis': analysis,
                'extraction_info': result
            }
        else:
            return {'success': False, 'error': 'Could not extract text from file'}
    finally:
        try:
            os.remove(temp_path)
        except:
            pass

# Main App
def main():
    # Header
    st.markdown("<h1>🎓 Certificate Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Transform your certificates into professional LinkedIn captions instantly!</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # AI Mode Selection
        st.markdown("### 🤖 Generation Mode")
        
        # Initialize hybrid generator to check available modes
        analyzer = CertificateAnalyzer()
        hybrid_gen = HybridCaptionGenerator(analyzer)
        available_modes = hybrid_gen.get_available_modes()
        
        mode_descriptions = {
            "template": "⚡ Fast & Reliable (Rule-based)",
            "blip": "🔮 BLIP-2 Vision AI",
            "llama": "🦙 LLaMA 3.2 Language Model",
            "hybrid": "🚀 Full AI Pipeline (BLIP + LLaMA)"
        }
        
        # Show available modes with descriptions
        mode_options = [mode_descriptions.get(m, m) for m in available_modes]
        selected_mode_display = st.selectbox(
            "AI Generation Mode",
            mode_options,
            help="Choose how captions are generated. Hybrid mode uses multi-modal AI pipeline."
        )
        
        # Extract actual mode name
        generation_mode = available_modes[mode_options.index(selected_mode_display)]
        
        # Show mode info
        if generation_mode == "template":
            st.info("📝 Using template-based generation (fast, consistent)")
        elif generation_mode == "blip":
            st.info("🎨 Using BLIP-2 for visual understanding")
        elif generation_mode == "llama":
            st.info("🧠 Using LLaMA 3.2 for natural language generation")
        elif generation_mode == "hybrid":
            st.success("✨ Using full AI pipeline: BLIP-2 + OCR + LLaMA 3.2")
        
        # Show unavailable modes
        all_modes = ["template", "blip", "llama", "hybrid"]
        unavailable = [m for m in all_modes if m not in available_modes]
        if unavailable:
            with st.expander("📦 Enhance with AI Models"):
                st.markdown("**Available modes:**")
                for mode in available_modes:
                    st.markdown(f"✅ {mode}")
                
                st.markdown("**Install for more AI features:**")
                if "blip" in unavailable:
                    st.code("pip install transformers torch", language="bash")
                if "llama" in unavailable or "hybrid" in unavailable:
                    st.markdown("🦙 Install Ollama: https://ollama.ai")
                    st.code("ollama pull llama3.2", language="bash")
        
        st.markdown("---")
        
        # Traditional settings
        style = st.selectbox(
            "Caption Style",
            ["professional", "enthusiastic", "technical"],
            help="Choose the tone (applies to template mode)"
        )
        
        platform = st.selectbox(
            "Platform",
            ["linkedin", "twitter", "instagram", "portfolio"],
            help="Optimize caption for specific platform"
        )
        
        include_skills = st.checkbox("Include Skills", value=True, help="Add extracted skills to caption")
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("✅ Multi-Modal AI Pipeline")
        st.markdown("✅ BLIP-2 Vision Understanding")
        st.markdown("✅ LLaMA 3.2 Integration")
        st.markdown("✅ Multi-engine OCR")
        st.markdown("✅ Smart industry detection")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.info("For best results, use clear, high-resolution images or PDFs")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Certificate")
        
        # Show Poppler status
        if not POPPLER_AVAILABLE:
            st.warning("⚠️ **PDF Support Limited**: Poppler not detected. Please upload image files (PNG/JPG) for best results.")
            st.info("💡 To enable full PDF support, install Poppler and set POPPLER_PATH in the code.")
        
        uploaded_file = st.file_uploader(
            "Choose your certificate file",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload PDF or image (PNG, JPG, JPEG). Images recommended if Poppler is not installed."
        )
        
        if uploaded_file:
            # Display preview
            if uploaded_file.type == "application/pdf":
                st.info(f"📄 PDF uploaded: {uploaded_file.name}")
                if not POPPLER_AVAILABLE:
                    st.warning("⚠️ Poppler not available. PDF OCR may not work. Try converting to PNG/JPG first.")
            else:
                st.image(uploaded_file, caption="Uploaded Certificate", use_column_width=True)
        
        # Manual input option
        with st.expander("📝 Or Enter Details Manually"):
            manual_title = st.text_input("Certificate Title", placeholder="e.g., Data Science Bootcamp")
            manual_org = st.text_input("Organization", placeholder="e.g., Tech Academy")
            manual_skills = st.text_area("Skills (comma-separated)", placeholder="e.g., Python, Machine Learning, Data Analysis")
    
    with col2:
        st.markdown("### ✨ Generated Caption")
        
        if st.button("🚀 Generate Caption", use_container_width=True):
            if uploaded_file or (manual_title and manual_org):
                with st.spinner("🔄 Processing your certificate..."):
                    try:
                        if uploaded_file:
                            # Process uploaded file
                            file_bytes = uploaded_file.read()
                            result = process_certificate_file(file_bytes, uploaded_file.name)
                            
                            if result['success']:
                                analysis = result['analysis']
                            else:
                                st.error(f"❌ Error: {result.get('error', 'Processing failed')}")
                                return
                        else:
                            # Use manual input
                            analysis = CertificateAnalysis()
                            analysis.title = manual_title
                            analysis.organization = manual_org
                            analysis.certificate_type = 'course'
                            analysis.completion_status = 'completed'
                            analysis.confidence_score = 0.9
                            if manual_skills:
                                analysis.skills_covered = [s.strip() for s in manual_skills.split(',')]
                            analysis.industry = analyzer.detect_industry(manual_title + " " + manual_org + " " + manual_skills)
                        
                        # Get image for AI models (if uploaded)
                        image_for_ai = None
                        if uploaded_file and uploaded_file.type != "application/pdf":
                            uploaded_file.seek(0)  # Reset file pointer
                            image_for_ai = Image.open(uploaded_file)
                        
                        # Generate caption based on selected mode
                        if generation_mode in ["blip", "llama", "hybrid"] and image_for_ai:
                            # Use AI pipeline
                            with st.spinner(f"🤖 Generating with {generation_mode.upper()} AI..."):
                                caption, mode_used = hybrid_generator.generate(image_for_ai, analysis, generation_mode)
                                st.info(f"🎯 Generated using: **{mode_used}**")
                        else:
                            # Use template generation
                            if generation_mode != "template" and not image_for_ai:
                                st.warning("⚠️ AI modes require image upload. Using template mode.")
                            caption = generator.generate_caption(analysis, style, platform, include_skills)
                            mode_used = "template"
                        
                        hashtags = generator.generate_hashtags(analysis, platform)
                        
                        # Display results
                        st.success("✅ Caption generated successfully!")
                        
                        # Analysis details
                        with st.expander("📊 Certificate Analysis", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Title", analysis.title[:30] + "..." if len(analysis.title) > 30 else analysis.title)
                                st.metric("Type", analysis.certificate_type.title())
                                st.metric("Industry", analysis.industry.replace('_', ' ').title())
                            with col_b:
                                st.metric("Organization", analysis.organization or "Not detected")
                                st.metric("Skills Found", len(analysis.skills_covered))
                                st.metric("Confidence", f"{analysis.confidence_score:.0%}")
                        
                        # Caption display
                        st.markdown("### 📝 Your Caption")
                        st.text_area(
                            "Copy and paste to LinkedIn:",
                            caption,
                            height=300,
                            label_visibility="collapsed"
                        )
                        
                        # Metrics
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("Characters", len(caption))
                        with col_m2:
                            st.metric("Words", len(caption.split()))
                        with col_m3:
                            st.metric("Hashtags", len(hashtags))
                        
                        # Download button
                        st.download_button(
                            label="💾 Download Caption",
                            data=caption,
                            file_name=f"linkedin_caption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                        # Skills display
                        if analysis.skills_covered:
                            st.markdown("### 🎯 Detected Skills")
                            skills_html = " ".join([f"<span style='background: #667eea; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0.2rem; display: inline-block;'>{skill}</span>" 
                                                   for skill in analysis.skills_covered])
                            st.markdown(skills_html, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Processing error: {e}")
            else:
                st.warning("⚠️ Please upload a certificate or fill in manual details")

if __name__ == "__main__":
    main()
