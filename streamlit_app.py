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

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .css-1d391kg {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #667eea;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    h2, h3 {
        color: #764ba2;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 1rem 0;
    }
    .caption-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
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
            'technology': ['#TechSkills', '#Programming', '#SoftwareDevelopment', '#Innovation'],
            'data_science': ['#DataScience', '#MachineLearning', '#Analytics', '#AI'],
            'design': ['#Design', '#UXDesign', '#CreativeSkills', '#UserExperience'],
            'business': ['#BusinessSkills', '#Leadership', '#Management', '#Strategy'],
            'marketing': ['#DigitalMarketing', '#MarketingStrategy', '#SocialMedia'],
            'finance': ['#Finance', '#FinTech', '#Investment', '#Accounting'],
            'healthcare': ['#Healthcare', '#MedicalTraining', '#HealthTech'],
            'education': ['#Education', '#Teaching', '#LearningAndDevelopment'],
            'general': ['#ProfessionalDevelopment', '#SkillBuilding', '#CareerGrowth']
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
                    'certification': "I have earned certification in {title}{organization_text}."
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
                    'certification': "🏆 Officially certified in {title}{organization_text}! 🎊"
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
                    'certification': "Achieved professional certification in {title}{organization_text}."
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

# Initialize analyzer and generator
@st.cache_resource
def get_analyzer():
    return CertificateAnalyzer()

@st.cache_resource
def get_generator(_analyzer):
    return CaptionGenerator(_analyzer)

analyzer = get_analyzer()
generator = get_generator(analyzer)

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
        
        style = st.selectbox(
            "Caption Style",
            ["professional", "enthusiastic", "technical"],
            help="Choose the tone of your caption"
        )
        
        platform = st.selectbox(
            "Platform",
            ["linkedin", "twitter", "instagram", "portfolio"],
            help="Optimize caption for specific platform"
        )
        
        include_skills = st.checkbox("Include Skills", value=True, help="Add extracted skills to caption")
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("✅ Multi-engine OCR")
        st.markdown("✅ Smart industry detection")
        st.markdown("✅ 3 caption styles")
        st.markdown("✅ 4 platform formats")
        st.markdown("✅ Instant generation")
        
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
                        
                        # Generate caption
                        caption = generator.generate_caption(analysis, style, platform, include_skills)
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
