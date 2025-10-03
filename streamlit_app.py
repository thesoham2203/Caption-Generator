"""
Certificate Caption Generator with Mistral 7B
Clean architecture: Streamlit Frontend + Ollama API Backend
Optimized for RTX 3050 6GB GPU
"""

import streamlit as st
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

# Core libraries
import cv2
import numpy as np
from PIL import Image
import pytesseract
from textblob import TextBlob

# Try optional imports
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Certificate Caption Generator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - FUTURISTIC DARK THEME
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0ff;
    }
    
    /* Headers with gradient */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 0.8rem !important;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: rgba(102, 126, 234, 0.8) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        backdrop-filter: blur(15px);
    }
    
    /* Caption output box */
    .caption-box {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Metrics cards */
    .metric-card {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Success message */
    .success-box {
        background: rgba(76, 175, 80, 0.15);
        border-left: 4px solid #4caf50;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #90ee90;
    }
    
    /* Info box */
    .info-box {
        background: rgba(33, 150, 243, 0.15);
        border-left: 4px solid #2196f3;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        color: #90caf9;
    }
    
    /* Skill badges */
    .skill-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border: 1px solid rgba(102, 126, 234, 0.4);
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        color: #b8b8ff;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CertificateData:
    """Extracted certificate information"""
    title: str = ""
    organization: str = ""
    recipient_name: str = ""
    date_issued: str = ""
    skills: List[str] = field(default_factory=list)
    industry: str = "general"
    certificate_type: str = "course"
    raw_text: str = ""

@dataclass
class CaptionPreferences:
    """User preferences for caption generation"""
    tone: str = "professional"
    platform: str = "linkedin"
    length: str = "medium"
    include_hashtags: bool = True
    include_skills: bool = True
    custom_message: str = ""
    emoji_style: str = "minimal"

# ============================================================================
# OCR ENGINE
# ============================================================================

class CertificateOCR:
    """Handles OCR extraction from certificates"""
    
    def __init__(self):
        self.pytesseract_config = r'--oem 3 --psm 6 -l eng'
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], verbose=False)
            except:
                pass
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    def extract_from_image(self, image_path: str) -> Dict:
        """Extract text from image using dual OCR"""
        results = {
            'text': '',
            'confidence': 0.0,
            'method': 'pytesseract',
            'error': None
        }
        
        try:
            img = cv2.imread(image_path)
            
            if img is None:
                results['error'] = "Could not read image file. The file may be corrupted or in an unsupported format."
                return results
            
            # Check image dimensions
            height, width = img.shape[:2]
            if width < 100 or height < 100:
                results['error'] = "Image is too small (minimum 100x100 pixels). Please upload a higher resolution image."
                return results
            
            if width > 10000 or height > 10000:
                st.info("📏 Large image detected. Resizing for faster processing...")
                scale = min(10000/width, 10000/height)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            processed = self.preprocess_image(img)
            
        except Exception as e:
            results['error'] = f"Image preprocessing failed: {str(e)}"
            return results
        
        # Try PyTesseract first
        try:
            text = pytesseract.image_to_string(processed, config=self.pytesseract_config)
            results['text'] = text.strip()
            results['confidence'] = 0.8
        except pytesseract.TesseractNotFoundError:
            results['error'] = "Tesseract OCR is not installed. Please install Tesseract and try again."
            return results
        except Exception as e:
            st.warning(f"⚠️ PyTesseract extraction had issues: {str(e)}")
        
        # Try EasyOCR if available and text is poor
        if self.easyocr_reader and (not results['text'] or len(results['text']) < 50):
            try:
                st.info("🔄 Trying enhanced OCR (EasyOCR) for better accuracy...")
                easyocr_results = self.easyocr_reader.readtext(image_path)
                easyocr_text = ' '.join([text for (_, text, conf) in easyocr_results if conf > 0.3])
                
                if len(easyocr_text) > len(results['text']):
                    results['text'] = easyocr_text
                    results['method'] = 'easyocr'
                    results['confidence'] = 0.85
            except Exception as e:
                st.warning(f"⚠️ EasyOCR also had issues: {str(e)}")
        
        # Final validation
        if not results['text'] or len(results['text'].strip()) < 20:
            results['error'] = "Could not extract sufficient text from the image. The image may be too blurry, low quality, or not contain readable text. Try uploading a clearer image or use manual input."
        
        return results
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """Extract text from PDF (handles both text and image-based PDFs)"""
        results = {
            'text': '',
            'confidence': 0.9,
            'method': 'pymupdf'
        }
        
        if not PDF_SUPPORT:
            return results
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # First, try to extract text directly
                page_text = page.get_text().strip()
                
                if page_text and len(page_text) > 50:
                    # Good text extraction
                    text += page_text + "\n"
                else:
                    # Text is poor/empty, try OCR on page images
                    st.info(f"Page {page_num + 1}: Text extraction poor, using OCR...")
                    
                    # Get images from the page
                    image_list = page.get_images(full=True)
                    
                    if image_list:
                        # Extract and OCR the largest image on the page
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                
                                # Convert to numpy array for OCR
                                nparr = np.frombuffer(image_bytes, np.uint8)
                                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if img_cv is not None:
                                    # Preprocess and OCR
                                    processed = self.preprocess_image(img_cv)
                                    
                                    # Try PyTesseract
                                    ocr_text = pytesseract.image_to_string(processed, config=self.pytesseract_config)
                                    
                                    if ocr_text and len(ocr_text) > 50:
                                        text += ocr_text + "\n"
                                        results['method'] = 'pymupdf_ocr'
                                        results['confidence'] = 0.8
                                        break  # Got good text, move to next page
                                    
                                    # If PyTesseract fails, try EasyOCR
                                    if self.easyocr_reader and (not ocr_text or len(ocr_text) < 50):
                                        # Save temp image for EasyOCR
                                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
                                            cv2.imwrite(tmp_img.name, img_cv)
                                            tmp_img_path = tmp_img.name
                                        
                                        try:
                                            easyocr_results = self.easyocr_reader.readtext(tmp_img_path)
                                            easyocr_text = ' '.join([txt for (_, txt, conf) in easyocr_results if conf > 0.3])
                                            
                                            if len(easyocr_text) > len(ocr_text):
                                                text += easyocr_text + "\n"
                                                results['method'] = 'pymupdf_easyocr'
                                                results['confidence'] = 0.85
                                                break
                                        finally:
                                            os.unlink(tmp_img_path)
                                
                            except Exception as e:
                                st.warning(f"Failed to OCR image {img_index + 1} on page {page_num + 1}: {e}")
                                continue
                    else:
                        # No images found, render page as image and OCR
                        st.info(f"Page {page_num + 1}: Rendering page as image for OCR...")
                        
                        # Render page to image (300 DPI for better quality)
                        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to numpy array
                        img_data = pix.tobytes("png")
                        nparr = np.frombuffer(img_data, np.uint8)
                        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img_cv is not None:
                            # Preprocess and OCR
                            processed = self.preprocess_image(img_cv)
                            ocr_text = pytesseract.image_to_string(processed, config=self.pytesseract_config)
                            
                            if ocr_text:
                                text += ocr_text + "\n"
                                results['method'] = 'pymupdf_page_render'
                                results['confidence'] = 0.75
            
            doc.close()
            results['text'] = text
            return results
            
        except fitz.FileDataError:
            results['error'] = "The PDF file is corrupted or password-protected. Please try a different file."
            return results
        except fitz.EmptyFileError:
            results['error'] = "The PDF file is empty. Please upload a valid certificate PDF."
            return results
        except Exception as e:
            results['error'] = f"PDF extraction failed: {str(e)}. The file may be corrupted or in an unsupported format."
            return results
        finally:
            # Ensure document is closed
            try:
                doc.close()
            except:
                pass
        
        # Final validation
        if not text or len(text.strip()) < 20:
            results['error'] = "Could not extract sufficient text from the PDF. The PDF may contain only images or be of poor quality. Try using manual input instead."
        
        return results

# ============================================================================
# DATA EXTRACTOR
# ============================================================================

class DataExtractor:
    """Extracts structured data from OCR text"""
    
    def __init__(self):
        self.industry_keywords = {
            'technology': ['programming', 'software', 'coding', 'development', 'tech', 'computer', 
                          'python', 'java', 'javascript', 'web', 'mobile', 'app', 'data', 'ai', 'ml'],
            'data_science': ['data science', 'machine learning', 'analytics', 'statistics', 'big data',
                            'data analysis', 'visualization', 'pandas', 'numpy', 'tensorflow'],
            'business': ['business', 'management', 'marketing', 'finance', 'accounting', 'mba',
                        'leadership', 'strategy', 'entrepreneur', 'sales'],
            'design': ['design', 'ui', 'ux', 'graphic', 'creative', 'photoshop', 'illustrator',
                      'figma', 'adobe', 'branding'],
            'cloud': ['aws', 'azure', 'cloud', 'devops', 'kubernetes', 'docker', 'gcp'],
            'security': ['security', 'cybersecurity', 'ethical hacking', 'penetration', 'cissp'],
        }
    
    def extract(self, text: str) -> CertificateData:
        """Extract all relevant data from text"""
        data = CertificateData()
        data.raw_text = text
        
        # Extract title
        data.title = self._extract_title(text)
        
        # Extract organization
        data.organization = self._extract_organization(text)
        
        # Extract date
        data.date_issued = self._extract_date(text)
        
        # Extract skills
        data.skills = self._extract_skills(text)
        
        # Detect industry
        data.industry = self._detect_industry(text)
        
        # Determine certificate type
        data.certificate_type = self._determine_type(text)
        
        return data
    
    def _extract_title(self, text: str) -> str:
        """Extract certificate title"""
        # Common patterns
        patterns = [
            r'certificate of (.*?)(?:\n|issued|presented|awarded)',
            r'certification in (.*?)(?:\n|from|by)',
            r'this certifies that.*?(?:completed|finished)\s+(.*?)(?:\n|from|on)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                title = match.group(1).strip()
                # Clean up
                title = re.sub(r'\s+', ' ', title)
                if 10 < len(title) < 100:
                    return title
        
        # Fallback: look for course/program keywords
        lines = text.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['course', 'program', 'certification', 'bootcamp']):
                if 10 < len(line) < 100:
                    return line.strip()
        
        return "Professional Certification"
    
    def _extract_organization(self, text: str) -> str:
        """Extract issuing organization"""
        # Common patterns
        patterns = [
            r'(?:from|by|issued by)\s+([A-Z][A-Za-z\s&]+(?:University|Institute|Academy|College|Foundation|Inc|Ltd|Corporation|Company))',
            r'([A-Z][A-Za-z\s&]+(?:University|Institute|Academy|College|Foundation))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                org = match.group(1).strip()
                if 5 < len(org) < 50:
                    return org
        
        return "Professional Institution"
    
    def _extract_date(self, text: str) -> str:
        """Extract issue date"""
        # Date patterns
        patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{4}\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return datetime.now().strftime("%B %Y")
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills mentioned in certificate"""
        skills = set()
        
        # Look for skills section
        skills_section_match = re.search(r'(?:skills?|topics?|covered)[\s:]+(.+?)(?:\n\n|\Z)', 
                                        text, re.IGNORECASE | re.DOTALL)
        
        if skills_section_match:
            skills_text = skills_section_match.group(1)
            # Split by common delimiters
            potential_skills = re.split(r'[,;•\n]', skills_text)
            for skill in potential_skills:
                skill = skill.strip()
                if 2 < len(skill) < 30 and not any(char.isdigit() for char in skill):
                    skills.add(skill.title())
        
        # Also check for common tech skills in full text
        common_skills = [
            'Python', 'Java', 'JavaScript', 'SQL', 'Machine Learning', 
            'Data Analysis', 'Web Development', 'Cloud Computing', 'AWS',
            'React', 'Node.js', 'Docker', 'Kubernetes', 'Git'
        ]
        
        for skill in common_skills:
            if skill.lower() in text.lower():
                skills.add(skill)
        
        return list(skills)[:10]  # Limit to 10 skills
    
    def _detect_industry(self, text: str) -> str:
        """Detect industry category"""
        text_lower = text.lower()
        scores = {industry: 0 for industry in self.industry_keywords}
        
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[industry] += 1
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return 'general'
    
    def _determine_type(self, text: str) -> str:
        """Determine certificate type"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['completion', 'completed', 'finished']):
            return 'completion'
        elif any(word in text_lower for word in ['achievement', 'excellence', 'award']):
            return 'achievement'
        elif any(word in text_lower for word in ['participation', 'attended', 'workshop']):
            return 'participation'
        else:
            return 'course'

# ============================================================================
# MISTRAL AI CAPTION GENERATOR
# ============================================================================

class MistralCaptionGenerator:
    """Generate captions using Mistral 7B via Ollama API"""
    
    def __init__(self, model_name: str = "mistral:7b-instruct-q4_K_M"):
        self.model_name = model_name
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if Mistral is available"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'mistral' in result.stdout.lower()
        except:
            return False
    
    def generate_caption(self, cert_data: CertificateData, prefs: CaptionPreferences) -> Dict:
        """Generate caption using Mistral 7B"""
        if not self.available:
            return {
                'success': False,
                'caption': '',
                'error': 'Mistral model not available. Run: ollama pull mistral:7b-instruct-q4_K_M'
            }
        
        # Build prompt
        prompt = self._build_prompt(cert_data, prefs)
        
        # Call Ollama
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=30,
                encoding='utf-8',
                errors='replace'
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                caption = result.stdout.strip()
                
                # Post-process caption
                caption = self._post_process(caption, prefs)
                
                return {
                    'success': True,
                    'caption': caption,
                    'generation_time': elapsed,
                    'model': self.model_name
                }
            else:
                return {
                    'success': False,
                    'caption': '',
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'caption': '',
                'error': 'Generation timed out (>30s). Try again.'
            }
        except Exception as e:
            return {
                'success': False,
                'caption': '',
                'error': str(e)
            }
    
    def _build_prompt(self, cert_data: CertificateData, prefs: CaptionPreferences) -> str:
        """Build optimized prompt for Mistral with role-based prompting"""
        
        # Tone descriptions
        tone_map = {
            'professional': 'formal and business-appropriate',
            'enthusiastic': 'energetic and excited',
            'humble': 'modest and grateful',
            'confident': 'assertive and achievement-focused',
            'casual': 'friendly and conversational'
        }
        
        # Length specifications
        length_map = {
            'short': '100-150 words',
            'medium': '150-200 words',
            'long': '200-250 words'
        }
        
        # Emoji specifications
        emoji_map = {
            'none': 'No emojis',
            'minimal': '2-3 emojis total',
            'moderate': '4-6 emojis',
            'enthusiastic': '8-10 emojis'
        }
        
        # System prompt with role-based prompting
        system_prompt = [
            "You are an expert social media content creator and LinkedIn influencer with 10+ years of experience.",
            "You specialize in crafting professional, engaging posts that showcase achievements and certifications.",
            "",
            "Your expertise includes:",
            "- Converting technical achievements into compelling narratives",
            "- Understanding platform-specific content strategies",
            "- Creating authentic, human-sounding content that drives engagement",
            "- Balancing professionalism with personality",
            "- Industry knowledge across technology, business, design, and data science",
            "",
            "Your writing style is:",
            "- Authentic and personal (never generic)",
            "- Value-focused (emphasizes impact and growth)",
            "- Engaging without being overly promotional",
            "- Optimized for each social media platform",
            "",
            "You NEVER use clichéd phrases like 'excited to announce' or 'pleased to share'.",
            "Instead, you craft unique, compelling openings that grab attention."
        ]
        
        # Task-specific prompt
        task_prompt = [
            f"Create a {prefs.platform.upper()} post caption for this certificate achievement:",
            "",
            "CERTIFICATE DETAILS:",
            f"- Certificate: {cert_data.title}",
            f"- Organization: {cert_data.organization}",
            f"- Date: {cert_data.date_issued}",
        ]
        
        if cert_data.skills:
            task_prompt.append(f"- Key Skills: {', '.join(cert_data.skills[:5])}")
        
        if cert_data.industry:
            task_prompt.append(f"- Industry: {cert_data.industry}")
        
        task_prompt.extend([
            "",
            "CONTENT REQUIREMENTS:",
            f"- Tone: {tone_map.get(prefs.tone, 'professional')}",
            f"- Length: {length_map.get(prefs.length, '150-200 words')}",
            f"- Emojis: {emoji_map.get(prefs.emoji_style, 'minimal')}",
        ])
        
        if prefs.custom_message:
            task_prompt.append(f"- Include this personal message: {prefs.custom_message}")
        
        # Platform-specific guidance
        platform_guidance = {
            'linkedin': "Focus on professional growth, career development, and industry value.",
            'twitter': "Make it concise, engaging, and thread-worthy with strong hooks.",
            'instagram': "Tell a visual story with aspirational and inspirational elements.",
            'facebook': "Personal achievement sharing with community focus."
        }
        
        task_prompt.extend([
            "",
            "PLATFORM STRATEGY:",
            f"- {platform_guidance.get(prefs.platform, 'Professional achievement sharing')}",
            "",
            "CONTENT STRUCTURE:",
            "1. Hook: Start with a compelling, unique opening (not clichéd)",
            "2. Story: Describe the learning journey and key insights",
            "3. Value: Explain the impact and why it matters",
        ])
        
        if prefs.include_skills:
            task_prompt.append("4. Skills: Naturally weave in key competencies gained")
        
        if prefs.include_hashtags:
            task_prompt.append("5. Hashtags: End with 5-8 relevant, strategic hashtags")
        
        task_prompt.extend([
            "",
            "QUALITY STANDARDS:",
            "- Write as if you're personally sharing your achievement",
            "- Make it feel human and authentic, not AI-generated",
            "- Use industry-specific language when appropriate",
            "- Create content that encourages engagement and conversation",
            "- Ensure every sentence adds value to the reader",
            "",
            "Now create the perfect caption:"
        ])
        
        # Combine system prompt and task prompt
        full_prompt = "\n".join(system_prompt + ["", "---", ""] + task_prompt)
        
        return full_prompt
    
    def _post_process(self, caption: str, prefs: CaptionPreferences) -> str:
        """Clean up and format the caption"""
        # Remove any prompt artifacts
        caption = caption.strip()
        
        # Ensure hashtags are at the end if requested
        if prefs.include_hashtags and '#' in caption:
            parts = caption.split('#')
            text = parts[0].strip()
            hashtags = ['#' + tag.strip() for tag in parts[1:] if tag.strip()]
            caption = f"{text}\n\n{' '.join(hashtags)}"
        
        return caption
    
    def generate_hashtags(self, cert_data: CertificateData, count: int = 8) -> List[str]:
        """Generate relevant hashtags"""
        hashtags = set()
        
        # Industry-based
        industry_tags = {
            'technology': ['Tech', 'Programming', 'SoftwareDevelopment', 'CodingLife'],
            'data_science': ['DataScience', 'MachineLearning', 'AI', 'BigData', 'Analytics'],
            'business': ['Business', 'Leadership', 'Management', 'Entrepreneurship'],
            'design': ['Design', 'UX', 'UI', 'CreativeDesign', 'GraphicDesign'],
            'cloud': ['CloudComputing', 'AWS', 'Azure', 'DevOps'],
            'security': ['Cybersecurity', 'InfoSec', 'EthicalHacking'],
        }
        
        if cert_data.industry in industry_tags:
            hashtags.update(industry_tags[cert_data.industry][:3])
        
        # Generic professional tags
        hashtags.update(['ProfessionalDevelopment', 'LifelongLearning', 'CareerGrowth'])
        
        # Skills-based
        for skill in cert_data.skills[:3]:
            tag = skill.replace(' ', '').replace('-', '')
            hashtags.add(tag)
        
        return ['#' + tag for tag in list(hashtags)[:count]]

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🎓 AI Certificate Caption Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #b8b8ff;'>Powered by Mistral 7B AI • Optimized for Professional LinkedIn Posts</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid rgba(102, 126, 234, 0.3);'>", unsafe_allow_html=True)
    
    # Initialize components
    try:
        ocr = CertificateOCR()
    except Exception as e:
        st.error("⚠️ **OCR Initialization Failed!**")
        st.markdown(f"""
            ### 🔍 OCR Engine Error
            
            **Error:** {str(e)}
            
            **Possible Causes:**
            1. **Tesseract OCR not installed**
               - Download: https://github.com/UB-Mannheim/tesseract/wiki
               - Install and add to PATH
            
            2. **EasyOCR installation issues**
               - Optional but recommended
               - Install: `pip install easyocr`
            
            ### 📋 Quick Fix:
            ```bash
            # Install Tesseract (Windows)
            # Download from: https://github.com/UB-Mannheim/tesseract/wiki
            # Add to PATH: C:\\Program Files\\Tesseract-OCR
            
            # Verify installation
            tesseract --version
            ```
            
            **After installing, restart the app.**
        """)
        st.stop()
    
    extractor = DataExtractor()
    generator = MistralCaptionGenerator()
    
    # Check Mistral availability
    if not generator.available:
        st.error("⚠️ **Mistral 7B AI Model Not Found!**")
        st.markdown("""
            ### 🤖 Setup Required
            
            The AI caption generator requires the Mistral 7B model to be installed.
            
            **Installation Steps:**
            
            1. **Install Ollama** (if not already installed)
               - Visit: https://ollama.ai/download
               - Download and install for your system
            
            2. **Download Mistral 7B Model**
               ```bash
               ollama pull mistral:7b-instruct-q4_K_M
               ```
               - Size: ~4.1 GB
               - Time: 5-10 minutes (depending on internet speed)
            
            3. **Verify Installation**
               ```bash
               ollama list
               ```
               - You should see `mistral:7b-instruct-q4_K_M` in the list
            
            4. **Refresh this page** and start generating captions!
            
            ---
            
            ### 📋 System Requirements
            - **RAM**: 8GB+ recommended
            - **GPU**: 4GB+ VRAM (optional but faster)
            - **Disk Space**: 5GB free
            - **OS**: Windows 10/11, macOS, or Linux
            
            ### ❓ Need Help?
            - Check README.md for detailed setup instructions
            - Ensure Ollama service is running: `ollama serve`
        """)
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Caption Settings")
        
        # Tone selection
        tone = st.selectbox(
            "Tone",
            ["professional", "enthusiastic", "humble", "confident", "casual"],
            help="Choose the overall tone of your caption"
        )
        
        # Platform selection
        platform = st.selectbox(
            "Platform",
            ["linkedin", "twitter", "instagram", "facebook"],
            help="Optimize for specific social media platform"
        )
        
        # Length selection
        length = st.radio(
            "Caption Length",
            ["short", "medium", "long"],
            index=1,
            help="Short: ~100 words, Medium: ~150 words, Long: ~200 words"
        )
        
        # Emoji style
        emoji_style = st.select_slider(
            "Emoji Usage",
            options=["none", "minimal", "moderate", "enthusiastic"],
            value="minimal",
            help="How many emojis to include"
        )
        
        # Toggles
        include_hashtags = st.checkbox("Include Hashtags", value=True)
        include_skills = st.checkbox("Highlight Skills", value=True)
        
        # Custom message
        custom_message = st.text_area(
            "Custom Message (Optional)",
            placeholder="Add a personal note to include in the caption...",
            help="This will be woven into your caption"
        )
        
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.info("**Mistral 7B Q4**\nSpeed: ~6-8 seconds\nQuality: Excellent")
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("✅ Dual OCR Engine")
        st.markdown("✅ Smart Data Extraction")
        st.markdown("✅ AI Caption Generation")
        st.markdown("✅ Industry Detection")
        st.markdown("✅ Custom Preferences")
        
        st.markdown("---")
        
        # Troubleshooting section
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
                **Common Issues:**
                
                **1. "Could not extract text"**
                - Ensure image is clear (300+ DPI)
                - Avoid blurry or low-light photos
                - Try manual input instead
                
                **2. "File too large"**
                - Compress image before upload
                - Max size: 200MB
                - Use PNG/JPG for best results
                
                **3. "Generation timeout"**
                - Close other GPU applications
                - Wait and try again
                - Check Ollama is running
                
                **4. "Mistral not found"**
                - Run: `ollama pull mistral:7b-instruct-q4_K_M`
                - Verify: `ollama list`
                
                **5. "Unclear extraction"**
                - Use better lighting
                - Scan at higher resolution
                - Use manual input as backup
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Certificate")
        
        uploaded_file = st.file_uploader(
            "Choose your certificate",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            help="Upload PNG, JPG, or PDF certificate (Max 200MB)"
        )
        
        # Validate file size
        if uploaded_file and uploaded_file.size > 200 * 1024 * 1024:
            st.error("🚫 **File too large!** Please upload a file smaller than 200MB.")
            st.stop()
        
        if uploaded_file:
            # Display preview
            if uploaded_file.type == "application/pdf":
                st.info(f"📄 PDF uploaded: {uploaded_file.name}")
            else:
                st.image(uploaded_file, caption="Certificate Preview", use_column_width=True)
        
        # Manual input option
        with st.expander("📝 Or Enter Details Manually"):
            manual_title = st.text_input("Certificate Title*", placeholder="e.g., Python Data Science Specialization")
            manual_org = st.text_input("Organization*", placeholder="e.g., Coursera, Google, IBM")
            manual_skills = st.text_area("Skills (comma-separated)", placeholder="e.g., Python, Machine Learning, Data Analysis")
            manual_date = st.text_input("Date", placeholder="e.g., October 2025")
            
            use_manual = st.checkbox("Use manual input instead of OCR")
    
    with col2:
        st.markdown("### ✨ Generated Caption")
        
        if st.button("🚀 Generate AI Caption", use_container_width=True, type="primary"):
            # Validation checks
            if not uploaded_file and not (use_manual and manual_title and manual_org):
                st.error("❌ **Missing Input!** Please either upload a certificate or enter details manually (Title and Organization are required).")
                st.info("💡 **Tip:** Use the manual input option below if you're having trouble with file uploads.")
            elif use_manual and (not manual_title or not manual_org):
                st.error("❌ **Incomplete Information!** When using manual input, both Certificate Title and Organization are required.")
            else:
                with st.spinner("🔄 Processing certificate..."):
                    # Extract data
                    if use_manual and manual_title and manual_org:
                        # Use manual input
                        cert_data = CertificateData(
                            title=manual_title,
                            organization=manual_org,
                            date_issued=manual_date or datetime.now().strftime("%B %Y"),
                            skills=[s.strip() for s in manual_skills.split(',')] if manual_skills else [],
                            industry=extractor._detect_industry(manual_title + " " + manual_skills)
                        )
                        st.success("✅ Using manual input")
                    else:
                        # Extract from file
                        # Save temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        
                        try:
                            # Validate file type
                            file_ext = Path(uploaded_file.name).suffix.lower()
                            if file_ext not in ['.pdf', '.png', '.jpg', '.jpeg']:
                                st.error(f"🚫 **Unsupported file type '{file_ext}'!** Please upload PNG, JPG, or PDF files only.")
                                os.unlink(tmp_path)
                                st.stop()
                            
                            # OCR extraction
                            if uploaded_file.type == "application/pdf":
                                ocr_result = ocr.extract_from_pdf(tmp_path)
                            else:
                                ocr_result = ocr.extract_from_image(tmp_path)
                            
                            # Check for errors
                            if ocr_result.get('error'):
                                st.error(f"❌ **Extraction Failed:** {ocr_result['error']}")
                                st.markdown("""
                                    ### 💡 **Troubleshooting Tips:**
                                    1. **Image Quality**: Ensure the certificate is clear and well-lit
                                    2. **Resolution**: Use images with at least 300 DPI
                                    3. **File Format**: Try converting to PNG if using other formats
                                    4. **Manual Input**: Use the manual entry option below as a fallback
                                    5. **Text Visibility**: Make sure text is not obscured by watermarks
                                """)
                                os.unlink(tmp_path)
                                st.stop()
                            
                            if not ocr_result['text'] or len(ocr_result['text'].strip()) < 20:
                                st.error("❌ **Insufficient text extracted!** The image may be too unclear or low quality.")
                                st.warning("⚠️ **Extracted text:** " + (ocr_result['text'][:100] + '...' if len(ocr_result['text']) > 100 else ocr_result['text'] or '(empty)'))
                                st.markdown("""
                                    ### 📝 **Please try one of these options:**
                                    1. **Upload a clearer image** - Take a new photo with better lighting
                                    2. **Use a scanner** - Scan the certificate at 300 DPI or higher
                                    3. **Try a different file format** - Convert to PNG for best results
                                    4. **Use manual input** - Enter certificate details manually below
                                """)
                                os.unlink(tmp_path)
                                st.stop()
                            
                            st.success(f"✅ Text extracted using {ocr_result['method']} (Confidence: {ocr_result.get('confidence', 0.8)*100:.0f}%)")
                            
                            # Extract structured data
                            cert_data = extractor.extract(ocr_result['text'])
                            
                            # Validate extracted data quality
                            if not cert_data.title or len(cert_data.title) < 5:
                                st.warning("⚠️ **Low Quality Extraction:** Could not identify certificate title clearly. Please review extracted data.")
                            
                            if not cert_data.organization or cert_data.organization == "Professional Institution":
                                st.info("ℹ️ **Note:** Organization name not detected. Using default value.")
                            
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                    
                    # Show extracted data
                    with st.expander("📋 Extracted Certificate Data", expanded=False):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Title:** {cert_data.title}")
                            st.markdown(f"**Organization:** {cert_data.organization}")
                            st.markdown(f"**Date:** {cert_data.date_issued}")
                        with col_b:
                            st.markdown(f"**Industry:** {cert_data.industry.replace('_', ' ').title()}")
                            st.markdown(f"**Type:** {cert_data.certificate_type.title()}")
                        
                        if cert_data.skills:
                            st.markdown("**Skills Detected:**")
                            skills_html = "".join([f"<span class='skill-badge'>{skill}</span>" for skill in cert_data.skills])
                            st.markdown(skills_html, unsafe_allow_html=True)
                
                # Generate caption
                with st.spinner("🤖 AI is crafting your caption... (~6-8 seconds)"):
                    prefs = CaptionPreferences(
                        tone=tone,
                        platform=platform,
                        length=length,
                        include_hashtags=include_hashtags,
                        include_skills=include_skills,
                        custom_message=custom_message,
                        emoji_style=emoji_style
                    )
                    
                    result = generator.generate_caption(cert_data, prefs)
                
                if result['success']:
                    # Display caption
                    st.markdown(f"""
                        <div class='success-box'>
                            <strong>✅ Caption generated in {result['generation_time']:.1f} seconds!</strong>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    caption = result['caption']
                    
                    st.markdown(f"""
                        <div class='caption-box'>
                            {caption.replace(chr(10), '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons
                    col_x, col_y, col_z = st.columns(3)
                    
                    with col_x:
                        st.download_button(
                            "📥 Download",
                            caption,
                            file_name=f"caption_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col_y:
                        if st.button("📋 Copy to Clipboard", use_container_width=True):
                            st.code(caption, language=None)
                            st.info("👆 Select and copy the text above")
                    
                    with col_z:
                        if st.button("🔄 Regenerate", use_container_width=True):
                            st.rerun()
                    
                    # Analytics
                    st.markdown("---")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <strong>Word Count</strong><br>
                                <span style='font-size: 1.5rem; color: #667eea;'>{len(caption.split())}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m2:
                        hashtag_count = caption.count('#')
                        st.markdown(f"""
                            <div class='metric-card'>
                                <strong>Hashtags</strong><br>
                                <span style='font-size: 1.5rem; color: #667eea;'>{hashtag_count}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_m3:
                        st.markdown(f"""
                            <div class='metric-card'>
                                <strong>Generation Time</strong><br>
                                <span style='font-size: 1.5rem; color: #667eea;'>{result['generation_time']:.1f}s</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"❌ **Caption Generation Failed!**")
                    
                    # Specific error messages based on error type
                    if 'not available' in error_msg.lower() or 'not found' in error_msg.lower():
                        st.markdown("""
                            ### 🤖 **Mistral Model Not Available**
                            
                            The AI model is not installed. Please run:
                            ```bash
                            ollama pull mistral:7b-instruct-q4_K_M
                            ```
                            
                            **Steps:**
                            1. Open a terminal/command prompt
                            2. Run the command above
                            3. Wait for download (~4.1 GB)
                            4. Refresh this page and try again
                        """)
                    elif 'timeout' in error_msg.lower():
                        st.markdown("""
                            ### ⏱️ **Generation Timed Out**
                            
                            The AI took too long to respond (>30 seconds).
                            
                            **Possible causes:**
                            - System is under heavy load
                            - GPU/CPU is busy with other tasks
                            - Network/Ollama connection issues
                            
                            **Try:**
                            1. Wait a moment and try again
                            2. Close other GPU-intensive applications
                            3. Restart Ollama: `ollama serve`
                        """)
                    elif 'connection' in error_msg.lower():
                        st.markdown("""
                            ### 🔌 **Connection Error**
                            
                            Cannot connect to Ollama service.
                            
                            **Please check:**
                            1. Ollama is running: `ollama serve`
                            2. Ollama is accessible: `ollama list`
                            3. No firewall blocking port 11434
                            
                            **Restart Ollama:**
                            ```bash
                            # Stop Ollama
                            taskkill /F /IM ollama.exe
                            
                            # Start Ollama
                            ollama serve
                            ```
                        """)
                    else:
                        st.markdown(f"""
                            ### 🔍 **Error Details**
                            
                            ```
                            {error_msg}
                            ```
                            
                            ### 💡 **Troubleshooting:**
                            1. **Check Ollama status:** Run `ollama list` in terminal
                            2. **Verify model:** Ensure mistral:7b-instruct-q4_K_M is installed
                            3. **Restart Ollama:** `ollama serve`
                            4. **Check resources:** Ensure enough RAM/VRAM (4GB+ needed)
                            5. **Try again:** Sometimes temporary issues resolve themselves
                            
                            If problems persist, try manual input with simpler details.
                        """)
                    
                    st.info("💡 **Alternative:** Try using manual input with simplified details to reduce AI processing load.")
    
    # Footer
    st.markdown("<hr style='border: 1px solid rgba(102, 126, 234, 0.3); margin-top: 3rem;'>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888;'>Built with ❤️ using Streamlit & Mistral 7B AI</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
