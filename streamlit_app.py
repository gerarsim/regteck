# streamlit_app.py - VERSION WITH OLLAMA LLM INTEGRATION
import streamlit as st
import streamlit.components.v1 as components
import sys
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import re
from pathlib import Path
import pandas as pd

# ============================================================================
# CONFIGURATION AND PATH SETUP
# ============================================================================

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Page configuration
try:
    st.set_page_config(
        page_title="LexAI - Regulatory Compliance Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except:
    pass

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

# Check environment variables for Ollama
USE_LLM = os.getenv('USE_LLM_ANALYZER', 'false').lower() == 'true'
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
LLM_MODE = os.getenv('LLM_ANALYSIS_MODE', 'hybrid')

logger.info(f"🔧 Ollama configuration: USE_LLM={USE_LLM}, HOST={OLLAMA_HOST}, MODEL={OLLAMA_MODEL}, MODE={LLM_MODE}")

# ============================================================================
# CORRECTED SCORE FORMATTING FUNCTIONS
# ============================================================================

def format_score_decimal(score: Any) -> float:
    """
    DEFINITIVE CORRECTED VERSION - Formats any score to proper decimal percentage

    Handles all problematic cases:
    - 4171 → 41.71% (divide by 100) - MAIN FIX
    - "89.23%" → 89.23%
    - 0.75 → 75.00% (multiply by 100)
    - 150.5 → 100.00% (cap at 100)
    - "invalid" → 0.00%
    - None → 0.00%
    - -50 → 0.00%
    """
    try:
        # Handle None or empty values
        if score is None or (isinstance(score, str) and not score.strip()):
            return 0.0

        # Convert to numeric value
        if isinstance(score, (int, float)):
            numeric_score = float(score)
        else:
            # Clean string input
            score_str = str(score).strip()
            # Remove all non-numeric characters except . and ,
            cleaned = re.sub(r'[^\d.,-]', '', score_str)

            # Replace comma with dot for decimal conversion
            cleaned = cleaned.replace(',', '.')

            if not cleaned:
                return 0.0

            # Handle multiple decimal points
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                if len(parts) > 2:
                    cleaned = ''.join(parts[:-1]) + '.' + parts[-1]

            try:
                numeric_score = float(cleaned)
            except ValueError:
                logger.warning(f"⚠️ Cannot convert '{score}' to number, using 0.0")
                return 0.0

        # Apply formatting logic based on value range
        if numeric_score < 0:
            return 0.0
        elif numeric_score > 10000:
            corrected = numeric_score / 10000
            return round(min(100.0, corrected), 2)
        elif numeric_score > 1000:
            corrected = numeric_score / 100
            return round(min(100.0, corrected), 2)
        elif numeric_score > 100.0:
            return 100.0
        elif numeric_score > 1.0:
            return round(numeric_score, 2)
        else:
            return round(numeric_score * 100.0, 2)

    except Exception as e:
        logger.error(f"❌ Error in format_score_decimal with '{score}': {e}")
        return 0.0

def fix_analysis_result_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    """Fixes all score formatting in an analysis result"""

    if not isinstance(result, dict):
        return result

    result = result.copy()

    score_fields = [
        'score', 'final_score', 'enhanced_score', 'base_score',
        'excellence_score', 'bonus_points', 'luxembourg_relevance',
        'overall_score', 'confidence_score'
    ]

    for field in score_fields:
        if field in result:
            result[field] = format_score_decimal(result[field])

    if 'issues' in result and isinstance(result['issues'], list):
        for issue in result['issues']:
            if isinstance(issue, dict):
                for score_field in ['confidence_score', 'weight', 'penalty_score']:
                    if score_field in issue:
                        issue[score_field] = format_score_decimal(issue[score_field])

    if 'final_score' not in result and 'score' in result:
        result['final_score'] = result['score']

    result['score_corrections_applied'] = True

    return result

# ============================================================================
# SAFE IMPORTS WITH OLLAMA PRIORITY
# ============================================================================

# Session Manager
try:
    from utils.session_manager import SessionManager
    SESSION_MANAGER_AVAILABLE = True
    logger.info("✅ Session Manager loaded successfully")
except ImportError as e:
    SESSION_MANAGER_AVAILABLE = False
    logger.warning(f"⚠️ Session Manager not available: {e}")

    class SessionManager:
        @classmethod
        def init_session(cls):
            defaults = {
                'current_analysis': None,
                'analysis_history': [],
                'language': 'fr',
                'current_page': 'home',
                'user': {},
                'authenticated': False
            }
            for key, value in defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = value

        @classmethod
        def get(cls, key, default=None):
            cls.init_session()
            return st.session_state.get(key, default)

        @classmethod
        def set(cls, key, value):
            cls.init_session()
            st.session_state[key] = value

        @classmethod
        def add_analysis(cls, analysis_data):
            cls.init_session()
            if 'analysis_history' not in st.session_state:
                st.session_state['analysis_history'] = []
            analysis_data['timestamp'] = datetime.now().isoformat()
            st.session_state['analysis_history'].insert(0, analysis_data)

# Text extraction
try:
    from utils.text_extraction import extract_text
    TEXT_EXTRACTION_AVAILABLE = True
    logger.info("✅ Text extraction loaded successfully")
except ImportError:
    TEXT_EXTRACTION_AVAILABLE = False

    def extract_text(file, detect_lang=False):
        if hasattr(file, 'read'):
            try:
                content = file.read().decode('utf-8', errors='ignore')
                return content, 'fr' if detect_lang else None
            except:
                return str(file), 'fr' if detect_lang else None
        return str(file), 'fr' if detect_lang else None

# ============================================================================
# OLLAMA LLM ANALYZER - PRIORITY
# ============================================================================

ANALYZER_TYPE = "unknown"
OLLAMA_AVAILABLE = False

if USE_LLM:
    logger.info("🤖 Attempting to load Ollama LLM analyzer...")
    try:
        from utils.llm_analyzer_ollama import (
            analyze_with_local_llm,
            check_ollama_status,
            OllamaConfig
        )

        # Check if Ollama is actually available
        ollama_status = check_ollama_status()

        if ollama_status.get('running', False):
            OLLAMA_AVAILABLE = True
            ANALYZER_TYPE = "ollama_llm"

            # Create analysis function wrapper
            def analyze_regulatory_compliance(text, doc_type="auto", language="auto"):
                """Wrapper for Ollama LLM analysis"""
                logger.info(f"🤖 Using Ollama LLM: {OLLAMA_MODEL} (mode: {LLM_MODE})")
                result = analyze_with_local_llm(
                    text=text,
                    doc_type=doc_type,
                    language=language,
                    model=OLLAMA_MODEL,
                    mode=LLM_MODE
                )
                return fix_analysis_result_scores(result)

            def check_ollama_installation():
                return ollama_status

            logger.info(f"✅ Ollama LLM loaded: {OLLAMA_MODEL} at {OLLAMA_HOST}")
            logger.info(f"📊 Available models: {ollama_status.get('models', [])}")

        else:
            logger.warning(f"⚠️ Ollama not running at {OLLAMA_HOST}, falling back to rule-based")
            OLLAMA_AVAILABLE = False

    except ImportError as e:
        logger.warning(f"⚠️ Ollama analyzer not found: {e}")
        OLLAMA_AVAILABLE = False
    except Exception as e:
        logger.error(f"❌ Error loading Ollama: {e}")
        OLLAMA_AVAILABLE = False

# Fallback to rule-based engine if Ollama not available
if not OLLAMA_AVAILABLE:
    logger.info("🔧 Loading rule-based engine as fallback...")

    try:
        from engine import analyze_document_compliance

        def analyze_regulatory_compliance(text, doc_type="auto", language="auto"):
            result = analyze_document_compliance(text, doc_type, language)
            return fix_analysis_result_scores(result)

        def check_ollama_installation():
            return {
                "installed": True,
                "running": True,
                "models": ["local_engine_v4.0_decimal_corrected"],
                "engine_type": "local_json_based",
                "status": "operational"
            }

        ANALYZER_TYPE = "rule_based"
        logger.info("✅ Rule-based engine loaded successfully")

    except ImportError:
        try:
            from utils.llm_analyzer import analyze_regulatory_compliance as original_analyze
            from utils.llm_analyzer import check_ollama_installation as original_check

            def analyze_regulatory_compliance(text, doc_type="auto", language="auto"):
                result = original_analyze(text, doc_type, language)
                return fix_analysis_result_scores(result)

            def check_ollama_installation():
                return original_check()

            ANALYZER_TYPE = "utils_analyzer"
            logger.info("✅ Utils analyzer loaded")

        except ImportError:
            def analyze_regulatory_compliance(text, doc_type="auto", language="auto"):
                return fix_analysis_result_scores({
                    "score": 70.0,
                    "issues": [],
                    "recommendations": ["Install analysis engine"],
                    "overall_assessment": "Fallback mode"
                })

            def check_ollama_installation():
                return {"installed": False, "running": False}

            ANALYZER_TYPE = "fallback"
            logger.warning("⚠️ Using basic fallback analyzer")

logger.info(f"📊 Final analyzer type: {ANALYZER_TYPE}")
logger.info(f"🤖 Ollama available: {OLLAMA_AVAILABLE}")

# ============================================================================
# INTERNATIONALIZATION
# ============================================================================

def get_localized_text(key: str, language: str = None) -> str:
    """Get localized text"""
    if language is None:
        language = SessionManager.get('language', 'fr')

    translations = {
        'app_title': {
            'fr': 'Assistant de Conformité Réglementaire',
            'en': 'Regulatory Compliance Assistant',
            'de': 'Regulatorischer Compliance-Assistent',
            'es': 'Asistente de Cumplimiento Normativo'
        },
        'document_analysis': {
            'fr': 'Analyse Document',
            'en': 'Document Analysis',
            'de': 'Dokumentanalyse',
            'es': 'Análisis de Documento'
        },
        'upload_document': {
            'fr': 'Déposez votre document',
            'en': 'Upload your document',
            'de': 'Laden Sie Ihr Dokument hoch',
            'es': 'Suba su documento'
        },
        'analyze_document': {
            'fr': 'ANALYSER LE DOCUMENT',
            'en': 'ANALYZE DOCUMENT',
            'de': 'DOKUMENT ANALYSIEREN',
            'es': 'ANALIZAR DOCUMENTO'
        },
        'analysis_in_progress': {
            'fr': 'Analyse en cours avec IA...' if OLLAMA_AVAILABLE else 'Analyse en cours...',
            'en': 'AI Analysis in progress...' if OLLAMA_AVAILABLE else 'Analysis in progress...',
            'de': 'KI-Analyse läuft...' if OLLAMA_AVAILABLE else 'Analyse läuft...',
            'es': 'Análisis IA en progreso...' if OLLAMA_AVAILABLE else 'Análisis en progreso...'
        }
    }

    if key in translations and language in translations[key]:
        return translations[key][language]

    return key.replace('_', ' ').title()

def render_language_selector():
    """Language selector"""
    current_lang = SessionManager.get('language', 'fr')

    languages = {
        'fr': '🇫🇷 Français',
        'en': '🇬🇧 English',
        'de': '🇩🇪 Deutsch',
        'es': '🇪🇸 Español'
    }

    selected_lang = st.selectbox(
        "🌍 Language",
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current_lang),
        key="main_language_selector"
    )

    if selected_lang != current_lang:
        SessionManager.set('language', selected_lang)
        st.success(f"✅ Language changed: {languages[selected_lang]}")
        time.sleep(0.1)
        st.rerun()

# ============================================================================
# ENHANCED ANALYSIS WITH OLLAMA
# ============================================================================

def analyze_document_with_ai(text: str, document_type: str = "auto", language: str = "fr") -> Dict[str, Any]:
    """Enhanced analysis using Ollama or fallback"""
    start_time = time.time()

    try:
        logger.info(f"📊 Starting analysis with {ANALYZER_TYPE}")

        result = analyze_regulatory_compliance(text, document_type, language)

        result['processing_time'] = round(time.time() - start_time, 3)
        result['analyzer_type'] = ANALYZER_TYPE
        result['ollama_powered'] = OLLAMA_AVAILABLE
        result['ollama_model'] = OLLAMA_MODEL if OLLAMA_AVAILABLE else None
        result['analysis_mode'] = LLM_MODE if OLLAMA_AVAILABLE else 'rule_based'

        return fix_analysis_result_scores(result)

    except Exception as e:
        logger.error(f"❌ Analysis error: {e}")
        return fix_analysis_result_scores({
            "score": 50.0,
            "issues": [],
            "recommendations": [f"Analysis error: {str(e)}"],
            "overall_assessment": "Error during analysis",
            "processing_time": round(time.time() - start_time, 3),
            "analyzer_type": "error_fallback"
        })

# ============================================================================
# UI COMPONENTS
# ============================================================================

def create_enhanced_upload_section():
    """Enhanced upload section"""
    current_lang = SessionManager.get('language', 'fr')

    st.markdown(f"### 📄 {get_localized_text('document_analysis')}")

    # Show analyzer status
    if OLLAMA_AVAILABLE:
        st.success(f"🤖 AI-Powered Analysis with {OLLAMA_MODEL}")
    else:
        st.info(f"📋 Rule-Based Analysis")

    uploaded_file = st.file_uploader(
        f"📁 {get_localized_text('upload_document')}",
        type=['pdf', 'docx', 'txt'],
        help="Formats: PDF, DOCX, TXT (max 10MB)"
    )

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        if st.button(f"🚀 {get_localized_text('analyze_document')}", type="primary", use_container_width=True):

            with st.spinner(get_localized_text('analysis_in_progress')):
                try:
                    # Extract text
                    if TEXT_EXTRACTION_AVAILABLE:
                        text_content, detected_lang = extract_text(uploaded_file, detect_lang=True)
                    else:
                        text_content = uploaded_file.read().decode('utf-8', errors='ignore')
                        detected_lang = current_lang

                    if not text_content.strip():
                        st.error("❌ Cannot extract text from file")
                        return

                    word_count = len(text_content.split())
                    st.info(f"📝 Extracted: {word_count} words, Language: {detected_lang}")

                    # Analyze with Ollama or fallback
                    result = analyze_document_with_ai(
                        text=text_content,
                        document_type="auto",
                        language=current_lang
                    )

                    # Save to history
                    analysis_data = result.copy()
                    analysis_data['file_name'] = uploaded_file.name
                    analysis_data['timestamp'] = datetime.now().isoformat()

                    SessionManager.add_analysis(analysis_data)

                    # Show results
                    final_score = format_score_decimal(result.get('final_score', result.get('score', 0)))
                    processing_time = result.get('processing_time', 0)

                    analyzer_info = f" with {OLLAMA_MODEL}" if OLLAMA_AVAILABLE else ""
                    st.success(f"✅ Analysis completed{analyzer_info} in {processing_time:.2f}s - Score: {final_score:.2f}%")
                    st.balloons()

                    display_analysis_results(result, uploaded_file.name)

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    logger.error(f"Analysis error: {e}")

def display_analysis_results(result: Dict[str, Any], filename: str):
    """Display analysis results"""

    st.markdown("---")
    st.markdown("## 📋 Analysis Results")
    st.markdown(f"### 📄 {filename}")

    result = fix_analysis_result_scores(result)

    score = format_score_decimal(result.get('final_score', result.get('score', 0)))
    issues_count = len(result.get('issues', []))
    processing_time = result.get('processing_time', 0.0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score_color = "🟢" if score >= 80.0 else "🟡" if score >= 60.0 else "🔴"
        st.metric("📊 Score", f"{score:.2f}%", f"{score_color}")

        if score >= 90.0:
            st.caption("Excellent")
        elif score >= 70.0:
            st.caption("Good")
        else:
            st.caption("Needs work")

    with col2:
        st.metric("⚠️ Issues", str(issues_count))

    with col3:
        st.metric("⏱️ Time", f"{processing_time:.2f}s")

    with col4:
        analyzer = "🤖 AI" if result.get('ollama_powered', False) else "📋 Rules"
        st.metric("Engine", analyzer)

    # Show AI model info if used
    if result.get('ollama_powered', False):
        st.info(f"🤖 Analyzed with: {result.get('ollama_model', 'Unknown')} (Mode: {result.get('analysis_mode', 'unknown')})")

    # Issues
    issues = result.get('issues', [])
    if issues:
        st.markdown("### ⚠️ Compliance Issues")

        for i, issue in enumerate(issues, 1):
            description = issue.get('description', 'Not specified')
            severity = issue.get('severity', 'medium')
            confidence = format_score_decimal(issue.get('confidence_score', 0))

            severity_icons = {'critical': '🚨', 'high': '⚠️', 'medium': '🟡', 'low': '🔵'}
            icon = severity_icons.get(severity, '⚠️')

            with st.expander(f"{icon} {description}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Severity:** {severity}")
                    st.write(f"**Confidence:** {confidence:.2f}%")

                with col2:
                    st.write(f"**Basis:** {issue.get('regulatory_basis', 'N/A')}")
                    st.write(f"**Action:** {issue.get('suggested_action', 'TBD')}")
    else:
        st.success("### ✅ No compliance issues detected!")

    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        st.markdown("### 💡 Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.info(f"**{i}.** {rec}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class LexAIApp:
    """Main LexAI Application with Ollama support"""

    def __init__(self):
        # Dark theme
        st.markdown("""
        <style>
        .stApp {
            background-color: #0a0a0a !important;
            color: #ffffff !important;
        }
        .main .block-container {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #404040 !important;
            border-radius: 12px !important;
            padding: 2rem !important;
        }
        .stButton > button {
            background-color: #3b82f6 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        .stButton > button:hover {
            background-color: #2563eb !important;
        }
        </style>
        """, unsafe_allow_html=True)

        SessionManager.init_session()
        logger.info("✅ LexAI initialized")

    def create_header(self):
        """Create header with Ollama status"""
        current_lang = SessionManager.get('language', 'fr')

        st.markdown("# ⚖️ LexAI - " + get_localized_text('app_title'))

        # Status
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if OLLAMA_AVAILABLE:
                st.markdown(f"**🤖 AI Engine**\n{OLLAMA_MODEL}\n✅ Active")
            else:
                st.markdown(f"**📋 Engine**\nRule-Based\n✅ Active")

        with col2:
            mode_display = LLM_MODE.capitalize() if OLLAMA_AVAILABLE else "Standard"
            st.markdown(f"**🔧 Mode**\n{mode_display}")

        with col3:
            session_count = len(SessionManager.get('analysis_history', []))
            st.markdown(f"**📊 Analyses**\n{session_count}")

        with col4:
            st.markdown(f"**🌙 Theme**\nDark Mode")

        st.markdown("---")
        render_language_selector()

        # Ollama status message
        if OLLAMA_AVAILABLE:
            st.success(f"🤖 AI-Powered Analysis Enabled: {OLLAMA_MODEL} ({LLM_MODE} mode)")
        else:
            st.info("📋 Using Rule-Based Analysis Engine")

    def run(self):
        """Run application"""
        self.create_header()

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs([
            f"📄 {get_localized_text('document_analysis')}",
            "📊 History",
            "🔧 Settings"
        ])

        with tab1:
            create_enhanced_upload_section()

        with tab2:
            self.render_history()

        with tab3:
            self.render_settings()

    def render_history(self):
        """Render history"""
        st.markdown("### 📊 Analysis History")

        history = SessionManager.get('analysis_history', [])

        if history:
            for i, analysis in enumerate(history[:10]):
                analysis = fix_analysis_result_scores(analysis)
                score = format_score_decimal(analysis.get('final_score', 0))
                timestamp = analysis.get('timestamp', '')[:16].replace('T', ' ')
                file_name = analysis.get('file_name', 'File')

                with st.expander(f"📄 {file_name} - {score:.2f}% - {timestamp}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Score:** {score:.2f}%")
                        st.write(f"**Issues:** {len(analysis.get('issues', []))}")

                    with col2:
                        analyzer = "🤖 AI" if analysis.get('ollama_powered', False) else "📋 Rules"
                        st.write(f"**Engine:** {analyzer}")
                        st.write(f"**Time:** {analysis.get('processing_time', 0):.2f}s")

            if st.button("🧹 Clear History"):
                SessionManager.set('analysis_history', [])
                st.success("✅ Cleared")
                st.rerun()
        else:
            st.info("No analyses yet")

    def render_settings(self):
        """Render settings with Ollama info"""
        st.markdown("### ⚙️ System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🤖 AI Configuration**")
            st.info(f"Ollama: {'✅ Connected' if OLLAMA_AVAILABLE else '❌ Not available'}")
            if OLLAMA_AVAILABLE:
                st.info(f"Host: {OLLAMA_HOST}")
                st.info(f"Model: {OLLAMA_MODEL}")
                st.info(f"Mode: {LLM_MODE}")
            else:
                st.warning("To enable AI analysis:")
                st.code("# Install Ollama\ncurl -fsSL https://ollama.com/install.sh | sh\n\n# Pull model\nollama pull llama3.2:3b")

        with col2:
            st.markdown("**📊 System Status**")

            status_items = [
                ("Analyzer", ANALYZER_TYPE),
                ("Ollama", "✅" if OLLAMA_AVAILABLE else "❌"),
                ("Text Extraction", "✅" if TEXT_EXTRACTION_AVAILABLE else "❌"),
                ("Session Manager", "✅" if SESSION_MANAGER_AVAILABLE else "❌")
            ]

            for name, status in status_items:
                st.write(f"**{name}:** {status}")

        # Test Ollama connection
        if st.button("🧪 Test Ollama Connection"):
            if OLLAMA_AVAILABLE:
                try:
                    status = check_ollama_status()
                    st.success(f"✅ Connected to Ollama")
                    st.json(status)
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
            else:
                st.warning("⚠️ Ollama not configured")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        app = LexAIApp()
        app.run()

        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🌙 LexAI v2.0**")
            st.caption("Compliance Assistant")

        with col2:
            if OLLAMA_AVAILABLE:
                st.markdown(f"**🤖 AI-Powered**")
                st.caption(f"{OLLAMA_MODEL}")
            else:
                st.markdown("**📋 Rule-Based**")
                st.caption("Standard Engine")

        with col3:
            st.markdown(f"**Analyzer:** {ANALYZER_TYPE}")
            st.caption(datetime.now().strftime("%Y-%m-%d"))

    except Exception as e:
        st.error(f"🚨 Error: {e}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()