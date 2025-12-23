# utils/__init__.py - OPTIMIZED WITH CENTRALIZED FALLBACKS
"""
LexAI utilities with fully centralized fallback system
NO MORE FALLBACK DUPLICATION - Everything is centralized here!
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CENTRALIZED FALLBACK SYSTEM - SINGLE SOURCE OF TRUTH
# ============================================================================

class CentralizedFallbackManager:
    """Master fallback manager - eliminates ALL duplication"""
    
    def __init__(self):
        self.module_status = {}
        self.fallback_count = 0
    
    def register_fallback(self, module_name: str, function_name: str):
        """Register when a fallback is used"""
        self.fallback_count += 1
        if module_name not in self.module_status:
            self.module_status[module_name] = {'available': False, 'fallbacks': []}
        self.module_status[module_name]['fallbacks'].append(function_name)
        logger.debug(f"📝 Fallback used: {module_name}.{function_name}")
    
    def mark_available(self, module_name: str):
        """Mark module as available"""
        if module_name not in self.module_status:
            self.module_status[module_name] = {'available': True, 'fallbacks': []}
        else:
            self.module_status[module_name]['available'] = True
    
    def get_status(self):
        """Get complete fallback status"""
        return {
            'total_fallbacks_used': self.fallback_count,
            'modules': self.module_status,
            'available_modules': [k for k, v in self.module_status.items() if v.get('available', False)],
            'fallback_modules': [k for k, v in self.module_status.items() if not v.get('available', False)]
        }

# Initialize global fallback manager
_fallback_manager = CentralizedFallbackManager()

# ============================================================================
# CENTRALIZED FALLBACK FUNCTIONS - NO DUPLICATION
# ============================================================================

# Language fallbacks
def _fallback_detect_language(text: str) -> str:
    """Centralized language detection fallback"""
    _fallback_manager.register_fallback('language', 'detect_language')
    french_indicators = ['le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'et ', 'ou ', 'à ']
    english_indicators = ['the ', 'and ', 'or ', 'of ', 'to ', 'in ', 'for ', 'with ']
    
    text_lower = text.lower()
    french_count = sum(1 for indicator in french_indicators if indicator in text_lower)
    english_count = sum(1 for indicator in english_indicators if indicator in text_lower)
    
    return "fr" if french_count >= english_count else "en"

def _fallback_get_translation(key: str, language: str = "fr") -> str:
    """Centralized translation fallback"""
    _fallback_manager.register_fallback('language', 'get_translation')
    translations = {
        'compliance_score': {'fr': 'Score de Conformité', 'en': 'Compliance Score'},
        'analysis_complete': {'fr': 'Analyse Terminée', 'en': 'Analysis Complete'},
        'issues_found': {'fr': 'Problèmes Détectés', 'en': 'Issues Found'},
        'recommendations': {'fr': 'Recommandations', 'en': 'Recommendations'}
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return key.replace('_', ' ').title()

def _fallback_get_available_languages() -> Dict[str, str]:
    """Centralized available languages fallback"""
    _fallback_manager.register_fallback('language', 'get_available_languages')
    return {"fr": "Français", "en": "English"}

# Text extraction fallbacks
def _fallback_extract_text(file, detect_lang=False, **kwargs):
    """Centralized text extraction fallback"""
    _fallback_manager.register_fallback('text_extraction', 'extract_text')
    
    if hasattr(file, 'read'):
        try:
            content = file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            lang = _fallback_detect_language(content) if detect_lang else None
            return content, lang
        except Exception as e:
            return f"Text extraction error: {e}", "fr"
    return str(file), "fr"

class _FallbackDocumentProcessor:
    """Centralized document processor fallback"""
    def __init__(self):
        _fallback_manager.register_fallback('text_extraction', 'DocumentProcessor')

# Data manager fallbacks
class _FallbackComplianceDataManager:
    """Centralized data manager fallback with actual file loading"""
    
    def __init__(self):
        _fallback_manager.register_fallback('data_manager', 'ComplianceDataManager')
        self._cache = {}
        self._load_data_files()
    
    def _load_data_files(self):
        """Load actual data files if available"""
        data_files = [
            'analyses.json', 'compliance_penalties.json', 'compliance_rules.json',
            'cross_border_regulations.json', 'dynamic_rules.json', 'financial_institutions.json',
            'issue_descriptions.json', 'lux_keywords.json', 'regulations.json',
            'reporting_requirements.json', 'sanctions_lists.json'
        ]
        
        for filename in data_files:
            filepath = os.path.join('data', filename)
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        self._cache[filename] = json.loads(content) if content else {}
                else:
                    self._cache[filename] = {}
            except Exception:
                self._cache[filename] = {}
    
    def get_compliance_rules(self): return self._cache.get('compliance_rules.json', {})
    def get_lux_keywords(self): return self._cache.get('lux_keywords.json', {})
    def get_compliance_penalties(self): return self._cache.get('compliance_penalties.json', {})
    def get_regulations(self): return self._cache.get('regulations.json', {})
    def get_financial_institutions(self): return self._cache.get('financial_institutions.json', {})
    def get_analyses(self): return self._cache.get('analyses.json', {})
    def get_cross_border_regulations(self): return self._cache.get('cross_border_regulations.json', {})
    def get_dynamic_rules(self): return self._cache.get('dynamic_rules.json', {})
    def get_issue_descriptions(self): return self._cache.get('issue_descriptions.json', {})
    def get_reporting_requirements(self): return self._cache.get('reporting_requirements.json', {})
    def get_sanctions_lists(self): return self._cache.get('sanctions_lists.json', {})
    
    def get_data_statistics(self):
        total_files = len(self._cache)
        files_with_data = sum(1 for data in self._cache.values() if data)
        return {
            '_summary': {
                'total_files': total_files,
                'files_with_data': files_with_data,
                'total_records': sum(len(data) if isinstance(data, dict) else 0 for data in self._cache.values()),
                'all_files_loaded': files_with_data == total_files
            }
        }
    
    def refresh_all_data(self): 
        self._load_data_files()

# Session manager fallback
class _FallbackSessionManager:
    """Centralized session manager fallback"""
    
    def __init__(self):
        _fallback_manager.register_fallback('session_manager', 'SessionManager')
    
    @staticmethod
    def init_session():
        try:
            import streamlit as st
            if 'initialized' not in st.session_state:
                st.session_state.initialized = True
                st.session_state.analysis_history = []
        except ImportError:
            pass
    
    @staticmethod
    def get(key, default=None):
        try:
            import streamlit as st
            return st.session_state.get(key, default)
        except ImportError:
            return default
    
    @staticmethod
    def set(key, value):
        try:
            import streamlit as st
            st.session_state[key] = value
        except ImportError:
            pass
    
    @staticmethod
    def add_analysis(data):
        try:
            import streamlit as st
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(data)
        except ImportError:
            pass

# LLM analyzer fallbacks
def _fallback_analyze_regulatory_compliance(*args, **kwargs):
    """Centralized compliance analysis fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'analyze_regulatory_compliance')
    return {
        "score": 0.7,
        "issues": [],
        "recommendations": ["Module d'analyse IA non disponible - utilisation des règles de base"],
        "analysis_engine": "fallback",
        "statistics": {
            "analysis_method": "basic_fallback",
            "ai_enabled": False,
            "processing_time": 0.1
        }
    }

def _fallback_identify_issues(*args, **kwargs):
    """Centralized issue identification fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'identify_issues')
    return [], 0.7

def _fallback_check_ollama_installation():
    """Centralized Ollama check fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'check_ollama_installation')
    return {"installed": False, "running": False, "models": []}

def _fallback_load_your_data_files():
    """Centralized data loading fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'load_your_data_files')
    return {}

# JSON utilities fallbacks
def _fallback_load_json(path, default=None, **kwargs):
    """Centralized JSON loading fallback"""
    _fallback_manager.register_fallback('json_utils', 'load_json')
    if default is None:
        default = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default

def _fallback_save_json(data, path, **kwargs):
    """Centralized JSON saving fallback"""
    _fallback_manager.register_fallback('json_utils', 'save_json')
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

# Styling fallbacks
def _fallback_styling_function(*args, **kwargs):
    """Universal styling fallback"""
    _fallback_manager.register_fallback('styling', 'various_functions')
    pass

def _fallback_get_card_html(*args, **kwargs):
    """Centralized card HTML fallback"""
    _fallback_manager.register_fallback('styling', 'get_card_html')
    return "<div>Card not available</div>"

# ============================================================================
# SAFE IMPORTS WITH CENTRALIZED FALLBACKS
# ============================================================================

# Language utilities
try:
    from .language import detect_language, get_translation, get_available_languages
    _fallback_manager.mark_available('language')
    logger.debug("✅ Language utilities loaded")
except ImportError as e:
    logger.warning(f"⚠️ Language utilities fallback: {e}")
    detect_language = _fallback_detect_language
    get_translation = _fallback_get_translation
    get_available_languages = _fallback_get_available_languages

# Text extraction
try:
    from .text_extraction import extract_text, DocumentProcessor
    _fallback_manager.mark_available('text_extraction')
    logger.debug("✅ Text extraction loaded")
except ImportError as e:
    logger.warning(f"⚠️ Text extraction fallback: {e}")
    extract_text = _fallback_extract_text
    DocumentProcessor = _FallbackDocumentProcessor

# Data manager
try:
    from .data_manager import ComplianceDataManager
    _fallback_manager.mark_available('data_manager')
    logger.debug("✅ Data manager loaded")
except ImportError as e:
    logger.warning(f"⚠️ Data manager fallback: {e}")
    ComplianceDataManager = _FallbackComplianceDataManager

# Session manager
try:
    from .session_manager import SessionManager
    _fallback_manager.mark_available('session_manager')
    logger.debug("✅ Session manager loaded")
except ImportError as e:
    logger.warning(f"⚠️ Session manager fallback: {e}")
    SessionManager = _FallbackSessionManager()

# LLM analyzer
try:
    from .llm_analyzer import (
        analyze_regulatory_compliance, 
        identify_issues,
        check_ollama_installation,
        load_your_data_files
    )
    _fallback_manager.mark_available('llm_analyzer')
    logger.debug("✅ LLM analyzer loaded")
except ImportError as e:
    logger.warning(f"⚠️ LLM analyzer fallback: {e}")
    analyze_regulatory_compliance = _fallback_analyze_regulatory_compliance
    identify_issues = _fallback_identify_issues
    check_ollama_installation = _fallback_check_ollama_installation
    load_your_data_files = _fallback_load_your_data_files

# Config
try:
    from .config import config, get_data_path
    _fallback_manager.mark_available('config')
    logger.debug("✅ Config loaded")
except ImportError as e:
    logger.warning(f"⚠️ Config fallback: {e}")
    def get_data_path(filename="", subfolder=None):
        return os.path.join("data", subfolder or "", filename)
    
    class config:
        @staticmethod
        def get_data_path(filename="", subfolder=None):
            return get_data_path(filename, subfolder)

# JSON utilities
try:
    from .json_utils import load_json, save_json
    _fallback_manager.mark_available('json_utils')
    logger.debug("✅ JSON utilities loaded")
except ImportError as e:
    logger.warning(f"⚠️ JSON utilities fallback: {e}")
    load_json = _fallback_load_json
    save_json = _fallback_save_json

# Styling utilities
try:
    from .styling import (
        set_custom_style, render_compliance_score, render_processing_animation,
        get_card_html, apply_emergency_visibility_fix, create_optimized_banner,
        create_simple_metric_card, create_status_indicator, initialize_demo_styling,
        check_styling_compatibility
    )
    _fallback_manager.mark_available('styling')
    logger.debug("✅ Styling utilities loaded")
except ImportError as e:
    logger.warning(f"⚠️ Styling utilities fallback: {e}")
    set_custom_style = _fallback_styling_function
    render_compliance_score = _fallback_styling_function
    render_processing_animation = _fallback_styling_function
    get_card_html = _fallback_get_card_html
    apply_emergency_visibility_fix = _fallback_styling_function
    create_optimized_banner = _fallback_styling_function
    create_simple_metric_card = _fallback_styling_function
    create_status_indicator = _fallback_styling_function
    initialize_demo_styling = lambda: {}
    check_styling_compatibility = _fallback_styling_function
    # Backward compatibility
    create_emergency_banner = create_optimized_banner
    create_emergency_metric_card = create_simple_metric_card

# ============================================================================
# UTILITY FUNCTIONS (OPTIMIZED)
# ============================================================================

def get_asset_path(filename="", subfolder=None):
    """Get asset path"""
    return os.path.join("assets", subfolder or "", filename)

def get_log_path(filename="", subfolder=None):
    """Get log path"""
    return os.path.join("logs", subfolder or "", filename)

def get_model_path(filename="", subfolder=None):
    """Get model path"""
    return os.path.join("models_cache", subfolder or "", filename)

def ensure_data_file(path, default_content=None):
    """Ensure data file exists"""
    if default_content is None:
        default_content = {}
    
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_json(data=default_content, path=path)
        return True
    return False

def check_your_data_files():
    """Check which data files are available"""
    your_files = [
        "analyses.json", "compliance_penalties.json", "compliance_rules.json",
        "cross_border_regulations.json", "dynamic_rules.json", "financial_institutions.json",
        "issue_descriptions.json", "lux_keywords.json", "regulations.json",
        "reporting_requirements.json", "sanctions_lists.json"
    ]
    
    status = {}
    for filename in your_files:
        filepath = os.path.join("data", filename)
        status[filename] = {
            'exists': os.path.exists(filepath),
            'path': filepath,
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0
        }
    
    return status

def get_fallback_status():
    """Get complete fallback system status"""
    return _fallback_manager.get_status()

# ============================================================================
# BACKWARD COMPATIBILITY (OPTIMIZED)
# ============================================================================

def find_non_conformities_semantic(text: str, rules: List[Dict], **kwargs) -> List[Tuple]:
    """Legacy compatibility function"""
    try:
        issues, confidence = identify_issues(text)
        results = []
        for issue in issues:
            result_tuple = (
                issue.get("description", "")[:100],
                issue.get("regulatory_basis", ""),
                issue.get("rule_id", ""),
                issue.get("confidence_score", confidence),
                issue.get("severity", "medium")
            )
            results.append(result_tuple)
        return results
    except Exception:
        return []

def guess_doc_type(text: str, language: str = "fr") -> str:
    """Legacy document type detection"""
    text_lower = text.lower()
    if any(word in text_lower for word in ['contrat', 'contract']):
        return 'contract'
    elif any(word in text_lower for word in ['politique', 'policy']):
        return 'policy'
    elif any(word in text_lower for word in ['rapport', 'report']):
        return 'report'
    return 'auto'

def get_regulatory_rules(regulation_type: str = "all") -> List[Dict]:
    """Get regulatory rules using data manager"""
    try:
        data_manager = ComplianceDataManager()
        return data_manager.get_rules(regulation_type) if hasattr(data_manager, 'get_rules') else []
    except Exception:
        return []

def validate_document_compliance(text: str, doc_type: str = "auto", language: str = "auto") -> Dict[str, Any]:
    """Legacy compliance validation"""
    try:
        return analyze_regulatory_compliance(text, doc_type=doc_type, language=language)
    except Exception as e:
        return {
            "score": 0.0,
            "issues": [],
            "recommendations": [f"Error: {str(e)}"],
            "doc_type": doc_type,
            "language": language
        }

# ============================================================================
# PACKAGE INFO AND INITIALIZATION
# ============================================================================

__version__ = "2.0.0"
__author__ = "LexAI Team"

PACKAGE_INFO = {
    "name": "LexAI Utils",
    "version": __version__,
    "description": "Complete utilities with centralized fallbacks for LexAI",
    "fallback_system": "centralized",
    "data_files_supported": [
        "analyses.json", "compliance_penalties.json", "compliance_rules.json",
        "cross_border_regulations.json", "dynamic_rules.json", 
        "financial_institutions.json", "issue_descriptions.json",
        "lux_keywords.json", "regulations.json", 
        "reporting_requirements.json", "sanctions_lists.json"
    ]
}

def get_package_info() -> Dict[str, Any]:
    """Get package information with fallback status"""
    info = PACKAGE_INFO.copy()
    info['data_files_status'] = check_your_data_files()
    info['fallback_status'] = get_fallback_status()
    return info

def check_dependencies() -> Dict[str, bool]:
    """Check available dependencies"""
    deps = {}
    
    data_status = check_your_data_files()
    deps['your_data_files'] = sum(1 for f in data_status.values() if f['exists']) >= 5
    
    for module in ['requests', 'pypdf', 'docx']:
        try:
            __import__(module)
            deps[module] = True
        except ImportError:
            deps[module] = False
    
    return deps

# Initialize package
try:
    # Create essential directories
    for directory in ['data', 'logs', 'assets']:
        os.makedirs(directory, exist_ok=True)
    
    # Check data files
    data_status = check_your_data_files()
    available_files = sum(1 for f in data_status.values() if f['exists'])
    
    # Get fallback status
    fallback_status = get_fallback_status()
    
    logger.info(f"✅ LexAI Utils v{__version__} initialized with centralized fallbacks")
    logger.info(f"📊 Data files: {available_files}/11 available")
    logger.info(f"🔄 Fallbacks: {fallback_status['total_fallbacks_used']} used")
    
except Exception as e:
    logger.error(f"❌ Package initialization failed: {e}")

# ============================================================================
# COMPLETE EXPORTS LIST
# ============================================================================

__all__ = [
    # Core functions
    "analyze_regulatory_compliance", "identify_issues", "detect_language",
    "get_translation", "extract_text", "load_your_data_files",
    
    # Path utilities
    "get_data_path", "get_asset_path", "get_log_path", "get_model_path",
    
    # Data utilities
    "load_json", "save_json", "ensure_data_file", "check_your_data_files",
    
    # Styling utilities
    "apply_emergency_visibility_fix", "create_optimized_banner",
    "create_simple_metric_card", "create_status_indicator", "initialize_demo_styling",
    "check_styling_compatibility", "set_custom_style", "render_compliance_score",
    "render_processing_animation", "get_card_html",
    # Backward compatibility
    "create_emergency_banner", "create_emergency_metric_card",
    
    # Classes
    "ComplianceDataManager", "SessionManager", "DocumentProcessor",
    
    # Backward compatibility
    "find_non_conformities_semantic", "guess_doc_type", "get_regulatory_rules",
    "validate_document_compliance",
    
    # Package info and system
    "get_package_info", "check_dependencies", "get_fallback_status",
    "PACKAGE_INFO", "__version__"
]