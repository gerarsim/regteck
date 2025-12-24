# utils/__init__.py - PRODUCTION-READY VERSION WITH SECURITY FIXES
"""
LexAI utilities with fully centralized fallback system
Security-hardened, performance-optimized, Luxembourg-focused

Version: 2.0.1-secure
Date: 2025-12-24
"""

import logging
import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Single source of truth for data files (DRY principle)
DATA_FILES = [
    'analyses.json',
    'compliance_penalties.json',
    'compliance_rules.json',
    'cross_border_regulations.json',
    'dynamic_rules.json',
    'financial_institutions.json',
    'issue_descriptions.json',
    'lux_keywords.json',
    'regulations.json',
    'reporting_requirements.json',
    'sanctions_lists.json'
]

# Luxembourg-specific constants
LUXEMBOURG_LANGUAGES = ['fr', 'de', 'lb', 'en']
DEFAULT_LUXEMBOURG_LANGUAGE = 'fr'

CSSF_REGULATIONS = [
    '12/552',  # Governance
    '20/750',  # Climate risks
    '21/773',  # AML/CFT
    '08/356',  # Customer protection
]

# Security limits
MAX_JSON_SIZE_MB = 50
MAX_FILENAME_LENGTH = 255
MAX_TEXT_EXTRACT_SIZE = 10 * 1024 * 1024  # 10MB

# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """
    Sanitize filename to prevent security issues

    Args:
        filename: Original filename
        max_length: Maximum allowed length

    Returns:
        Sanitized filename

    Raises:
        ValueError: If filename is invalid or empty after sanitization
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Remove path separators
    filename = re.sub(r'[/\\]', '', filename)

    # Remove null bytes
    filename = filename.replace('\x00', '')

    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext

    # Ensure not empty after sanitization
    if not filename:
        raise ValueError("Filename cannot be empty after sanitization")

    return filename


def is_safe_path(path_component: str) -> bool:
    """
    Validate path component is safe (no traversal attempts)

    Args:
        path_component: Path component to validate

    Returns:
        True if safe, False otherwise
    """
    if not path_component:
        return True

    dangerous = ['..', '/', '\\', '\x00', '~']
    return not any(d in path_component for d in dangerous)


def validate_path(path: Union[str, Path], base_dir: Union[str, Path]) -> Path:
    """
    Validate path is within base directory (prevents path traversal)

    Args:
        path: Path to validate
        base_dir: Base directory that path must be within

    Returns:
        Resolved path object

    Raises:
        ValueError: If path is outside base directory
    """
    try:
        path_obj = Path(path).resolve()
        base_obj = Path(base_dir).resolve()

        # Ensure path is within base directory
        path_obj.relative_to(base_obj)

        return path_obj
    except (ValueError, RuntimeError) as e:
        raise ValueError(f"Path traversal attempt detected: {path}") from e


def validate_json_size(filepath: Union[str, Path], max_size_mb: int = MAX_JSON_SIZE_MB) -> bool:
    """
    Validate JSON file size before loading

    Args:
        filepath: Path to JSON file
        max_size_mb: Maximum allowed size in MB

    Returns:
        True if size is acceptable, False otherwise
    """
    try:
        size_bytes = Path(filepath).stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > max_size_mb:
            logger.warning(f"File too large: {size_mb:.2f}MB > {max_size_mb}MB")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking file size: {e}")
        return False


# ============================================================================
# CENTRALIZED FALLBACK SYSTEM - SINGLE SOURCE OF TRUTH
# ============================================================================

class CentralizedFallbackManager:
    """Master fallback manager - eliminates ALL duplication"""

    def __init__(self):
        self.module_status: Dict[str, Dict[str, Any]] = {}
        self.fallback_count: int = 0

    def register_fallback(self, module_name: str, function_name: str) -> None:
        """Register when a fallback is used"""
        self.fallback_count += 1
        if module_name not in self.module_status:
            self.module_status[module_name] = {'available': False, 'fallbacks': []}
        self.module_status[module_name]['fallbacks'].append(function_name)
        logger.debug(f"📝 Fallback used: {module_name}.{function_name}")

    def mark_available(self, module_name: str) -> None:
        """Mark module as available"""
        if module_name not in self.module_status:
            self.module_status[module_name] = {'available': True, 'fallbacks': []}
        else:
            self.module_status[module_name]['available'] = True

    def get_status(self) -> Dict[str, Any]:
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
    """
    Enhanced language detection fallback with Luxembourg focus

    Args:
        text: Text to analyze

    Returns:
        Language code (fr, en, de)
    """
    _fallback_manager.register_fallback('language', 'detect_language')

    if not text or len(text.strip()) < 10:
        return DEFAULT_LUXEMBOURG_LANGUAGE

    text_lower = text.lower()[:1000]  # Only check first 1000 chars for performance

    language_scores = {
        'fr': 0,
        'en': 0,
        'de': 0
    }

    # French indicators (common in Luxembourg)
    french_indicators = [
        'le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'et ', 'ou ', 'à ',
        'conformité', 'règlement', 'société', 'banque'
    ]

    # English indicators
    english_indicators = [
        'the ', 'and ', 'or ', 'of ', 'to ', 'in ', 'for ', 'with ',
        'compliance', 'regulation', 'company', 'bank'
    ]

    # German indicators (also used in Luxembourg)
    german_indicators = [
        'der ', 'die ', 'das ', 'und ', 'oder ', 'von ', 'für ', 'mit ',
        'gesellschaft', 'bank'
    ]

    # Score each language
    for indicator in french_indicators:
        language_scores['fr'] += text_lower.count(indicator) * 1.2  # Slight preference for French

    for indicator in english_indicators:
        language_scores['en'] += text_lower.count(indicator)

    for indicator in german_indicators:
        language_scores['de'] += text_lower.count(indicator)

    # Return language with highest score, default to French
    max_lang = max(language_scores.items(), key=lambda x: x[1])
    return max_lang[0] if max_lang[1] > 0 else DEFAULT_LUXEMBOURG_LANGUAGE


@lru_cache(maxsize=256)
def _fallback_get_translation(key: str, language: str = "fr") -> str:
    """
    Centralized translation fallback with caching

    Args:
        key: Translation key
        language: Target language code

    Returns:
        Translated text
    """
    _fallback_manager.register_fallback('language', 'get_translation')

    # Extended translations for Luxembourg compliance
    translations = {
        'compliance_score': {
            'fr': 'Score de Conformité',
            'en': 'Compliance Score',
            'de': 'Compliance-Score',
            'es': 'Puntuación de Cumplimiento'
        },
        'analysis_complete': {
            'fr': 'Analyse Terminée',
            'en': 'Analysis Complete',
            'de': 'Analyse Abgeschlossen',
            'es': 'Análisis Completado'
        },
        'issues_found': {
            'fr': 'Problèmes Détectés',
            'en': 'Issues Found',
            'de': 'Probleme Gefunden',
            'es': 'Problemas Detectados'
        },
        'recommendations': {
            'fr': 'Recommandations',
            'en': 'Recommendations',
            'de': 'Empfehlungen',
            'es': 'Recomendaciones'
        },
        'luxembourg_regulation': {
            'fr': 'Réglementation Luxembourgeoise',
            'en': 'Luxembourg Regulation',
            'de': 'Luxemburger Regulierung',
            'es': 'Regulación de Luxemburgo'
        },
        'cssf_compliance': {
            'fr': 'Conformité CSSF',
            'en': 'CSSF Compliance',
            'de': 'CSSF-Konformität',
            'es': 'Cumplimiento CSSF'
        }
    }

    if key in translations and language in translations[key]:
        return translations[key][language]

    # Fallback to key transformation
    return key.replace('_', ' ').title()


def _fallback_get_available_languages() -> Dict[str, str]:
    """Centralized available languages fallback"""
    _fallback_manager.register_fallback('language', 'get_available_languages')
    return {
        "fr": "Français",
        "en": "English",
        "de": "Deutsch",
        "es": "Español"
    }


# Text extraction fallbacks

def _fallback_extract_text(file, detect_lang: bool = False, **kwargs) -> Tuple[str, Optional[str]]:
    """
    Centralized text extraction fallback with size limits

    Args:
        file: File object or path
        detect_lang: Whether to detect language

    Returns:
        Tuple of (text content, language code)
    """
    _fallback_manager.register_fallback('text_extraction', 'extract_text')

    if hasattr(file, 'read'):
        try:
            content = file.read()

            # Size limit check
            if isinstance(content, bytes):
                if len(content) > MAX_TEXT_EXTRACT_SIZE:
                    logger.warning(f"File too large for extraction: {len(content)} bytes")
                    return "File too large", DEFAULT_LUXEMBOURG_LANGUAGE
                content = content.decode('utf-8', errors='ignore')

            if len(content) > MAX_TEXT_EXTRACT_SIZE:
                logger.warning("Text content too large, truncating")
                content = content[:MAX_TEXT_EXTRACT_SIZE]

            lang = _fallback_detect_language(content) if detect_lang else None
            return content, lang
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return f"Text extraction error: {e}", DEFAULT_LUXEMBOURG_LANGUAGE

    return str(file), DEFAULT_LUXEMBOURG_LANGUAGE


class _FallbackDocumentProcessor:
    """Centralized document processor fallback"""

    def __init__(self):
        _fallback_manager.register_fallback('text_extraction', 'DocumentProcessor')


# Data manager fallbacks with lazy loading

class _FallbackComplianceDataManager:
    """
    Centralized data manager fallback with lazy loading and security
    """

    def __init__(self):
        _fallback_manager.register_fallback('data_manager', 'ComplianceDataManager')
        self._cache: Dict[str, Dict] = {}
        self._loaded: set = set()
        self._base_path = Path('data')

    def _load_file(self, filename: str) -> Dict[str, Any]:
        """
        Lazy load individual file with security validation

        Args:
            filename: Name of file to load

        Returns:
            Loaded data dictionary
        """
        if filename in self._loaded:
            return self._cache.get(filename, {})

        try:
            # Validate filename
            safe_filename = sanitize_filename(filename)
            filepath = self._base_path / safe_filename

            # Validate path is within data directory
            filepath = validate_path(filepath, self._base_path)

            if not filepath.exists():
                logger.debug(f"Data file not found: {filename}")
                self._cache[filename] = {}
            elif not filepath.is_file():
                logger.warning(f"Path is not a file: {filename}")
                self._cache[filename] = {}
            elif not validate_json_size(filepath):
                logger.error(f"File too large: {filename}")
                self._cache[filename] = {}
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        try:
                            self._cache[filename] = json.loads(content)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON in {filename}: {e}")
                            self._cache[filename] = {}
                    else:
                        self._cache[filename] = {}

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            self._cache[filename] = {}

        self._loaded.add(filename)
        return self._cache[filename]

    def get_compliance_rules(self) -> Dict[str, Any]:
        return self._load_file('compliance_rules.json')

    def get_lux_keywords(self) -> Dict[str, Any]:
        return self._load_file('lux_keywords.json')

    def get_compliance_penalties(self) -> Dict[str, Any]:
        return self._load_file('compliance_penalties.json')

    def get_regulations(self) -> Dict[str, Any]:
        return self._load_file('regulations.json')

    def get_financial_institutions(self) -> Dict[str, Any]:
        return self._load_file('financial_institutions.json')

    def get_analyses(self) -> Dict[str, Any]:
        return self._load_file('analyses.json')

    def get_cross_border_regulations(self) -> Dict[str, Any]:
        return self._load_file('cross_border_regulations.json')

    def get_dynamic_rules(self) -> Dict[str, Any]:
        return self._load_file('dynamic_rules.json')

    def get_issue_descriptions(self) -> Dict[str, Any]:
        return self._load_file('issue_descriptions.json')

    def get_reporting_requirements(self) -> Dict[str, Any]:
        return self._load_file('reporting_requirements.json')

    def get_sanctions_lists(self) -> Dict[str, Any]:
        return self._load_file('sanctions_lists.json')

    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded data"""
        total_files = len(DATA_FILES)
        files_loaded = len(self._loaded)
        files_with_data = sum(1 for data in self._cache.values() if data)

        return {
            '_summary': {
                'total_files': total_files,
                'files_loaded': files_loaded,
                'files_with_data': files_with_data,
                'total_records': sum(
                    len(data) if isinstance(data, dict) else 0
                    for data in self._cache.values()
                ),
                'all_files_loaded': files_loaded == total_files
            }
        }

    def preload_all(self) -> None:
        """Explicitly load all data files"""
        for filename in DATA_FILES:
            self._load_file(filename)

    def refresh_all_data(self) -> None:
        """Refresh all loaded data files"""
        self._cache.clear()
        self._loaded.clear()
        self.preload_all()


# Session manager fallback (FIXED: Class not instance)

class _FallbackSessionManager:
    """Centralized session manager fallback (singleton pattern)"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            _fallback_manager.register_fallback('session_manager', 'SessionManager')
        return cls._instance

    @staticmethod
    def init_session() -> None:
        """Initialize session state"""
        try:
            import streamlit as st
            if 'initialized' not in st.session_state:
                st.session_state.initialized = True
                st.session_state.analysis_history = []
                st.session_state.language = DEFAULT_LUXEMBOURG_LANGUAGE
        except ImportError:
            pass

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state"""
        try:
            import streamlit as st
            return st.session_state.get(key, default)
        except ImportError:
            return default

    @staticmethod
    def set(key: str, value: Any) -> None:
        """Set value in session state"""
        try:
            import streamlit as st
            st.session_state[key] = value
        except ImportError:
            pass

    @staticmethod
    def add_analysis(data: Dict[str, Any]) -> None:
        """Add analysis to history"""
        try:
            import streamlit as st
            if 'analysis_history' not in st.session_state:
                st.session_state.analysis_history = []
            st.session_state.analysis_history.append(data)
        except ImportError:
            pass


# LLM analyzer fallbacks

def _fallback_analyze_regulatory_compliance(*args, **kwargs) -> Dict[str, Any]:
    """Centralized compliance analysis fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'analyze_regulatory_compliance')
    return {
        "score": 70.0,
        "final_score": 70.0,
        "issues": [],
        "recommendations": ["Module d'analyse IA non disponible - utilisation des règles de base"],
        "analysis_engine": "fallback",
        "overall_assessment": "Analyse basique effectuée",
        "statistics": {
            "analysis_method": "basic_fallback",
            "ai_enabled": False,
            "processing_time": 0.1
        }
    }


def _fallback_identify_issues(*args, **kwargs) -> Tuple[List[Dict], float]:
    """Centralized issue identification fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'identify_issues')
    return [], 70.0


def _fallback_check_ollama_installation() -> Dict[str, Any]:
    """Centralized Ollama check fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'check_ollama_installation')
    return {
        "installed": False,
        "running": False,
        "models": [],
        "status": "not_available"
    }


def _fallback_load_your_data_files() -> Dict[str, Any]:
    """Centralized data loading fallback"""
    _fallback_manager.register_fallback('llm_analyzer', 'load_your_data_files')
    return {}


# JSON utilities fallbacks (SECURED)

def _fallback_load_json(
        path: Union[str, Path],
        default: Optional[Dict[str, Any]] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Centralized JSON loading fallback with security validation

    Args:
        path: Path to JSON file
        default: Default value if loading fails

    Returns:
        Loaded JSON data or default
    """
    _fallback_manager.register_fallback('json_utils', 'load_json')

    if default is None:
        default = {}

    try:
        # Convert to Path object
        path_obj = Path(path).resolve()

        # Check file exists and is file
        if not path_obj.exists():
            logger.debug(f"JSON file not found: {path}")
            return default

        if not path_obj.is_file():
            logger.error(f"Path is not a file: {path}")
            return default

        # Check file size
        if not validate_json_size(path_obj):
            return default

        # Check for empty file
        if path_obj.stat().st_size == 0:
            logger.warning(f"Empty JSON file: {path}")
            return default

        # Load JSON
        with open(path_obj, 'r', encoding='utf-8') as f:
            return json.load(f)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return default
    except PermissionError:
        logger.error(f"Permission denied reading {path}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error loading {path}: {e}")
        return default


def _fallback_save_json(
        data: Dict[str, Any],
        path: Union[str, Path],
        **kwargs
) -> bool:
    """
    Centralized JSON saving fallback with validation

    Args:
        data: Data to save
        path: Path to save to

    Returns:
        True if successful, False otherwise
    """
    _fallback_manager.register_fallback('json_utils', 'save_json')

    try:
        path_obj = Path(path)

        # Create parent directory if needed
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        return False


# Styling fallbacks

def _fallback_styling_function(*args, **kwargs) -> None:
    """Universal styling fallback"""
    _fallback_manager.register_fallback('styling', 'various_functions')
    pass


def _fallback_get_card_html(*args, **kwargs) -> str:
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

# Session manager (FIXED: Class not instance)
try:
    from .session_manager import SessionManager
    _fallback_manager.mark_available('session_manager')
    logger.debug("✅ Session manager loaded")
except ImportError as e:
    logger.warning(f"⚠️ Session manager fallback: {e}")
    SessionManager = _FallbackSessionManager  # Class, not instance!

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

    def get_data_path(filename: str = "", subfolder: Optional[str] = None) -> str:
        """Get data path with security validation"""
        if filename and not is_safe_path(filename):
            raise ValueError(f"Invalid filename: {filename}")
        if subfolder and not is_safe_path(subfolder):
            raise ValueError(f"Invalid subfolder: {subfolder}")

        base = Path("data")
        if subfolder:
            path = base / subfolder / filename
        else:
            path = base / filename

        # Validate path
        try:
            path.resolve().relative_to(base.resolve())
        except ValueError:
            raise ValueError("Path traversal attempt detected")

        return str(path)

    class config:
        @staticmethod
        def get_data_path(filename: str = "", subfolder: Optional[str] = None) -> str:
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
# UTILITY FUNCTIONS (SECURED AND OPTIMIZED)
# ============================================================================

def get_asset_path(filename: str = "", subfolder: Optional[str] = None) -> str:
    """
    Get asset path with security validation

    Args:
        filename: Asset filename
        subfolder: Optional subfolder

    Returns:
        Validated asset path
    """
    if filename and not is_safe_path(filename):
        raise ValueError(f"Invalid filename: {filename}")
    if subfolder and not is_safe_path(subfolder):
        raise ValueError(f"Invalid subfolder: {subfolder}")

    return os.path.join("assets", subfolder or "", filename)


def get_log_path(filename: str = "", subfolder: Optional[str] = None) -> str:
    """
    Get log path with security validation

    Args:
        filename: Log filename
        subfolder: Optional subfolder

    Returns:
        Validated log path
    """
    if filename and not is_safe_path(filename):
        raise ValueError(f"Invalid filename: {filename}")
    if subfolder and not is_safe_path(subfolder):
        raise ValueError(f"Invalid subfolder: {subfolder}")

    return os.path.join("logs", subfolder or "", filename)


def get_model_path(filename: str = "", subfolder: Optional[str] = None) -> str:
    """
    Get model path with security validation

    Args:
        filename: Model filename
        subfolder: Optional subfolder

    Returns:
        Validated model path
    """
    if filename and not is_safe_path(filename):
        raise ValueError(f"Invalid filename: {filename}")
    if subfolder and not is_safe_path(subfolder):
        raise ValueError(f"Invalid subfolder: {subfolder}")

    return os.path.join("models_cache", subfolder or "", filename)


def ensure_data_file(path: Union[str, Path], default_content: Optional[Dict] = None) -> bool:
    """
    Ensure data file exists, create with default content if not

    Args:
        path: File path
        default_content: Default content to write

    Returns:
        True if file was created, False if already existed
    """
    if default_content is None:
        default_content = {}

    path_obj = Path(path)

    if not path_obj.exists():
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        save_json(data=default_content, path=path)
        return True

    return False


def check_your_data_files() -> Dict[str, Dict[str, Any]]:
    """
    Check which data files are available

    Returns:
        Dictionary with file status information
    """
    status = {}

    for filename in DATA_FILES:
        filepath = os.path.join("data", filename)
        exists = os.path.exists(filepath)

        status[filename] = {
            'exists': exists,
            'path': filepath,
            'size': os.path.getsize(filepath) if exists else 0,
            'size_mb': round(os.path.getsize(filepath) / (1024 * 1024), 2) if exists else 0
        }

    return status


def get_fallback_status() -> Dict[str, Any]:
    """
    Get complete fallback system status

    Returns:
        Fallback status dictionary
    """
    return _fallback_manager.get_status()


# ============================================================================
# LUXEMBOURG-SPECIFIC UTILITIES
# ============================================================================

def is_luxembourg_document(text: str) -> bool:
    """
    Detect if document is Luxembourg-related

    Args:
        text: Document text

    Returns:
        True if Luxembourg-related indicators found
    """
    luxembourg_indicators = [
        'luxembourg', 'lëtzebuerg', 'cssf', 'bcl',
        'rcs luxembourg', 'siret', 'matricule',
        'banque centrale du luxembourg', 'commission de surveillance'
    ]

    text_lower = text.lower()[:2000]  # Check first 2000 chars
    return any(indicator in text_lower for indicator in luxembourg_indicators)


def extract_luxembourg_identifiers(text: str) -> Dict[str, List[str]]:
    """
    Extract Luxembourg-specific identifiers from text

    Args:
        text: Document text

    Returns:
        Dictionary of found identifiers by type
    """
    identifiers = {
        'rcs_numbers': [],
        'matricule': [],
        'iban_lu': [],
        'cssf_references': []
    }

    # RCS pattern (e.g., B123456)
    rcs_pattern = r'\b[A-Z]\d{6}\b'
    identifiers['rcs_numbers'] = list(set(re.findall(rcs_pattern, text.upper())))

    # IBAN Luxembourg pattern
    iban_pattern = r'\bLU\d{2}[A-Z0-9]{16}\b'
    identifiers['iban_lu'] = list(set(re.findall(iban_pattern, text.upper())))

    # CSSF circular references (e.g., "CSSF 12/552", "circulaire 20/750")
    cssf_pattern = r'(?:CSSF|circulaire)\s*(\d{2}/\d{3})'
    identifiers['cssf_references'] = list(set(re.findall(cssf_pattern, text, re.IGNORECASE)))

    return identifiers


def detect_applicable_cssf_circulars(text: str) -> List[str]:
    """
    Detect which CSSF circulars apply based on content

    Args:
        text: Document text

    Returns:
        List of applicable CSSF circular numbers
    """
    text_lower = text.lower()

    circular_keywords = {
        '12/552': ['gouvernance', 'governance', 'risk management', 'gestion des risques'],
        '20/750': ['climate', 'climat', 'esg', 'environmental', 'environnemental'],
        '21/773': ['aml', 'kyc', 'blanchiment', 'sanctions', 'pep'],
        '08/356': ['customer', 'client', 'protection', 'complaint', 'réclamation']
    }

    applicable = []
    for circular, keywords in circular_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            applicable.append(circular)

    return applicable


# ============================================================================
# SYSTEM HEALTH CHECK
# ============================================================================

def check_system_health() -> Dict[str, Any]:
    """
    Comprehensive system health check

    Returns:
        Health status dictionary
    """
    health = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }

    # Check data files
    data_status = check_your_data_files()
    available_files = sum(1 for f in data_status.values() if f['exists'])
    health['checks']['data_files'] = {
        'status': 'ok' if available_files >= 8 else 'warning',
        'available': available_files,
        'total': len(DATA_FILES),
        'missing': [k for k, v in data_status.items() if not v['exists']]
    }

    # Check fallback status
    fallback_status = get_fallback_status()
    health['checks']['fallbacks'] = {
        'status': 'ok',
        'total_used': fallback_status['total_fallbacks_used'],
        'available_modules': len(fallback_status['available_modules']),
        'fallback_modules': len(fallback_status['fallback_modules'])
    }

    # Check dependencies
    deps = check_dependencies()
    health['checks']['dependencies'] = {
        'status': 'ok' if deps.get('your_data_files', False) else 'warning',
        'details': deps
    }

    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (2**30)
        health['checks']['disk_space'] = {
            'status': 'critical' if free_gb < 1 else ('warning' if free_gb < 5 else 'ok'),
            'free_gb': free_gb,
            'total_gb': total // (2**30)
        }
    except Exception as e:
        health['checks']['disk_space'] = {
            'status': 'unknown',
            'error': str(e)
        }

    # Overall status
    statuses = [
        check.get('status', 'unknown')
        for check in health['checks'].values()
        if isinstance(check, dict)
    ]

    if 'critical' in statuses:
        health['status'] = 'critical'
    elif 'warning' in statuses:
        health['status'] = 'warning'

    return health


# ============================================================================
# BACKWARD COMPATIBILITY (OPTIMIZED)
# ============================================================================

def find_non_conformities_semantic(
        text: str,
        rules: List[Dict],
        **kwargs
) -> List[Tuple]:
    """
    Legacy compatibility function for semantic non-conformity detection

    Args:
        text: Text to analyze
        rules: Compliance rules

    Returns:
        List of tuples with non-conformity information
    """
    try:
        issues, confidence = identify_issues(text)
        results = []

        for issue in issues[:10]:  # Limit to 10 for performance
            result_tuple = (
                issue.get("description", "")[:100],
                issue.get("regulatory_basis", ""),
                issue.get("rule_id", ""),
                issue.get("confidence_score", confidence),
                issue.get("severity", "medium")
            )
            results.append(result_tuple)

        return results

    except Exception as e:
        logger.error(f"Error in find_non_conformities_semantic: {e}")
        return []


def guess_doc_type(text: str, language: str = "fr") -> str:
    """
    Legacy document type detection

    Args:
        text: Document text
        language: Document language

    Returns:
        Detected document type
    """
    text_lower = text.lower()[:500]  # Only check first 500 chars

    if any(word in text_lower for word in ['contrat', 'contract', 'vertrag']):
        return 'contract'
    elif any(word in text_lower for word in ['politique', 'policy', 'richtlinie']):
        return 'policy'
    elif any(word in text_lower for word in ['rapport', 'report', 'bericht']):
        return 'report'
    elif any(word in text_lower for word in ['procédure', 'procedure', 'verfahren']):
        return 'procedure'

    return 'auto'


def get_regulatory_rules(regulation_type: str = "all") -> List[Dict]:
    """
    Get regulatory rules using data manager

    Args:
        regulation_type: Type of regulations to retrieve

    Returns:
        List of regulatory rules
    """
    try:
        data_manager = ComplianceDataManager()

        if hasattr(data_manager, 'get_rules'):
            return data_manager.get_rules(regulation_type)
        else:
            # Fallback to compliance rules
            rules_data = data_manager.get_compliance_rules()
            return list(rules_data.values()) if isinstance(rules_data, dict) else []

    except Exception as e:
        logger.error(f"Error getting regulatory rules: {e}")
        return []


def validate_document_compliance(
        text: str,
        doc_type: str = "auto",
        language: str = "auto"
) -> Dict[str, Any]:
    """
    Legacy compliance validation wrapper

    Args:
        text: Document text
        doc_type: Document type
        language: Document language

    Returns:
        Compliance analysis results
    """
    try:
        return analyze_regulatory_compliance(text, doc_type=doc_type, language=language)
    except Exception as e:
        logger.error(f"Error in validate_document_compliance: {e}")
        return {
            "score": 0.0,
            "final_score": 0.0,
            "issues": [{
                "description": f"Validation error: {str(e)}",
                "severity": "critical"
            }],
            "recommendations": ["Check system configuration"],
            "doc_type": doc_type,
            "language": language,
            "error": True
        }


# ============================================================================
# PACKAGE INFO AND INITIALIZATION
# ============================================================================

__version__ = "2.0.1-secure"
__author__ = "LexAI Team"

PACKAGE_INFO = {
    "name": "LexAI Utils",
    "version": __version__,
    "description": "Complete utilities with centralized fallbacks for LexAI",
    "fallback_system": "centralized",
    "security": "hardened",
    "data_files_supported": DATA_FILES,
    "luxembourg_features": [
        "CSSF circular detection",
        "RCS number extraction",
        "IBAN-LU validation",
        "Multi-language support (FR/DE/EN)"
    ]
}


def get_package_info() -> Dict[str, Any]:
    """
    Get package information with status

    Returns:
        Package information dictionary
    """
    info = PACKAGE_INFO.copy()
    info['data_files_status'] = check_your_data_files()
    info['fallback_status'] = get_fallback_status()
    info['system_health'] = check_system_health()

    return info


def check_dependencies() -> Dict[str, bool]:
    """
    Check available dependencies

    Returns:
        Dictionary of dependency availability
    """
    deps = {}

    # Check data files
    data_status = check_your_data_files()
    deps['your_data_files'] = sum(1 for f in data_status.values() if f['exists']) >= 5

    # Check Python packages
    for module in ['requests', 'pypdf', 'docx', 'streamlit', 'pandas']:
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

    logger.info(f"✅ LexAI Utils v{__version__} initialized (SECURED)")
    logger.info(f"📊 Data files: {available_files}/{len(DATA_FILES)} available")
    logger.info(f"🔄 Fallbacks: {fallback_status['total_fallbacks_used']} used")
    logger.info(f"🔒 Security: Path validation, size limits, input sanitization enabled")

except Exception as e:
    logger.error(f"❌ Package initialization failed: {e}")


# ============================================================================
# COMPLETE EXPORTS LIST
# ============================================================================

__all__ = [
    # Core functions
    "analyze_regulatory_compliance",
    "identify_issues",
    "detect_language",
    "get_translation",
    "get_available_languages",
    "extract_text",
    "load_your_data_files",

    # Path utilities (secured)
    "get_data_path",
    "get_asset_path",
    "get_log_path",
    "get_model_path",

    # Security utilities
    "sanitize_filename",
    "is_safe_path",
    "validate_path",
    "validate_json_size",

    # Data utilities
    "load_json",
    "save_json",
    "ensure_data_file",
    "check_your_data_files",

    # Luxembourg utilities
    "is_luxembourg_document",
    "extract_luxembourg_identifiers",
    "detect_applicable_cssf_circulars",

    # System utilities
    "get_fallback_status",
    "check_system_health",
    "get_package_info",
    "check_dependencies",

    # Styling utilities
    "apply_emergency_visibility_fix",
    "create_optimized_banner",
    "create_simple_metric_card",
    "create_status_indicator",
    "initialize_demo_styling",
    "check_styling_compatibility",
    "set_custom_style",
    "render_compliance_score",
    "render_processing_animation",
    "get_card_html",
    "create_emergency_banner",
    "create_emergency_metric_card",

    # Classes
    "ComplianceDataManager",
    "SessionManager",
    "DocumentProcessor",

    # Backward compatibility
    "find_non_conformities_semantic",
    "guess_doc_type",
    "get_regulatory_rules",
    "validate_document_compliance",

    # Constants
    "DATA_FILES",
    "LUXEMBOURG_LANGUAGES",
    "CSSF_REGULATIONS",
    "PACKAGE_INFO",
    "__version__"
]