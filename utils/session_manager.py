# utils/session_manager.py - CORRECTED VERSION WITH PROPER @classmethod
import streamlit as st
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Thread-safe Streamlit session state manager - FIXED"""
    
    DEFAULT_VALUES = {
        'current_analysis': None,
        'analysis_history': [],
        'language': 'fr',
        'authenticated': False,
        'user': {},
        'current_page': 'home',
        'show_auth': False,
        'session_id': None,
    }
    
    @classmethod
    def init_session(cls) -> None:
        """Initialize session state WITHOUT threading locks"""
        # Streamlit handles concurrency internally - no need for locks
        for key, value in cls.DEFAULT_VALUES.items():
            if key not in st.session_state:
                st.session_state[key] = value
        st.session_state['last_activity'] = datetime.now().isoformat()
    
    @classmethod 
    def get(cls, key: str, default: Any = None) -> Any:
        """Safely get value from session state"""
        cls.init_session()
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Safely set value in session state"""
        cls.init_session()
        st.session_state[key] = value
        st.session_state['last_activity'] = datetime.now().isoformat()
    
    @classmethod
    def clear(cls, keep_keys: Optional[List[str]] = None) -> None:
        """Clear session state, optionally keeping specific keys"""
        if keep_keys is None:
            keep_keys = ['language']
        
        keys_to_remove = [key for key in st.session_state.keys() if key not in keep_keys]
        for key in keys_to_remove:
            del st.session_state[key]
        
        # Reinitialize with defaults
        cls.init_session()
    
    @classmethod
    def add_analysis(cls, analysis_data: Dict[str, Any]) -> None:
        """Add analysis to history"""
        cls.init_session()
        history = cls.get('analysis_history', [])
        
        # Add timestamp if not present
        if 'timestamp' not in analysis_data:
            analysis_data['timestamp'] = datetime.now().isoformat()
        
        history.insert(0, analysis_data)  # Add to beginning
        
        # Keep only last 50 analyses to prevent memory issues
        if len(history) > 50:
            history = history[:50]
        
        cls.set('analysis_history', history)
        logger.info(f"Analysis added to history. Total: {len(history)}")
    
    @classmethod
    def get_analysis_count(cls) -> int:
        """Get total number of analyses in history"""
        history = cls.get('analysis_history', [])
        return len(history)
    
    @classmethod
    def clear_analysis_history(cls) -> None:
        """Clear all analysis history"""
        cls.set('analysis_history', [])
        logger.info("Analysis history cleared")
    
    @classmethod
    def is_authenticated(cls) -> bool:
        """Check if user is authenticated"""
        return cls.get('authenticated', False)
    
    @classmethod
    def get_user(cls) -> Dict[str, Any]:
        """Get current user information"""
        return cls.get('user', {})
    
    @classmethod
    def logout(cls) -> None:
        """Logout user and clear sensitive data"""
        sensitive_keys = ['authenticated', 'user', 'session_id', 'current_analysis']
        for key in sensitive_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Reset to defaults
        cls.init_session()
        logger.info("User logged out")
    
    @classmethod
    def update_language(cls, new_language: str) -> None:
        """Update language and force UI refresh"""
        cls.init_session()
        old_language = cls.get('language', 'fr')
        if old_language != new_language:
            cls.set('language', new_language)
            cls.set('language_changed', True)  # Flag pour forcer le refresh
            logger.info(f"Language changed from {old_language} to {new_language}")

# ============================================================================
# INTERNATIONALIZATION HELPERS - FIXED
# ============================================================================

# Import with fallback
try:
    from .language import get_translation
    LANGUAGE_AVAILABLE = True
except ImportError:
    LANGUAGE_AVAILABLE = False
    def get_translation(key, language='fr'):
        return key.replace('_', ' ').title()

def render_language_selector():
    """Render improved language selector with immediate refresh"""
    current_lang = SessionManager.get('language', 'fr')
    
    # Available languages
    languages = {
        'fr': '🇫🇷 Français',
        'en': '🇬🇧 English', 
        'de': '🇩🇪 Deutsch',
        'es': '🇪🇸 Español'
    }
    
    # Create selectbox with current language
    selected_lang = st.selectbox(
        get_translation("select_language", current_lang),
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(current_lang),
        key="language_selector"
    )
    
    # Detect language change and update immediately
    if selected_lang != current_lang:
        SessionManager.update_language(selected_lang)
        st.success(f"✅ {get_translation('language_changed', selected_lang)}: {languages[selected_lang]}")
        st.rerun()  # Force immediate refresh

def get_localized_text(key: str, **kwargs) -> str:
    """
    Get localized text with fallback and formatting support
    
    Args:
        key: Translation key
        **kwargs: Format parameters
    
    Returns:
        Localized and formatted text
    """
    current_lang = SessionManager.get('language', 'fr')
    
    # Get translation
    if LANGUAGE_AVAILABLE:
        text = get_translation(key, current_lang)
    else:
        # Fallback translations
        fallback_translations = {
            'language_changed': {
                'fr': 'Langue modifiée',
                'en': 'Language changed',
                'de': 'Sprache geändert', 
                'es': 'Idioma cambiado'
            },
            'select_language': {
                'fr': 'Sélectionner la langue',
                'en': 'Select language',
                'de': 'Sprache auswählen',
                'es': 'Seleccionar idioma'
            },
            'analyze_document': {
                'fr': 'Analyser le document',
                'en': 'Analyze document',
                'de': 'Dokument analysieren',
                'es': 'Analizar documento'
            },
            'upload_file': {
                'fr': 'Télécharger un fichier',
                'en': 'Upload file',
                'de': 'Datei hochladen',
                'es': 'Subir archivo'
            },
            'analysis_results': {
                'fr': 'Résultats d\'analyse',
                'en': 'Analysis results',
                'de': 'Analyseergebnisse',
                'es': 'Resultados de análisis'
            },
            'compliance_score': {
                'fr': 'Score de conformité',
                'en': 'Compliance score',
                'de': 'Compliance-Score',
                'es': 'Puntuación de cumplimiento'
            }
        }
        
        if key in fallback_translations and current_lang in fallback_translations[key]:
            text = fallback_translations[key][current_lang]
        else:
            text = key.replace('_', ' ').title()
    
    # Apply formatting if kwargs provided
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass  # Keep original text if formatting fails
    
    return text

def render_sidebar_with_language():
    """Render sidebar with integrated language selector"""
    current_lang = SessionManager.get('language', 'fr')
    
    with st.sidebar:
        # Logo et titre
        try:
            st.image("assets/lexai_banner.png", width=200)
        except:
            st.markdown("## ⚖️ LexAI")
        
        st.markdown(f"### {get_localized_text('app_title')}")
        
        # Sélecteur de langue en haut de la sidebar
        st.markdown("---")
        st.markdown(f"**🌐 {get_localized_text('language')}**")
        render_language_selector()
        
        # Navigation principale
        st.markdown("---")
        st.markdown(f"**📋 {get_localized_text('navigation')}**")
        
        # Boutons de navigation traduits
        if st.button(f"🏠 {get_localized_text('home')}", use_container_width=True):
            SessionManager.set('current_page', 'home')
            st.rerun()
            
        if st.button(f"📊 {get_localized_text('dashboard')}", use_container_width=True):
            SessionManager.set('current_page', 'dashboard')
            st.rerun()
            
        if st.button(f"📚 {get_localized_text('history')}", use_container_width=True):
            SessionManager.set('current_page', 'history')
            st.rerun()
            
        if st.button(f"⚙️ {get_localized_text('administration')}", use_container_width=True):
            SessionManager.set('current_page', 'admin')
            st.rerun()

def check_language_change():
    """Check if language has changed and handle UI updates"""
    if SessionManager.get('language_changed', False):
        SessionManager.set('language_changed', False)
        # Force UI refresh for all components
        st.rerun()

def get_language_flag(lang_code: str) -> str:
    """Get flag emoji for language code"""
    flags = {
        'fr': '🇫🇷',
        'en': '🇬🇧', 
        'de': '🇩🇪',
        'es': '🇪🇸'
    }
    return flags.get(lang_code, '🌐')

def format_localized_date(date_str: str) -> str:
    """Format date according to current language"""
    current_lang = SessionManager.get('language', 'fr')
    
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        
        if current_lang == 'fr':
            return dt.strftime('%d/%m/%Y %H:%M')
        elif current_lang == 'en':
            return dt.strftime('%m/%d/%Y %I:%M %p')
        elif current_lang == 'de':
            return dt.strftime('%d.%m.%Y %H:%M')
        elif current_lang == 'es':
            return dt.strftime('%d/%m/%Y %H:%M')
        else:
            return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return date_str