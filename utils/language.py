# utils/language.py - FIXED VERSION WITH PROPER TRANSLATION LOADING AND LANGUAGE DETECTION
import json
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

class TranslationManager:
    """Centralized translation management"""
    
    def __init__(self):
        self._translations = {}
        self._loaded = False
        self._load_translations()
    
    def _load_translations(self):
        """Load translations from JSON file"""
        try:
            # Try multiple possible paths
            possible_paths = [
                Path("data/translations.json"),
                Path("./data/translations.json"),
                Path("../data/translations.json"),
                Path(__file__).parent.parent / "data" / "translations.json"
            ]
            
            translations_file = None
            for path in possible_paths:
                if path.exists():
                    translations_file = path
                    break
            
            if not translations_file:
                logger.error("translations.json file not found in any expected location")
                self._create_fallback_translations()
                return
            
            with open(translations_file, 'r', encoding='utf-8') as f:
                self._translations = json.load(f)
                self._loaded = True
                logger.info(f"Translations loaded successfully from {translations_file}")
                
        except Exception as e:
            logger.error(f"Error loading translations: {e}")
            self._create_fallback_translations()
    
    def _create_fallback_translations(self):
        """Create fallback translations for essential keys"""
        self._translations = {
            "profile": {
                "fr": "Profil",
                "en": "Profile", 
                "de": "Profil",
                "es": "Perfil"
            },
            "logout": {
                "fr": "Déconnexion",
                "en": "Logout",
                "de": "Abmelden", 
                "es": "Cerrar Sesión"
            },
            "home": {
                "fr": "Accueil",
                "en": "Home",
                "de": "Startseite",
                "es": "Inicio"
            },
            "dashboard": {
                "fr": "Tableau de bord",
                "en": "Dashboard",
                "de": "Dashboard", 
                "es": "Panel"
            },
            "history": {
                "fr": "Historique",
                "en": "History",
                "de": "Verlauf",
                "es": "Historial"
            },
            "administration": {
                "fr": "Administration",
                "en": "Administration",
                "de": "Verwaltung",
                "es": "Administración"
            },
            "welcome": {
                "fr": "Bienvenue",
                "en": "Welcome",
                "de": "Willkommen",
                "es": "Bienvenido"
            },
            "app_title": {
                "fr": "Assistant de conformité réglementaire",
                "en": "Regulatory Compliance Assistant",
                "de": "Regulatorischer Compliance-Assistent",
                "es": "Asistente de Cumplimiento Normativo"
            },
            "authentication": {
                "fr": "Authentification",
                "en": "Authentication",
                "de": "Authentifizierung",
                "es": "Autenticación"
            },
            "login": {
                "fr": "Connexion",
                "en": "Login",
                "de": "Anmelden",
                "es": "Iniciar Sesión"
            },
            "register": {
                "fr": "Inscription",
                "en": "Register",
                "de": "Registrieren",
                "es": "Registrarse"
            },
            "email": {
                "fr": "Email",
                "en": "Email",
                "de": "E-Mail",
                "es": "Correo electrónico"
            },
            "password": {
                "fr": "Mot de passe",
                "en": "Password",
                "de": "Passwort",
                "es": "Contraseña"
            },
            "company": {
                "fr": "Entreprise",
                "en": "Company",
                "de": "Unternehmen",
                "es": "Empresa"
            }
        }
        logger.info("Fallback translations created")

# Global translation manager instance
_translation_manager = TranslationManager()

def get_translation(key: str, language: str = 'fr') -> str:
    """
    Get translation for a key in specified language
    
    Args:
        key: Translation key
        language: Language code (fr, en, de, es)
    
    Returns:
        Translated text or fallback
    """
    try:
        # Check if key exists in translations
        if key in _translation_manager._translations:
            translations_for_key = _translation_manager._translations[key]
            
            # Check if language exists for this key
            if language in translations_for_key:
                return translations_for_key[language]
            
            # Fallback to French if requested language not available
            if 'fr' in translations_for_key:
                return translations_for_key['fr']
            
            # Fallback to English if French not available
            if 'en' in translations_for_key:
                return translations_for_key['en']
            
            # Fallback to first available language
            if translations_for_key:
                return list(translations_for_key.values())[0]
        
        # If key not found, don't log as warning if it's a common case
        if not key.startswith('_') and len(key) > 0:
            logger.debug(f"Translation key not found: {key}")
        
        # Return readable version of key
        return key.replace('_', ' ').title()
        
    except Exception as e:
        logger.error(f"Error getting translation for key '{key}': {e}")
        return key.replace('_', ' ').title()

def get_all_translation_keys() -> List[str]:
    """
    Get all available translation keys
    
    Returns:
        List of all translation keys
    """
    try:
        return list(_translation_manager._translations.keys())
    except Exception as e:
        logger.error(f"Error getting translation keys: {e}")
        return []

def get_available_languages() -> Dict[str, str]:
    """Get available languages"""
    return {
        'fr': 'Français',
        'en': 'English', 
        'de': 'Deutsch',
        'es': 'Español'
    }

def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return 'fr'  # Return default for short texts
    try:
        from langdetect import detect
        detected = detect(text)
        return detected if detected in ['fr', 'en', 'de', 'es'] else 'fr'
    except:
        return _simple_language_detection(text)

def _simple_language_detection(text: str) -> str:
    """Simple language detection based on common words"""
    text_lower = text.lower()
    
    # Language indicators
    french_indicators = ['le', 'la', 'les', 'de', 'des', 'du', 'et', 'est', 'une', 'dans', 'pour', 'avec', 'sur', 'par', 'article', 'conformément']
    english_indicators = ['the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'by', 'article', 'pursuant', 'accordance']
    german_indicators = ['der', 'die', 'das', 'und', 'oder', 'von', 'zu', 'in', 'für', 'mit', 'auf', 'durch', 'artikel', 'gemäß']
    spanish_indicators = ['el', 'la', 'los', 'las', 'de', 'del', 'y', 'o', 'en', 'para', 'con', 'por', 'artículo', 'conforme']
    
    # Count indicators
    scores = {
        'fr': sum(1 for word in french_indicators if word in text_lower),
        'en': sum(1 for word in english_indicators if word in text_lower),
        'de': sum(1 for word in german_indicators if word in text_lower),
        'es': sum(1 for word in spanish_indicators if word in text_lower)
    }
    
    # Return language with highest score, default to French
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return 'fr'

def reload_translations():
    """Reload translations from file"""
    global _translation_manager
    _translation_manager = TranslationManager()

def add_translation(key: str, translations: Dict[str, str]):
    """Add or update a translation"""
    _translation_manager._translations[key] = translations

def load_translations() -> Dict[str, Any]:
    """Load and return all translations (for backward compatibility)"""
    return _translation_manager._translations

# For backward compatibility
def get_text(key: str, language: str = 'fr') -> str:
    """Alias for get_translation"""
    return get_translation(key, language)