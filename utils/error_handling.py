# utils/error_handling.py - Robust error handling for Streamlit

import streamlit as st
import logging
import traceback
import functools
from typing import Callable, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

def with_error_recovery(
    fallback_message: str = "Une erreur s'est produite",
    show_details: bool = False,
    clear_on_error: bool = False,
    fallback_return: Any = None
):
    """
    Decorator for robust error handling in Streamlit functions
    
    Args:
        fallback_message: Message to show users on error
        show_details: Whether to show technical details to users
        clear_on_error: Whether to clear session state on error
        fallback_return: Value to return on error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                logger.error(f"Error {error_id} in {func.__name__}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Show user-friendly error
                st.error(f"🚨 {fallback_message}")
                
                # Show technical details if requested
                if show_details:
                    with st.expander("Détails techniques", expanded=False):
                        st.code(f"Erreur ID: {error_id}")
                        st.code(f"Fonction: {func.__name__}")
                        st.code(f"Message: {str(e)}")
                        if st.checkbox("Afficher la trace complète", key=f"trace_{error_id}"):
                            st.code(traceback.format_exc())
                
                # Recovery options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Réessayer", key=f"retry_{error_id}"):
                        st.rerun()
                
                with col2:
                    if st.button("🏠 Retour accueil", key=f"home_{error_id}"):
                        if 'SessionManager' in globals():
                            SessionManager.set('current_page', 'home')
                        else:
                            st.session_state['current_page'] = 'home'
                        st.rerun()
                
                with col3:
                    if clear_on_error and st.button("🧹 Réinitialiser", key=f"reset_{error_id}"):
                        if 'SessionManager' in globals():
                            SessionManager.clear(['language'])
                        else:
                            for key in list(st.session_state.keys()):
                                if key != 'language':
                                    del st.session_state[key]
                        st.rerun()
                
                return fallback_return
                
        return wrapper
    return decorator

def safe_streamlit_operation(operation: Callable, fallback_message: str = "Opération échouée"):
    """
    Safely execute a Streamlit operation with error handling
    
    Args:
        operation: Function to execute
        fallback_message: Message on error
    
    Returns:
        Result of operation or None on error
    """
    try:
        return operation()
    except Exception as e:
        logger.error(f"Safe operation failed: {e}")
        st.warning(f"⚠️ {fallback_message}")
        return None

class ErrorRecoveryManager:
    """Centralized error recovery management"""
    
    @staticmethod
    def handle_auth_error():
        """Handle authentication errors"""
        st.error("🔐 Erreur d'authentification")
        if st.button("🔄 Se reconnecter", key="auth_recovery"):
            if 'SessionManager' in globals():
                SessionManager.clear(['language'])
            st.rerun()
    
    @staticmethod
    def handle_file_error(error_msg: str):
        """Handle file processing errors"""
        st.error(f"📄 Erreur de traitement du fichier: {error_msg}")
        st.info("💡 Vérifiez que le fichier n'est pas corrompu et fait moins de 10MB")
    
    @staticmethod
    def handle_analysis_error():
        """Handle analysis errors"""
        st.error("🧠 Erreur lors de l'analyse")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Réessayer l'analyse", key="retry_analysis"):
                st.rerun()
        
        with col2:
            if st.button("📄 Changer de document", key="change_doc"):
                if 'SessionManager' in globals():
                    SessionManager.set('current_analysis', None)
                st.rerun()
    
    @staticmethod
    def handle_network_error():
        """Handle network/API errors"""
        st.error("🌐 Erreur de connexion")
        st.info("Vérifiez votre connexion internet et réessayez")
        
        if st.button("🔄 Réessayer", key="network_retry"):
            st.rerun()

# Context manager for error recovery
class error_context:
    """Context manager for handling errors in code blocks"""
    
    def __init__(self, fallback_message: str = "Une erreur s'est produite"):
        self.fallback_message = fallback_message
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Context error: {exc_val}")
            st.error(f"🚨 {self.fallback_message}")
            
            if st.button("🔄 Réessayer", key=f"context_retry_{datetime.now().timestamp()}"):
                st.rerun()
            
            return True  # Suppress the exception
        return False