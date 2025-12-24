# utils/error_handling.py
# Robust, Streamlit-safe error handling (v2)

import streamlit as st
import logging
import traceback
import functools
import uuid
from typing import Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# -----------------------------
# Session helpers (safe)
# -----------------------------
def _set_session(key: str, value: Any):
    st.session_state[key] = value


def _clear_session(except_keys=None):
    except_keys = except_keys or []
    for key in list(st.session_state.keys()):
        if key not in except_keys:
            del st.session_state[key]


# -----------------------------
# Error state handling
# -----------------------------
def _set_error_state(error_id: str):
    st.session_state["_last_error_id"] = error_id


def _clear_error_state():
    st.session_state.pop("_last_error_id", None)


# -----------------------------
# Decorator
# -----------------------------
def with_error_recovery(
        fallback_message: str = "Une erreur s'est produite",
        show_details: bool = False,
        clear_on_error: bool = False,
        fallback_return: Any = None
):
    """
    Decorator for robust Streamlit error handling
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                _clear_error_state()
                return func(*args, **kwargs)

            except Exception as e:
                error_id = str(uuid.uuid4())[:8]
                _set_error_state(error_id)

                logger.error(
                    f"[{error_id}] Error in {func.__name__}",
                    exc_info=True
                )

                _render_error_ui(
                    error_id=error_id,
                    fallback_message=fallback_message,
                    show_details=show_details,
                    exception=e,
                    clear_on_error=clear_on_error,
                )

                return fallback_return

        return wrapper

    return decorator


# -----------------------------
# UI rendering
# -----------------------------
def _render_error_ui(
        error_id: str,
        fallback_message: str,
        show_details: bool,
        exception: Exception,
        clear_on_error: bool,
):
    st.error(f"🚨 {fallback_message}")

    if show_details:
        with st.expander("Détails techniques"):
            st.code(f"Erreur ID : {error_id}")
            st.code(f"Type : {type(exception).__name__}")
            st.code(f"Message : {str(exception)}")
            st.code(traceback.format_exc())

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Réessayer", key=f"retry_{error_id}"):
            _clear_error_state()
            st.rerun()

    with col2:
        if st.button("🏠 Accueil", key=f"home_{error_id}"):
            _set_session("current_page", "home")
            _clear_error_state()
            st.rerun()

    with col3:
        if clear_on_error and st.button("🧹 Réinitialiser", key=f"reset_{error_id}"):
            _clear_session(except_keys=["language"])
            _clear_error_state()
            st.rerun()


# -----------------------------
# Safe execution helper
# -----------------------------
def safe_streamlit_operation(
        operation: Callable,
        fallback_message: str = "Opération échouée"
):
    try:
        return operation()
    except Exception as e:
        logger.error("Safe operation failed", exc_info=True)
        st.warning(f"⚠️ {fallback_message}")
        return None


# -----------------------------
# Centralized recovery manager
# -----------------------------
class ErrorRecoveryManager:

    @staticmethod
    def auth_error():
        st.error("🔐 Erreur d'authentification")
        if st.button("🔄 Se reconnecter", key="auth_retry"):
            _clear_session(except_keys=["language"])
            st.rerun()

    @staticmethod
    def file_error(message: str):
        st.error(f"📄 Erreur fichier : {message}")
        st.info("💡 Vérifiez le format et la taille (<10MB).")

    @staticmethod
    def analysis_error():
        st.error("🧠 Erreur lors de l'analyse")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Réessayer", key="analysis_retry"):
                st.rerun()

        with col2:
            if st.button("📄 Changer de document", key="analysis_change"):
                _set_session("current_analysis", None)
                st.rerun()

    @staticmethod
    def network_error():
        st.error("🌐 Erreur réseau")
        st.info("Vérifiez votre connexion internet.")
        if st.button("🔄 Réessayer", key="network_retry"):
            st.rerun()


# -----------------------------
# Context manager
# -----------------------------
class error_context:
    """
    Context manager for local error handling
    """

    def __init__(self, fallback_message: str = "Une erreur s'est produite"):
        self.fallback_message = fallback_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            error_id = str(uuid.uuid4())[:8]
            logger.error(f"[{error_id}] Context error", exc_info=True)
            st.error(f"🚨 {self.fallback_message}")

            if st.button("🔄 Réessayer", key=f"context_retry_{error_id}"):
                st.rerun()

            return True  # suppress exception

        return False
