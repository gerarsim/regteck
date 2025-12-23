# utils/styling.py - VERSION COMPLÈTE CORRIGÉE POUR THÈME SOMBRE
import streamlit as st
import re
from typing import Dict, Any, Optional
import os

def load_css_file(css_file_path: str) -> str:
    """Load CSS from external file."""
    try:
        if os.path.exists(css_file_path):
            with open(css_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            st.warning(f"CSS file not found: {css_file_path}")
            return ""
    except Exception as e:
        st.error(f"Error loading CSS file: {e}")
        return ""

def set_custom_style() -> None:
    """Set custom styles for the Streamlit app with DARK THEME - VERSION CORRIGÉE"""
    try:
        # Load external CSS from assets/style.css
        external_css = load_css_file("assets/style.css")
        
        # THÈME SOMBRE FORCÉ - Combine with comprehensive dark styles
        combined_css = f"""
        <style>
        {external_css}
        
        /* ==============================================
           FORCER LE THÈME SOMBRE - ARRIÈRE-PLAN NOIR
           ============================================== */
        
        /* Variables CSS pour le thème sombre */
        :root {{
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --text-muted: #b0b0b0;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --border-color: #404040;
            --hover-bg: #333333;
        }}

        /* FORCER LE FOND NOIR POUR TOUTE L'APPLICATION */
        .stApp {{
            background-color: var(--bg-primary) !important;
            background-image: none !important;
            color: var(--text-primary) !important;
        }}

        /* Container principal avec fond noir opaque */
        .main .block-container {{
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            margin-top: 1rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7) !important;
        }}

        /* FORCER TOUT LE TEXTE EN BLANC */
        .main, .main *, .stMarkdown, .stText, p, span, div, li, ul, ol {{
            color: var(--text-primary) !important;
        }}

        /* Headers et titres */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            text-shadow: none !important;
        }}

        h1 {{
            font-size: 2.5rem !important;
            text-align: center !important;
            margin-bottom: 2rem !important;
        }}

        h2 {{
            font-size: 2rem !important;
            border-bottom: 2px solid var(--accent-blue) !important;
            padding-bottom: 0.5rem !important;
        }}

        h3 {{
            font-size: 1.5rem !important;
            color: var(--accent-blue) !important;
        }}

        /* CARDS AVEC THÈME SOMBRE */
        .enhanced-card, .status-card, .warning-card, .error-card, .success-card, .info-card {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
            transition: all 0.3s ease !important;
        }}

        .enhanced-card:hover, .status-card:hover, .warning-card:hover, 
        .error-card:hover, .success-card:hover, .info-card:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.7) !important;
            border-color: var(--accent-blue) !important;
        }}

        /* Cards colorées avec borders */
        .error-card {{
            border-left: 4px solid var(--accent-red) !important;
        }}
        .success-card {{
            border-left: 4px solid var(--accent-green) !important;
        }}
        .warning-card {{
            border-left: 4px solid var(--accent-orange) !important;
        }}
        .info-card {{
            border-left: 4px solid var(--accent-blue) !important;
        }}

        /* Tout le texte dans les cards en blanc */
        .enhanced-card *, .status-card *, .warning-card *, 
        .error-card *, .success-card *, .info-card * {{
            color: var(--text-primary) !important;
        }}

        /* BOUTONS AVEC THÈME SOMBRE */
        .stButton > button, button {{
            background-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }}

        .stButton > button:hover, button:hover {{
            background-color: #2563eb !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
        }}

        /* FILE UPLOADER SOMBRE */
        [data-testid="stFileUploader"] > div:first-child {{
            background-color: var(--bg-tertiary) !important;
            border: 2px dashed var(--accent-blue) !important;
            border-radius: 12px !important;
            padding: 2rem !important;
            color: var(--text-primary) !important;
            transition: all 0.3s ease !important;
        }}

        [data-testid="stFileUploader"] > div:first-child:hover {{
            background-color: var(--hover-bg) !important;
            border-color: var(--accent-green) !important;
        }}

        /* INPUTS ET SELECTS SOMBRES */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div,
        .stSelectbox select {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 8px !important;
        }}

        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3) !important;
        }}

        /* TABS SOMBRES */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: var(--bg-tertiary) !important;
            border-radius: 8px !important;
            padding: 4px !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent !important;
            color: var(--text-muted) !important;
            border-radius: 6px !important;
            padding: 12px 20px !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
        }}

        .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
            background-color: var(--hover-bg) !important;
            color: var(--text-primary) !important;
        }}

        /* MÉTRIQUES SOMBRES */
        .metric-box, .score-container {{
            background-color: var(--bg-tertiary) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
            padding: 1rem !important;
            border-radius: 8px !important;
            text-align: center !important;
        }}

        /* SIDEBAR SOMBRE */
        .css-1d391kg, .css-1l02zno {{
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }}

        .css-1d391kg * {{
            color: var(--text-primary) !important;
        }}

        /* DATAFRAMES SOMBRES */
        .stDataFrame {{
            background-color: var(--bg-tertiary) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        .stDataFrame [data-testid="stTable"] {{
            background-color: var(--bg-tertiary) !important;
        }}

        .stDataFrame th {{
            background-color: var(--accent-blue) !important;
            color: var(--text-primary) !important;
        }}

        .stDataFrame td {{
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-bottom: 1px solid var(--border-color) !important;
        }}

        /* PROGRESS BARS SOMBRES */
        .stProgress > div > div {{
            background-color: var(--accent-blue) !important;
        }}

        /* ALERTS ET NOTIFICATIONS SOMBRES */
        .stAlert {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        /* EXPANDER SOMBRE */
        .streamlit-expanderHeader {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        .streamlit-expanderContent {{
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border-color) !important;
            color: var(--text-primary) !important;
        }}

        /* SCROLLBARS SOMBRES */
        ::-webkit-scrollbar {{
            width: 8px !important;
            height: 8px !important;
        }}

        ::-webkit-scrollbar-track {{
            background: var(--bg-secondary) !important;
        }}

        ::-webkit-scrollbar-thumb {{
            background: var(--border-color) !important;
            border-radius: 4px !important;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: var(--text-muted) !important;
        }}

        /* RESPONSIVE MOBILE */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem !important;
                margin: 0.5rem !important;
            }}
            
            .enhanced-card, .status-card, .warning-card, .error-card, .success-card, .info-card {{
                padding: 1rem !important;
            }}
        }}

        /* ANIMATION POUR AMÉLIORER L'UX */
        .enhanced-card, .metric-box, .info-card, .success-card, .warning-card, .error-card {{
            transition: all 0.3s ease !important;
        }}

        .enhanced-card:hover, .metric-box:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5) !important;
        }}

        /* USER INFO BAR SOMBRE */
        .user-info-bar {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin-bottom: 2rem !important;
        }}

        /* SCORE CIRCLE SOMBRE */
        .score-circle {{
            background: conic-gradient(var(--accent-blue) 0% 75%, var(--bg-tertiary) 75% 100%) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-color) !important;
        }}

        .score-value {{
            color: var(--text-primary) !important;
            text-shadow: none !important;
        }}

        .score-label {{
            color: var(--text-secondary) !important;
        }}

        /* PROCESSING ANIMATION SOMBRE */
        .processing-animation {{
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
        }}

        /* CORRECTION APPLIQUÉE POUR CARDS */
        .score-correction-applied {{
            background-color: rgba(16, 185, 129, 0.2) !important;
            border-left: 4px solid var(--accent-green) !important;
            color: var(--text-primary) !important;
            padding: 0.5rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 4px !important;
        }}

        .score-debug {{
            background-color: rgba(245, 158, 11, 0.2) !important;
            color: var(--text-primary) !important;
            padding: 0.5rem !important;
            border-radius: 4px !important;
            border: 1px solid var(--accent-orange) !important;
        }}

        .corrected-score {{
            background-color: rgba(59, 130, 246, 0.2) !important;
            border: 1px solid var(--accent-blue) !important;
            color: var(--text-primary) !important;
            border-radius: 4px !important;
            padding: 0.25rem 0.5rem !important;
            margin: 0.25rem 0 !important;
        }}
        </style>
        """
        
        st.markdown(combined_css, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error applying custom styling: {e}")

def get_card_html(title: str, content: str, card_type: str = "info-card", icon: str = None) -> str:
    """
    Create HTML for a styled card component with dark theme.
    
    Args:
        title: Card title
        content: Card content (HTML)
        card_type: Type of card (status-card, info-card, warning-card, error-card, success-card)
        icon: Optional emoji icon to show before title
        
    Returns:
        HTML string
    """
    # Default icons if none provided
    if icon is None:
        icons = {
            "info-card": "ℹ️",
            "success-card": "✅",
            "warning-card": "⚠️",
            "error-card": "❌",
            "status-card": "📊",
            "enhanced-card": "🎯"
        }
        icon = icons.get(card_type, "📄")
    
    html = f"""
    <div class="{card_type}">
        <h3><span style="margin-right: 0.5rem;">{icon}</span>{title}</h3>
        <div style="margin-top: 1rem;">{content}</div>
    </div>
    """
    
    return html

def apply_emergency_visibility_fix():
    """Apply emergency CSS fix for visibility issues - VERSION SOMBRE"""
    st.markdown("""
    <style>
    /* EMERGENCY DARK THEME FIX */
    .stApp {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
    }
    
    .main .block-container {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #404040 !important;
    }
    
    *, *::before, *::after {
        color: #ffffff !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    .stButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stMarkdown, .stText, p, span, div {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_optimized_banner(title: str = "LexAI", subtitle: str = "Compliance Assistant"):
    """Create an optimized dark banner component."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        border: 1px solid #404040;
    ">
        <h1 style="margin: 0; font-size: 2.5rem; color: #ffffff;">⚖️ {title}</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem; color: #e0e0e0;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def create_simple_metric_card(title: str, value: str, icon: str = "📊", color: str = "blue"):
    """Create a simple dark metric card component."""
    colors = {
        "blue": "#3b82f6",
        "green": "#10b981", 
        "orange": "#f59e0b",
        "purple": "#8b5cf6",
        "red": "#ef4444"
    }
    
    bg_color = colors.get(color, "#3b82f6")
    
    return f"""
    <div style="
        background: {bg_color};
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem;">{value}</div>
        <div style="font-size: 0.9rem; opacity: 0.9;">{title}</div>
    </div>
    """

def create_status_indicator(status: str, message: str = ""):
    """Create a dark status indicator component."""
    status_config = {
        "success": {"color": "#10b981", "icon": "✅"},
        "warning": {"color": "#f59e0b", "icon": "⚠️"},
        "error": {"color": "#ef4444", "icon": "❌"},
        "info": {"color": "#3b82f6", "icon": "ℹ️"},
        "processing": {"color": "#8b5cf6", "icon": "🔄"}
    }
    
    config = status_config.get(status, status_config["info"])
    
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: #2a2a2a;
        border-left: 4px solid {config['color']};
        border-radius: 4px;
        margin: 1rem 0;
        border: 1px solid #404040;
        color: #ffffff;
    ">
        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{config['icon']}</span>
        <span style="color: #ffffff; font-weight: 600;">{message}</span>
    </div>
    """, unsafe_allow_html=True)

def initialize_demo_styling():
    """Initialize demo-specific dark styling."""
    demo_styles = {
        "theme": "dark",
        "primary_color": "#3b82f6",
        "background_color": "#0a0a0a",
        "secondary_background": "#1a1a1a",
        "text_color": "#ffffff"
    }
    
    st.markdown("""
    <style>
    /* Demo-specific dark styles */
    .demo-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .demo-header {
        text-align: center;
        margin-bottom: 3rem;
        color: #ffffff;
    }
    
    .demo-section {
        margin-bottom: 2rem;
        padding: 1.5rem;
        border-radius: 8px;
        background-color: #2a2a2a;
        border: 1px solid #404040;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    return demo_styles

def check_styling_compatibility():
    """Check styling system compatibility."""
    compatibility_info = {
        "streamlit_version": st.__version__ if hasattr(st, '__version__') else "unknown",
        "css_support": True,
        "html_support": True,
        "markdown_support": True,
        "component_support": True,
        "dark_theme_applied": True
    }
    
    # Test basic functionality
    try:
        st.markdown("<div style='color: #ffffff;'>Test</div>", unsafe_allow_html=True)
        compatibility_info["html_injection"] = True
    except:
        compatibility_info["html_injection"] = False
    
    return compatibility_info

# Backward compatibility aliases
create_emergency_banner = create_optimized_banner
create_emergency_metric_card = create_simple_metric_card

def format_score_decimal(score: any) -> float:
    """
    Formate correctement un score en décimal - VERSION CORRIGÉE POUR STYLING
    """
    try:
        if score is None or (isinstance(score, str) and not score.strip()):
            return 0.0
        
        if isinstance(score, (int, float)):
            numeric_score = float(score)
        else:
            score_str = str(score).strip()
            cleaned = re.sub(r'[^\d.,-]', '', score_str)
            cleaned = cleaned.replace(',', '.')
            
            if not cleaned:
                return 0.0
                
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                if len(parts) > 2:
                    cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
            
            try:
                numeric_score = float(cleaned)
            except ValueError:
                return 0.0
        
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
        return 0.0

def render_compliance_score(score: float, size: str = "large", label: str = None) -> None:
    """
    Render an enhanced compliance score visualization with dark theme.
    """
    corrected_score = format_score_decimal(score)
    score_percent = int(corrected_score)
    score_display = f"{corrected_score:.2f}%".replace('.', ',')
    
    if corrected_score >= 80.0:
        color = "#10b981"
        status = label or "Excellente conformité"
    elif corrected_score >= 60.0:
        color = "#f59e0b"
        status = label or "Conformité acceptable"
    else:
        color = "#ef4444"
        status = label or "Risque de non-conformité"
    
    if size == "small":
        circle_size = 100
        font_size = 1.5
    elif size == "medium":
        circle_size = 130
        font_size = 2
    else:
        circle_size = 160
        font_size = 2.5
    
    st.markdown(f"""
    <div class="score-container">
        <div style="
            width: {circle_size}px; 
            height: {circle_size}px;
            border-radius: 50%;
            background: conic-gradient({color} 0% {score_percent}%, #2a2a2a {score_percent}% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
            border: 2px solid #404040;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        ">
            <div style="
                font-size: {font_size}rem;
                font-weight: bold;
                color: #ffffff;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            ">
                {score_display}
            </div>
        </div>
        <div style="color: #e0e0e0; margin-top: 1rem; text-align: center; font-size: 1.2rem;">{status}</div>
    </div>
    """, unsafe_allow_html=True)

def render_processing_animation(message: str = "Analyse en cours...") -> None:
    """Render an enhanced dark processing animation."""
    st.markdown(f"""
    <div style="
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    ">
        <h3 style="color: #ffffff; margin-bottom: 1rem;">🔄 {message}</h3>
        <div style="display: flex; justify-content: center; gap: 8px;">
            <div style="width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; animation: pulse 1.5s infinite ease-in-out;"></div>
            <div style="width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; animation: pulse 1.5s infinite ease-in-out 0.2s;"></div>
            <div style="width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; animation: pulse 1.5s infinite ease-in-out 0.4s;"></div>
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(0.8); opacity: 0.5; }}
        50% {{ transform: scale(1.3); opacity: 1; }}
        100% {{ transform: scale(0.8); opacity: 0.5; }}
    }}
    </style>
    """, unsafe_allow_html=True)

def render_lux_flag_accent():
    """Render Luxembourg flag colored accent line."""
    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ef4444 33.33%, #ffffff 33.33%, #ffffff 66.66%, #3b82f6 66.66%);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    "></div>
    """, unsafe_allow_html=True)

def highlight_text(text: str, keywords: list, highlight_color: str = "#f59e0b") -> str:
    """
    Highlight keywords in text with dark theme.
    """
    if not text or not keywords:
        return text
    
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    escaped_text = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace('"', "&quot;")
                       .replace("'", "&#39;"))
    
    for keyword in sorted_keywords:
        if not keyword:
            continue
        
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        escaped_text = pattern.sub(
            f'<span style="background: {highlight_color}; color: #000; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\\g<0></span>', 
            escaped_text
        )
    
    return escaped_text

def create_collapsible_section(header: str, content: str, expanded: bool = False) -> str:
    """
    Create an enhanced collapsible HTML section with dark theme.
    """
    expanded_attr = "open" if expanded else ""
    
    html = f"""
    <details {expanded_attr} style="
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        margin-bottom: 1rem;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    ">
        <summary style="
            padding: 1.5rem;
            cursor: pointer;
            background-color: #1a1a1a;
            font-weight: 600;
            color: #ffffff;
            border-bottom: 1px solid #404040;
        ">📋 {header}</summary>
        <div style="padding: 1.5rem; color: #ffffff;">{content}</div>
    </details>
    """
    
    return html

def create_enhanced_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    """Create an enhanced metric card with dark theme."""
    delta_html = ""
    if delta:
        delta_colors = {
            "normal": "#3b82f6",
            "inverse": "#ef4444",
            "off": "#e0e0e0"
        }
        color = delta_colors.get(delta_color, "#3b82f6")
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">📈 {delta}</div>'
    
    return f"""
    <div style="
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    ">
        <div style="font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">{title}</div>
        <div style="font-size: 2.5rem; font-weight: bold; color: #ffffff;">{value}</div>
        {delta_html}
    </div>
    """

def render_enhanced_banner_dark():
    """Render enhanced dark banner for the application."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        color: #ffffff;
        padding: 3rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 16px 64px rgba(0, 0, 0, 0.7);
        border: 2px solid #404040;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ef4444 33.33%, #ffffff 33.33%, #ffffff 66.66%, #3b82f6 66.66%);
        "></div>
        <h1 style="
            margin: 0;
            font-size: 3rem;
            color: #ffffff;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
            background: linear-gradient(135deg, #ffffff, #e0e7ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">⚖️ LexAI</h1>
        <p style="
            margin: 1rem 0 0 0;
            opacity: 0.9;
            font-size: 1.4rem;
            color: #e0e0e0;
            font-weight: 300;
        ">Assistant de Conformité Réglementaire</p>
        <p style="
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
            color: #b0b0b0;
        ">🌙 Interface Optimisée Thème Sombre</p>
    </div>
    """, unsafe_allow_html=True)

def create_dark_theme_card(title: str, content: str, card_type: str = "default", icon: str = "📄"):
    """Create a specialized dark theme card."""
    
    # Define card types with their specific styling
    card_styles = {
        "default": {
            "bg": "#2a2a2a",
            "border": "#404040",
            "accent": "#3b82f6"
        },
        "success": {
            "bg": "rgba(16, 185, 129, 0.1)",
            "border": "#10b981",
            "accent": "#10b981"
        },
        "warning": {
            "bg": "rgba(245, 158, 11, 0.1)",
            "border": "#f59e0b", 
            "accent": "#f59e0b"
        },
        "error": {
            "bg": "rgba(239, 68, 68, 0.1)",
            "border": "#ef4444",
            "accent": "#ef4444"
        },
        "info": {
            "bg": "rgba(59, 130, 246, 0.1)",
            "border": "#3b82f6",
            "accent": "#3b82f6"
        }
    }
    
    style = card_styles.get(card_type, card_styles["default"])
    
    st.markdown(f"""
    <div style="
        background-color: {style['bg']};
        border: 2px solid {style['border']};
        border-left: 6px solid {style['accent']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        color: #ffffff;
    ">
        <div style="
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            color: {style['accent']};
        ">
            <span style="margin-right: 0.75rem; font-size: 1.5rem;">{icon}</span>
            {title}
        </div>
        <div style="color: #e0e0e0; line-height: 1.6;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)

def apply_dark_theme_fix():
    """Apply comprehensive dark theme fix for all Streamlit components."""
    st.markdown("""
    <style>
    /* ==============================================
       FIX COMPLET THÈME SOMBRE - TOUS COMPOSANTS
       ============================================== */
    
    /* Variables globales */
    :root {
        --dark-bg-primary: #0a0a0a;
        --dark-bg-secondary: #1a1a1a;
        --dark-bg-tertiary: #2a2a2a;
        --dark-text-primary: #ffffff;
        --dark-text-secondary: #e0e0e0;
        --dark-border: #404040;
        --dark-accent: #3b82f6;
    }
    
    /* Force l'application complète */
    .stApp, .main, .block-container {
        background-color: var(--dark-bg-primary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Tous les éléments texte */
    *, *::before, *::after, p, span, div, h1, h2, h3, h4, h5, h6, 
    .stMarkdown, .stText, .stCaption, .stCode, .stLatex {
        color: var(--dark-text-primary) !important;
    }
    
    /* Conteneurs et blocks */
    .main .block-container {
        background-color: var(--dark-bg-secondary) !important;
        border: 1px solid var(--dark-border) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.7) !important;
    }
    
    /* Boutons */
    .stButton > button, button, .stDownloadButton > button {
        background-color: var(--dark-accent) !important;
        color: var(--dark-text-primary) !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    .stButton > button:hover, button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px) !important;
    }
    
    /* Inputs et formulaires */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
        border: 1px solid var(--dark-border) !important;
    }
    
    /* File uploader */
    .stFileUploader > div:first-child,
    [data-testid="stFileUploader"] > div:first-child {
        background-color: var(--dark-bg-tertiary) !important;
        border: 2px dashed var(--dark-accent) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--dark-bg-tertiary) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #b0b0b0 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--dark-accent) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1l02zno, .stSidebar {
        background-color: var(--dark-bg-secondary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Métriques */
    .metric-container, [data-testid="metric-container"] {
        background-color: var(--dark-bg-tertiary) !important;
        border: 1px solid var(--dark-border) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* DataFrames */
    .stDataFrame, [data-testid="stTable"] {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    .stDataFrame th {
        background-color: var(--dark-accent) !important;
        color: var(--dark-text-primary) !important;
    }
    
    .stDataFrame td {
        background-color: var(--dark-bg-secondary) !important;
        color: var(--dark-text-primary) !important;
        border-color: var(--dark-border) !important;
    }
    
    /* Alerts et messages */
    .stAlert, .stSuccess, .stError, .stWarning, .stInfo {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
        border: 1px solid var(--dark-border) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--dark-bg-secondary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background-color: var(--dark-accent) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--dark-accent) transparent transparent transparent !important;
    }
    
    /* Colonnes et containers */
    .element-container, .stHorizontal, .stVertical {
        color: var(--dark-text-primary) !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        background: var(--dark-bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--dark-border);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #606060;
    }
    
    /* Cache et performance */
    .stCaching {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Code blocks */
    .stCodeBlock, pre, code {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
        border: 1px solid var(--dark-border) !important;
    }
    
    /* JSON viewer */
    .stJson {
        background-color: var(--dark-bg-tertiary) !important;
        color: var(--dark-text-primary) !important;
    }
    
    /* Radio buttons et checkboxes */
    .stRadio > div, .stCheckbox > div {
        color: var(--dark-text-primary) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        color: var(--dark-text-primary) !important;
    }
    
    /* Camera input */
    .stCameraInput > div {
        background-color: var(--dark-bg-tertiary) !important;
        border: 2px dashed var(--dark-border) !important;
    }
    
    /* Audio et video */
    .stAudio, .stVideo {
        background-color: var(--dark-bg-tertiary) !important;
    }
    
    /* Balloons et snow */
    .stBalloons, .stSnow {
        filter: brightness(0.8) contrast(1.2);
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem !important;
            margin: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_dark_theme_status():
    """Render dark theme status indicator."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        border: 1px solid #404040;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    ">
        <div style="display: flex; align-items: center; color: #ffffff;">
            <span style="font-size: 1.5rem; margin-right: 0.75rem;">🌙</span>
            <div>
                <div style="font-weight: 600; font-size: 1.1rem;">Thème Sombre Activé</div>
                <div style="font-size: 0.9rem; color: #b0b0b0;">Interface optimisée pour réduire la fatigue oculaire</div>
            </div>
        </div>
        <div style="
            background-color: #10b981;
            color: #ffffff;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        ">
            ACTIF
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_dark_info_panel(title: str, items: list, icon: str = "ℹ️"):
    """Create an information panel with dark theme."""
    items_html = ""
    for item in items:
        items_html += f'<li style="margin: 0.5rem 0; color: #e0e0e0;">{item}</li>'
    
    st.markdown(f"""
    <div style="
        background-color: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    ">
        <h3 style="
            color: #3b82f6;
            margin: 0 0 1rem 0;
            display: flex;
            align-items: center;
            font-size: 1.2rem;
        ">
            <span style="margin-right: 0.5rem;">{icon}</span>
            {title}
        </h3>
        <ul style="
            margin: 0;
            padding-left: 1.5rem;
            color: #ffffff;
        ">
            {items_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)