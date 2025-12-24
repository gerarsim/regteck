# streamlit_app.py - Optimized Version with Flicker-Free Language, Enhanced History, and Improved Feedback
import streamlit as st
import streamlit.components.v1 as components
import sys, os, logging, time, uuid, re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# ========================================================================
# CONFIGURATION AND PATH SETUP
# ========================================================================

current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Page config
try:
    st.set_page_config(
        page_title="LexAI - Regulatory Compliance Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
except:
    pass

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ========================================================================
# OLLAMA CONFIGURATION
# ========================================================================

USE_LLM = os.getenv('USE_LLM_ANALYZER', 'false').lower() == 'true'
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://192.168.1.24:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
LLM_MODE = os.getenv('LLM_ANALYSIS_MODE', 'hybrid')
logger.info(f"🔧 Ollama config: USE_LLM={USE_LLM}, HOST={OLLAMA_HOST}, MODEL={OLLAMA_MODEL}, MODE={LLM_MODE}")

# ========================================================================
# UTILS - SCORE FORMATTING
# ========================================================================

def format_score_decimal(score: Any) -> float:
    """Correct decimal formatting for all score types"""
    try:
        if score is None or (isinstance(score, str) and not score.strip()):
            return 0.0
        if isinstance(score, (int, float)):
            numeric_score = float(score)
        else:
            cleaned = re.sub(r'[^\d.,-]', '', str(score)).replace(',', '.')
            numeric_score = float(cleaned) if cleaned else 0.0
        if numeric_score < 0:
            return 0.0
        elif numeric_score > 10000:
            return round(min(100.0, numeric_score / 10000), 2)
        elif numeric_score > 1000:
            return round(min(100.0, numeric_score / 100), 2)
        elif numeric_score > 100:
            return 100.0
        elif numeric_score <= 1:
            return round(numeric_score * 100.0, 2)
        return round(numeric_score, 2)
    except Exception as e:
        logger.error(f"Error in format_score_decimal('{score}'): {e}")
        return 0.0

def fix_analysis_result_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    """Apply decimal formatting to all relevant fields"""
    if not isinstance(result, dict):
        return result
    result = result.copy()
    score_fields = ['score', 'final_score', 'enhanced_score', 'base_score',
                    'excellence_score', 'bonus_points', 'luxembourg_relevance',
                    'overall_score', 'confidence_score']
    for field in score_fields:
        if field in result:
            result[field] = format_score_decimal(result[field])
    if 'issues' in result and isinstance(result['issues'], list):
        for issue in result['issues']:
            for score_field in ['confidence_score', 'weight', 'penalty_score']:
                if score_field in issue:
                    issue[score_field] = format_score_decimal(issue[score_field])
    if 'final_score' not in result and 'score' in result:
        result['final_score'] = result['score']
    result['score_corrections_applied'] = True
    return result

# ========================================================================
# SAFE IMPORTS WITH FALLBACK
# ========================================================================

try:
    from utils.session_manager import SessionManager
    SESSION_MANAGER_AVAILABLE = True
except ImportError:
    SESSION_MANAGER_AVAILABLE = False
    class SessionManager:
        @classmethod
        def init_session(cls):
            defaults = {'current_analysis': None, 'analysis_history': [],
                        'language': 'fr', 'current_page': 'home', 'user': {}, 'authenticated': False}
            for k,v in defaults.items():
                if k not in st.session_state: st.session_state[k]=v
        @classmethod
        def get(cls,key,default=None): cls.init_session(); return st.session_state.get(key,default)
        @classmethod
        def set(cls,key,value): cls.init_session(); st.session_state[key]=value
        @classmethod
        def add_analysis(cls,data): cls.init_session(); data['timestamp']=datetime.now().isoformat(); st.session_state['analysis_history'].insert(0,data)

try:
    from utils.text_extraction import extract_text
    TEXT_EXTRACTION_AVAILABLE = True
except ImportError:
    TEXT_EXTRACTION_AVAILABLE = False
    def extract_text(file, detect_lang=False):
        try:
            content = file.read().decode('utf-8', errors='ignore')
            return content, 'fr' if detect_lang else None
        except:
            return str(file), 'fr' if detect_lang else None

# ========================================================================
# OLLAMA LLM ANALYZER
# ========================================================================

ANALYZER_TYPE = "unknown"
OLLAMA_AVAILABLE = False
analyze_regulatory_compliance = lambda text, doc_type="auto", language="auto": fix_analysis_result_scores({"score":70,"issues":[],"recommendations":["Install analysis engine"],"overall_assessment":"Fallback mode"})

if USE_LLM:
    try:
        from utils.llm_analyzer_ollama import analyze_with_local_llm, check_ollama_status
        status = check_ollama_status()
        if status.get('running', False):
            OLLAMA_AVAILABLE = True
            ANALYZER_TYPE = "ollama_llm"
            def analyze_regulatory_compliance(text, doc_type="auto", language="auto"):
                return fix_analysis_result_scores(analyze_with_local_llm(
                    text=text, doc_type=doc_type, language=language, model=OLLAMA_MODEL, mode=LLM_MODE))
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")

if not OLLAMA_AVAILABLE:
    ANALYZER_TYPE = "rule_based"

# ========================================================================
# INTERNATIONALIZATION
# ========================================================================

LANGUAGES = {'fr':'🇫🇷 Français','en':'🇬🇧 English','de':'🇩🇪 Deutsch','es':'🇪🇸 Español'}

def get_localized_text(key:str, language:str=None) -> str:
    if language is None: language = SessionManager.get('language','fr')
    translations = {
        'app_title': {'fr':'Assistant de Conformité Réglementaire','en':'Regulatory Compliance Assistant','de':'Regulatorischer Compliance-Assistent','es':'Asistente de Cumplimiento Normativo'},
        'document_analysis': {'fr':'Analyse Document','en':'Document Analysis','de':'Dokumentanalyse','es':'Análisis de Documento'},
        'upload_document': {'fr':'Déposez votre document','en':'Upload your document','de':'Laden Sie Ihr Dokument hoch','es':'Suba su documento'},
        'analyze_document': {'fr':'ANALYSER LE DOCUMENT','en':'ANALYZE DOCUMENT','de':'DOKUMENT ANALYSIEREN','es':'ANALIZAR DOCUMENTO'},
        'analysis_in_progress': {'fr':'Analyse en cours...' if not OLLAMA_AVAILABLE else 'Analyse IA en cours...',
                                 'en':'Analysis in progress...' if not OLLAMA_AVAILABLE else 'AI Analysis in progress...'}
    }
    return translations.get(key, {}).get(language, key.replace('_',' ').title())

def render_language_selector():
    current_lang = SessionManager.get('language','fr')
    selected_lang = st.selectbox("🌍 Language", options=list(LANGUAGES.keys()), format_func=lambda x: LANGUAGES[x], index=list(LANGUAGES.keys()).index(current_lang), key="main_lang_selector")
    if selected_lang != current_lang:
        SessionManager.set('language', selected_lang)
        SessionManager.set('language_changed', True)

# ========================================================================
# ANALYSIS WITH FEEDBACK
# ========================================================================

def analyze_document_with_feedback(text:str, doc_type:str="auto", language:str="fr") -> Dict[str,Any]:
    start_time = time.time()
    try:
        word_count = len(text.split())
        if word_count>5000:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            for i in range(0,100,10):
                progress_text.text(f"Analyzing large document ({word_count} words)... {i}%")
                progress_bar.progress(i)
                time.sleep(0.05)
        result = analyze_regulatory_compliance(text, doc_type, language)
        result['processing_time'] = round(time.time()-start_time,3)
        result['analyzer_type'] = ANALYZER_TYPE
        result['ollama_powered'] = OLLAMA_AVAILABLE
        result['ollama_model'] = OLLAMA_MODEL if OLLAMA_AVAILABLE else None
        result['analysis_mode'] = LLM_MODE if OLLAMA_AVAILABLE else 'rule_based'
        return fix_analysis_result_scores(result)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return fix_analysis_result_scores({
            "score":50,"issues":[],"recommendations":[f"Error: {str(e)}"],"overall_assessment":"Error","processing_time":round(time.time()-start_time,3)
        })

# ========================================================================
# LEXAI APP CLASS
# ========================================================================

class LexAIApp:
    def __init__(self):
        st.markdown("""
        <style>
        .stApp {background-color:#0a0a0a;color:#fff;}
        .main .block-container {background-color:#1a1a1a;color:#fff;border:1px solid #404040;border-radius:12px;padding:2rem;}
        .stButton > button {background-color:#3b82f6;color:#fff;border-radius:8px;font-weight:600;}
        .stButton > button:hover {background-color:#2563eb;}
        </style>""",unsafe_allow_html=True)
        SessionManager.init_session()
        logger.info("LexAI initialized")

    def run(self):
        # Flicker-free language change
        if SessionManager.get('language_changed',False):
            SessionManager.set('language_changed',False)
            st.experimental_rerun()

        st.markdown("# ⚖️ LexAI - " + get_localized_text('app_title'))
        render_language_selector()
        st.markdown("---")
        tab1,tab2,tab3 = st.tabs([f"📄 {get_localized_text('document_analysis')}","📊 History","🔧 Settings"])
        with tab1: self.create_upload_section()
        with tab2: self.render_history()
        with tab3: self.render_settings()

    def create_upload_section(self):
        st.markdown(f"### 📄 {get_localized_text('document_analysis')}")
        uploaded_file = st.file_uploader(f"📁 {get_localized_text('upload_document')}", type=['pdf','docx','txt'])
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            if st.button(f"🚀 {get_localized_text('analyze_document')}",use_container_width=True):
                with st.spinner(get_localized_text('analysis_in_progress')):
                    text_content, detected_lang = extract_text(uploaded_file, detect_lang=True) if TEXT_EXTRACTION_AVAILABLE else (uploaded_file.read().decode('utf-8',errors='ignore'),'fr')
                    if not text_content.strip(): st.error("Cannot extract text"); return
                    result = analyze_document_with_feedback(text_content,language=SessionManager.get('language','fr'))
                    analysis_data = result.copy(); analysis_data['file_name']=uploaded_file.name; analysis_data['timestamp']=datetime.now().isoformat()
                    SessionManager.add_analysis(analysis_data)
                    final_score = format_score_decimal(result.get('final_score',0))
                    st.success(f"✅ Analysis completed in {result.get('processing_time',0):.2f}s - Score: {final_score:.2f}%")
                    st.balloons()
                    self.display_results(result, uploaded_file.name)

    def display_results(self,result,filename):
        st.markdown("---")
        st.markdown(f"## 📋 Analysis Results - {filename}")
        result = fix_analysis_result_scores(result)
        score = format_score_decimal(result.get('final_score',0))
        issues = result.get('issues',[])
        processing_time = result.get('processing_time',0)
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            color='🟢' if score>=80 else '🟡' if score>=60 else '🔴'
            st.metric("📊 Score",f"{score:.2f}%",color)
        with col2: st.metric("⚠️ Issues",str(len(issues)))
        with col3: st.metric("⏱️ Time",f"{processing_time:.2f}s")
        with col4: st.metric("Engine","🤖 AI" if result.get('ollama_powered',False) else "📋 Rules")
        if issues:
            st.markdown("### ⚠️ Compliance Issues")
            for i,issue in enumerate(issues,1):
                desc = issue.get('description','N/A'); sev = issue.get('severity','medium')
                conf = format_score_decimal(issue.get('confidence_score',0))
                icon={'critical':'🚨','high':'⚠️','medium':'🟡','low':'🔵'}.get(sev,'⚠️')
                with st.expander(f"{icon} {desc}"):
                    col1,col2=st.columns(2)
                    with col1: st.write(f"**Severity:** {sev}"); st.write(f"**Confidence:** {conf:.2f}%")
                    with col2: st.write(f"**Basis:** {issue.get('regulatory_basis','N/A')}"); st.write(f"**Action:** {issue.get('suggested_action','TBD')}")
        else: st.success("✅ No compliance issues detected!")
        recs = result.get('recommendations',[])
        if recs:
            st.markdown("### 💡 Recommendations")
            for i,rec in enumerate(recs,1): st.info(f"**{i}.** {rec}")

    def render_history(self):
        st.markdown("### 📊 Analysis History")
        history = SessionManager.get('analysis_history',[])
        if history:
            for analysis in history[:20]:
                analysis = fix_analysis_result_scores(analysis)
                score = format_score_decimal(analysis.get('final_score',0))
                timestamp = analysis.get('timestamp','')[:16].replace('T',' ')
                file_name = analysis.get('file_name','File')
                with st.expander(f"📄 {file_name} - {score:.2f}% - {timestamp}"):
                    col1,col2 = st.columns(2)
                    with col1: st.write(f"**Score:** {score:.2f}%"); st.write(f"**Issues:** {len(analysis.get('issues',[]))}")
                    with col2: st.write(f"**Engine:** {'🤖 AI' if analysis.get('ollama_powered',False) else '📋 Rules'}"); st.write(f"**Time:** {analysis.get('processing_time',0):.2f}s")
            if st.button("🧹 Clear History"): SessionManager.set('analysis_history',[]); st.success("✅ Cleared"); st.experimental_rerun()
        else: st.info("No analyses yet")

    def render_settings(self):
        st.markdown("### ⚙️ System Settings")
        col1,col2=st.columns(2)
        with col1:
            st.markdown("**🤖 AI Configuration**")
            st.info(f"Ollama: {'✅ Connected' if OLLAMA_AVAILABLE else '❌ Not available'}")
            if OLLAMA_AVAILABLE:
                st.info(f"Host: {OLLAMA_HOST}"); st.info(f"Model: {OLLAMA_MODEL}"); st.info(f"Mode: {LLM_MODE}")
            else:
                st.warning("To enable AI analysis:\n```bash\ncurl -fsSL https://ollama.com/install.sh | sh\nollama pull llama3.2:3b\n```")
        with col2:
            st.markdown("**📊 System Status**")
            for name,status in [("Analyzer",ANALYZER_TYPE),("Ollama","✅" if OLLAMA_AVAILABLE else "❌"),("Text Extraction","✅" if TEXT_EXTRACTION_AVAILABLE else "❌"),("Session Manager","✅" if SESSION_MANAGER_AVAILABLE else "❌")]:
                st.write(f"**{name}:** {status}")
        if st.button("🧪 Test Ollama Connection"):
            if OLLAMA_AVAILABLE:
                try: from utils.llm_analyzer_ollama import check_ollama_status; status=check_ollama_status(); st.success("✅ Connected"); st.json(status)
                except Exception as e: st.error(f"❌ Connection failed: {e}")
            else: st.warning("⚠️ Ollama not configured")

# ========================================================================
# MAIN ENTRY POINT
# ========================================================================

def main():
    try: LexAIApp().run()
    except Exception as e: st.error(f"🚨 Error: {e}"); logger.error(f"Application error: {e}")

if __name__=="__main__": main()