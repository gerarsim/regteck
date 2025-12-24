#!/usr/bin/env python3
"""
Ollama/Compliance Engine Health Checker (Refactored)
Supports local engine and Ollama LLM
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# =========================
# Logging Setup
# =========================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =========================
# Helper: Requests Session with Retry
# =========================
def requests_session_with_retries(total: int = 3, backoff: float = 0.5) -> requests.Session:
    session = requests.Session()
    retries = Retry(total=total, backoff_factor=backoff, status_forcelist=[500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


# =========================
# Ollama Health Checker
# =========================
class OllamaHealthChecker:
    def __init__(self):
        self.host = os.environ.get("OLLAMA_HOST") or \
                    os.environ.get("OLLAMA_BASE_URL") or \
                    os.environ.get("OLLAMA_API_URL") or \
                    "http://127.0.0.1:11434"
        self.model = os.environ.get("OLLAMA_MODEL") or os.environ.get("LLM_MODEL") or "llama3.2:3b"
        self.timeout = int(os.environ.get("OLLAMA_TIMEOUT", "10"))
        self.session = requests_session_with_retries()

    def check_ollama_status(self) -> Dict[str, Any]:
        result = {
            "timestamp": datetime.now().isoformat(),
            "host": self.host,
            "configured_model": self.model,
            "reachable": False,
            "api_responding": False,
            "models_available": [],
            "model_ready": False,
            "status": "unknown",
            "errors": [],
            "recommendations": []
        }

        # Host reachability
        try:
            r = self.session.get(f"{self.host}/", timeout=self.timeout)
            result["reachable"] = True
            logger.info(f"✅ Ollama host reachable: {self.host}")
        except requests.exceptions.RequestException as e:
            result["errors"].append(f"Cannot reach Ollama: {e}")
            result["status"] = "unreachable"
            result["recommendations"].extend([
                "Ensure Ollama is running (systemctl status ollama or ollama serve)",
                "Check Docker host IP if inside container (e.g., http://172.17.0.1:11434)"
            ])
            return result

        # API check
        try:
            r = self.session.get(f"{self.host}/api/tags", timeout=self.timeout)
            if r.status_code == 200:
                result["api_responding"] = True
                data = r.json()
                models = data.get("models", [])
                result["models_available"] = [m.get("name", m.get("model", "unknown")) for m in models]
                logger.info(f"✅ Ollama API responding, {len(models)} models available")
            else:
                result["errors"].append(f"API returned status {r.status_code}")
                result["status"] = "api_error"
                return result
        except requests.exceptions.RequestException as e:
            result["errors"].append(f"API request failed: {e}")
            result["status"] = "api_error"
            return result

        # Model availability
        if result["models_available"]:
            model_found = any(
                self.model in m or m.startswith(self.model.split(":")[0])
                for m in result["models_available"]
            )
            if model_found:
                result["model_ready"] = True
                result["status"] = "operational"
                logger.info(f"✅ Configured model '{self.model}' is available")
            else:
                result["status"] = "model_missing"
                result["errors"].append(f"Configured model '{self.model}' not found")
                result["recommendations"].append(f"Pull model: ollama pull {self.model}")
        else:
            result["status"] = "no_models"
            result["errors"].append("No models installed in Ollama")
            result["recommendations"].append(f"Pull model: ollama pull {self.model}")

        return result

    def test_generation(self, prompt: str = "Say 'OK' if you are working.") -> Dict[str, Any]:
        result = {"success": False, "response": None, "latency_ms": None, "error": None}
        try:
            import time
            start = time.time()
            r = self.session.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "options": {"num_predict": 50}},
                timeout=60
            )
            result["latency_ms"] = round((time.time() - start) * 1000, 2)
            if r.status_code == 200:
                data = r.json()
                result["success"] = True
                result["response"] = data.get("response", "")[:200]
                logger.info(f"✅ Generation test passed in {result['latency_ms']}ms")
            else:
                result["error"] = f"Generation failed with status {r.status_code}"
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Generation test failed: {e}")
        return result


# =========================
# Local Engine Health Checker
# =========================
class EngineHealthChecker:
    def __init__(self, base_dir: str = None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if base_dir is None:
            base_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) == "utils" else current_dir
        self.base_dir = base_dir
        self.data_dir = os.environ.get("DATA_DIR") or os.path.join(base_dir, "data")
        self.required_files = [
            "analyses.json", "compliance_rules.json", "compliance_penalties.json",
            "cross_border_regulations.json", "dynamic_rules.json", "financial_institutions.json",
            "issue_descriptions.json", "lux_keywords.json", "regulations.json",
            "reporting_requirements.json", "sanctions_lists.json"
        ]

    def check_engine_installation(self) -> Dict[str, Any]:
        result = {"timestamp": datetime.now().isoformat(), "errors": [], "warnings": [], "recommendations": []}
        result.update(self._check_engine_module())
        result["data_files"] = self._check_data_files()
        return self._determine_status(result)

    def _check_engine_module(self) -> Dict[str, Any]:
        result = {"installed": False, "running": False, "engine_type": "none", "models": [], "features": []}
        try:
            if self.base_dir not in sys.path:
                sys.path.insert(0, self.base_dir)
            from engine import LocalComplianceEngine
            result["installed"] = True
            result["engine_type"] = "local_json_based"
            try:
                engine = LocalComplianceEngine(self.data_dir)
                stats = getattr(engine, "get_analysis_statistics", lambda: {})()
                result.update({
                    "running": True,
                    "models": ["local_engine_v4.0"],
                    "features": [
                        "Rule-based analysis", "Luxembourg-specific compliance",
                        "Multi-language support", "Banking sector optimization"
                    ],
                    "statistics": stats
                })
            except Exception as e:
                result["running"] = False
                result["errors"] = [f"Engine initialization failed: {e}"]
        except ImportError as e:
            result["installed"] = False
            result["errors"] = [f"Engine module not found: {e}"]
        return result

    def _check_data_files(self) -> Dict[str, Any]:
        data_info = {"total_required": len(self.required_files), "found": 0, "missing": [], "valid": [], "invalid": [], "corrupted": [], "total_size_mb": 0.0, "details": {}}
        if not os.path.exists(self.data_dir):
            data_info["errors"] = [f"Data directory not found: {self.data_dir}"]
            data_info["missing"] = self.required_files.copy()
            return data_info

        for f in self.required_files:
            path = os.path.join(self.data_dir, f)
            if os.path.exists(path):
                is_valid, error, size_mb, records = self._validate_json(path)
                data_info["found"] += 1
                if is_valid:
                    data_info["valid"].append(f)
                    data_info["total_size_mb"] += size_mb
                else:
                    data_info["invalid"].append(f)
                    data_info["corrupted"].append(f)
                data_info["details"][f] = {"path": path, "valid": is_valid, "size_mb": round(size_mb, 2), "records": records, "error": error}
            else:
                data_info["missing"].append(f)
                data_info["details"][f] = {"path": path, "valid": False, "error": "File not found"}
        return data_info

    @staticmethod
    def _validate_json(filepath: str) -> Tuple[bool, str, float, int]:
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = len(data) if isinstance(data, (dict, list)) else 1
            return True, None, size_mb, records
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}", 0.0, 0
        except Exception as e:
            return False, f"Read error: {e}", 0.0, 0

    def _determine_status(self, result: Dict[str, Any]) -> Dict[str, Any]:
        df = result.get("data_files", {})
        found_ratio = df.get("found", 0) / max(df.get("total_required", 1), 1)
        valid_ratio = len(df.get("valid", [])) / max(df.get("total_required", 1), 1)
        if not result.get("installed"):
            result["status"] = "critical"
            result["warnings"].append("Engine module not installed")
        elif not result.get("running"):
            result["status"] = "error"
            result["warnings"].append("Engine installed but not running")
        elif found_ratio < 0.5:
            result["status"] = "critical"
            result["warnings"].append("Less than 50% of required data files found")
        elif found_ratio < 1.0:
            result["status"] = "degraded"
            result["warnings"].append(f"{df['total_required'] - df['found']} data files missing")
        elif valid_ratio < 1.0:
            result["status"] = "warning"
            result["warnings"].append(f"{len(df.get('invalid', []))} invalid data files")
        else:
            result["status"] = "operational"
        return result


# =========================
# Combined Health Checker
# =========================
class CombinedHealthChecker:
    def __init__(self, base_dir: str = None):
        self.ollama = OllamaHealthChecker()
        self.engine = EngineHealthChecker(base_dir)

    def check_all(self) -> Dict[str, Any]:
        result = {
            "timestamp": datetime.now().isoformat(),
            "ollama": self.ollama.check_ollama_status(),
            "local_engine": self.engine.check_engine_installation(),
            "overall_status": "unknown",
            "analysis_mode": "fallback",
            "recommendations": []
        }

        ollama_ok = result["ollama"]["status"] == "operational"
        engine_ok = result["local_engine"]["status"] == "operational"

        if ollama_ok and engine_ok:
            result["overall_status"] = "optimal"
            result["analysis_mode"] = "hybrid"
        elif ollama_ok:
            result["overall_status"] = "llm_only"
            result["analysis_mode"] = "llm_only"
            result["recommendations"].append("Local engine not available - using LLM only")
        elif engine_ok:
            result["overall_status"] = "engine_only"
            result["analysis_mode"] = "rules_only"
            result["recommendations"].append("Ollama not available - using rules-based analysis")
        else:
            result["overall_status"] = "degraded"
            result["analysis_mode"] = "fallback"
            result["recommendations"].append("Both Ollama and local engine unavailable")
            result["recommendations"].append("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")

        return result