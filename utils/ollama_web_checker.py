#!/usr/bin/env python3
"""
Ollama/Compliance Engine Health Checker
Supports both local engine and Ollama LLM
"""

import os
import sys
import json
import logging
import requests
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class OllamaHealthChecker:
    """Health checker for Ollama LLM service"""

    def __init__(self):
        """Initialize with configuration from environment"""
        self.host = os.environ.get('OLLAMA_HOST') or \
                    os.environ.get('OLLAMA_BASE_URL') or \
                    os.environ.get('OLLAMA_API_URL') or \
                    'http://localhost:11434'

        self.model = os.environ.get('OLLAMA_MODEL') or \
                     os.environ.get('LLM_MODEL') or \
                     'llama3.2:3b'

        self.timeout = int(os.environ.get('OLLAMA_TIMEOUT', '10'))

    def check_ollama_status(self) -> Dict[str, Any]:
        """
        Check Ollama service status
        Returns detailed status information
        """
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

        # Step 1: Check if host is reachable
        try:
            response = requests.get(f"{self.host}/", timeout=self.timeout)
            result["reachable"] = True
            logger.info(f"✅ Ollama host reachable: {self.host}")
        except requests.exceptions.ConnectionError as e:
            result["errors"].append(f"Cannot connect to Ollama at {self.host}")
            result["recommendations"].append(
                "Ensure Ollama is running: systemctl status ollama or ollama serve"
            )
            result["recommendations"].append(
                f"If running in Docker, use host IP (e.g., http://172.17.0.1:11434)"
            )
            result["status"] = "unreachable"
            return result
        except requests.exceptions.Timeout:
            result["errors"].append(f"Connection to {self.host} timed out")
            result["recommendations"].append("Check network connectivity and firewall rules")
            result["status"] = "timeout"
            return result
        except Exception as e:
            result["errors"].append(f"Unexpected error: {str(e)}")
            result["status"] = "error"
            return result

        # Step 2: Check API endpoint
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                result["api_responding"] = True
                data = response.json()
                models = data.get("models", [])
                result["models_available"] = [m.get("name", m.get("model", "unknown")) for m in models]
                logger.info(f"✅ Ollama API responding, {len(models)} models available")
            else:
                result["errors"].append(f"API returned status {response.status_code}")
                result["status"] = "api_error"
                return result
        except Exception as e:
            result["errors"].append(f"API check failed: {str(e)}")
            result["status"] = "api_error"
            return result

        # Step 3: Check if configured model is available
        if result["models_available"]:
            # Check exact match or partial match (e.g., "llama3.2:3b" matches "llama3.2:3b")
            model_found = any(
                self.model in m or m.startswith(self.model.split(':')[0])
                for m in result["models_available"]
            )
            if model_found:
                result["model_ready"] = True
                result["status"] = "operational"
                logger.info(f"✅ Configured model '{self.model}' is available")
            else:
                result["status"] = "model_missing"
                result["errors"].append(f"Configured model '{self.model}' not found")
                result["recommendations"].append(f"Pull the model: ollama pull {self.model}")
                if result["models_available"]:
                    result["recommendations"].append(
                        f"Available models: {', '.join(result['models_available'][:5])}"
                    )
        else:
            result["status"] = "no_models"
            result["errors"].append("No models installed in Ollama")
            result["recommendations"].append(f"Pull a model: ollama pull {self.model}")

        return result

    def test_generation(self, prompt: str = "Say 'OK' if you are working.") -> Dict[str, Any]:
        """Test actual generation capability"""
        result = {
            "success": False,
            "response": None,
            "latency_ms": None,
            "error": None
        }

        try:
            import time
            start = time.time()

            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 50
                    }
                },
                timeout=60
            )

            latency = (time.time() - start) * 1000
            result["latency_ms"] = round(latency, 2)

            if response.status_code == 200:
                data = response.json()
                result["success"] = True
                result["response"] = data.get("response", "")[:200]
                logger.info(f"✅ Generation test passed in {latency:.0f}ms")
            else:
                result["error"] = f"Generation failed with status {response.status_code}"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Generation test failed: {e}")

        return result


class EngineHealthChecker:
    """Health checker for local compliance engine with correct paths"""

    def __init__(self, base_dir: str = None):
        """
        Initialize with base directory
        If None, uses parent directory (project root)
        """
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # If in utils/, go up one level
            if os.path.basename(current_dir) == 'utils':
                base_dir = os.path.dirname(current_dir)
            else:
                base_dir = current_dir

        self.base_dir = base_dir
        self.data_dir = os.environ.get('DATA_DIR') or os.path.join(base_dir, 'data')

        self.required_files = [
            'analyses.json',
            'compliance_rules.json',
            'compliance_penalties.json',
            'cross_border_regulations.json',
            'dynamic_rules.json',
            'financial_institutions.json',
            'issue_descriptions.json',
            'lux_keywords.json',
            'regulations.json',
            'reporting_requirements.json',
            'sanctions_lists.json'
        ]

    def check_engine_installation(self) -> Dict[str, Any]:
        """Check if compliance engine is properly installed and operational"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "installed": False,
            "running": False,
            "status": "unknown",
            "engine_type": "unknown",
            "models": [],
            "features": [],
            "data_files": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }

        # Check engine module
        engine_check = self._check_engine_module()
        result.update(engine_check)

        # Check data files
        data_check = self._check_data_files()
        result["data_files"] = data_check

        # Determine final status
        result = self._determine_status(result)

        return result

    def _check_engine_module(self) -> Dict[str, Any]:
        """Check if engine.py module is available and functional"""
        result = {
            "installed": False,
            "running": False,
            "engine_type": "none",
            "models": [],
            "features": []
        }

        try:
            if self.base_dir not in sys.path:
                sys.path.insert(0, self.base_dir)

            from engine import LocalComplianceEngine, analyze_document_compliance

            result["installed"] = True
            result["engine_type"] = "local_json_based"

            try:
                engine = LocalComplianceEngine(self.data_dir)
                stats = engine.get_analysis_statistics()

                result["running"] = True
                result["models"] = ["local_engine_v4.0"]
                result["features"] = [
                    "Rule-based analysis",
                    "Luxembourg-specific compliance",
                    "Multi-language support",
                    "Banking sector optimization"
                ]
                result["statistics"] = stats

            except Exception as e:
                result["running"] = False
                result["errors"] = [f"Engine initialization failed: {str(e)}"]
                logger.warning(f"Engine init error: {e}")

        except ImportError as e:
            result["installed"] = False
            result["errors"] = [f"Engine module not found: {str(e)}"]

        return result

    def _check_data_files(self) -> Dict[str, Any]:
        """Check data files in data/ subdirectory"""
        data_info = {
            "total_required": len(self.required_files),
            "found": 0,
            "missing": [],
            "valid": [],
            "invalid": [],
            "corrupted": [],
            "total_size_mb": 0.0,
            "details": {}
        }

        if not os.path.exists(self.data_dir):
            data_info["errors"] = [f"Data directory not found: {self.data_dir}"]
            for filename in self.required_files:
                data_info["missing"].append(filename)
            return data_info

        for filename in self.required_files:
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                data_info["found"] += 1
                is_valid, error, size_mb, records = self._validate_json(filepath)

                data_info["details"][filename] = {
                    "path": filepath,
                    "valid": is_valid,
                    "size_mb": round(size_mb, 2),
                    "records": records,
                    "error": error
                }

                if is_valid:
                    data_info["valid"].append(filename)
                    data_info["total_size_mb"] += size_mb
                elif error:
                    data_info["invalid"].append(filename)
                    data_info["corrupted"].append(filename)
            else:
                data_info["missing"].append(filename)
                data_info["details"][filename] = {
                    "path": filepath,
                    "valid": False,
                    "error": "File not found"
                }

        return data_info

    def _validate_json(self, filepath: str) -> Tuple[bool, str, float, int]:
        """Validate JSON file"""
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                record_count = len(data)
            elif isinstance(data, list):
                record_count = len(data)
            else:
                record_count = 1

            return True, None, size_mb, record_count

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}", 0.0, 0
        except Exception as e:
            return False, f"Read error: {str(e)}", 0.0, 0

    def _determine_status(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine overall system status"""
        data_files = result["data_files"]

        if data_files["total_required"] == 0:
            found_ratio = 0
            valid_ratio = 0
        else:
            found_ratio = data_files["found"] / data_files["total_required"]
            valid_ratio = len(data_files["valid"]) / data_files["total_required"]

        if not result["installed"]:
            result["status"] = "critical"
            result["warnings"].append("Engine module not installed")
        elif not result["running"]:
            result["status"] = "error"
            result["warnings"].append("Engine installed but not running")
        elif found_ratio < 0.5:
            result["status"] = "critical"
            result["warnings"].append("Less than 50% of required data files found")
        elif found_ratio < 1.0:
            result["status"] = "degraded"
            result["warnings"].append(f"{data_files['total_required'] - data_files['found']} data files missing")
        elif valid_ratio < 1.0:
            result["status"] = "warning"
            result["warnings"].append(f"{len(data_files['invalid'])} invalid data files")
        else:
            result["status"] = "operational"

        return result

    def get_compact_status(self) -> Dict[str, Any]:
        """Get compact status for UI display"""
        full_status = self.check_engine_installation()

        return {
            "installed": full_status["installed"],
            "running": full_status["running"],
            "status": full_status["status"],
            "engine_type": full_status["engine_type"],
            "models": full_status["models"],
            "features": full_status["features"],
            "data_files_ok": len(full_status["data_files"]["valid"]) == full_status["data_files"]["total_required"],
            "data_summary": f"{len(full_status['data_files']['valid'])}/{full_status['data_files']['total_required']} files valid"
        }


class CombinedHealthChecker:
    """Combined health checker for both Ollama and local engine"""

    def __init__(self, base_dir: str = None):
        self.ollama_checker = OllamaHealthChecker()
        self.engine_checker = EngineHealthChecker(base_dir)

    def check_all(self) -> Dict[str, Any]:
        """Check all components and return combined status"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "ollama": self.ollama_checker.check_ollama_status(),
            "local_engine": self.engine_checker.check_engine_installation(),
            "overall_status": "unknown",
            "analysis_mode": "fallback",
            "recommendations": []
        }

        # Determine best available mode
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


# ============================================================================
# PUBLIC API - Backward compatible functions
# ============================================================================

def check_ollama_installation(base_dir: str = None) -> Dict[str, Any]:
    """
    Check Ollama installation - returns compact status
    Backward compatible with old API
    """
    # Check if we should use Ollama or local engine
    use_llm = os.environ.get('USE_LLM_ANALYZER', 'false').lower() == 'true'

    if use_llm:
        checker = OllamaHealthChecker()
        ollama_status = checker.check_ollama_status()
        return {
            "installed": ollama_status["reachable"],
            "running": ollama_status["api_responding"],
            "status": ollama_status["status"],
            "engine_type": "ollama_llm",
            "models": ollama_status["models_available"],
            "features": ["LLM-powered analysis", "Natural language understanding"],
            "host": ollama_status["host"],
            "model_ready": ollama_status["model_ready"]
        }
    else:
        checker = EngineHealthChecker(base_dir)
        return checker.get_compact_status()


def check_compliance_engine_online(base_dir: str = None) -> Dict[str, Any]:
    """Full health check with detailed information"""
    checker = CombinedHealthChecker(base_dir)
    return checker.check_all()


def get_engine_status_summary(base_dir: str = None) -> str:
    """Get human-readable status summary"""
    combined = check_compliance_engine_online(base_dir)

    status_emoji = {
        "optimal": "✅",
        "operational": "✅",
        "llm_only": "🤖",
        "engine_only": "📋",
        "degraded": "⚠️",
        "warning": "⚠️",
        "error": "❌",
        "critical": "🚫",
        "unreachable": "🔌",
        "unknown": "❓"
    }

    summary = []
    summary.append("=" * 60)
    summary.append("LexAI Compliance Engine Status")
    summary.append("=" * 60)

    # Overall status
    emoji = status_emoji.get(combined["overall_status"], "❓")
    summary.append(f"\n{emoji} Overall: {combined['overall_status'].upper()}")
    summary.append(f"📊 Analysis Mode: {combined['analysis_mode']}")

    # Ollama status
    ollama = combined["ollama"]
    emoji = status_emoji.get(ollama["status"], "❓")
    summary.append(f"\n🤖 Ollama LLM:")
    summary.append(f"   Status: {emoji} {ollama['status']}")
    summary.append(f"   Host: {ollama['host']}")
    if ollama["models_available"]:
        summary.append(f"   Models: {', '.join(ollama['models_available'][:3])}")
    if ollama["errors"]:
        for err in ollama["errors"]:
            summary.append(f"   ❌ {err}")

    # Local engine status
    engine = combined["local_engine"]
    emoji = status_emoji.get(engine["status"], "❓")
    summary.append(f"\n📋 Local Engine:")
    summary.append(f"   Status: {emoji} {engine['status']}")
    df = engine["data_files"]
    summary.append(f"   Data Files: {len(df['valid'])}/{df['total_required']} valid")

    # Recommendations
    all_recs = combined.get("recommendations", [])
    all_recs.extend(ollama.get("recommendations", []))
    all_recs.extend(engine.get("recommendations", []))
    if all_recs:
        summary.append(f"\n💡 Recommendations:")
        for rec in all_recs[:5]:
            summary.append(f"   • {rec}")

    summary.append("\n" + "=" * 60)
    return "\n".join(summary)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'OllamaHealthChecker',
    'EngineHealthChecker',
    'CombinedHealthChecker',
    'check_ollama_installation',
    'check_compliance_engine_online',
    'get_engine_status_summary'
]


if __name__ == "__main__":
    """Run diagnostics when executed directly"""
    print(get_engine_status_summary())

    print("\n" + "=" * 60)
    print("Detailed JSON Output:")
    print("=" * 60)

    result = check_compliance_engine_online()
    print(json.dumps(result, indent=2, default=str))