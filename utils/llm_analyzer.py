"""
Regulator-safe LLM Compliance Analyzer (Ollama)
LLM is used ONLY to extract factual compliance indicators.
All scoring and risk classification is deterministic.
"""

import os
import json
import re
import time
import logging
import requests
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

@dataclass
class OllamaConfig:
    host: str = None
    model: str = None
    timeout: int = None
    temperature: float = None
    max_tokens: int = None

    def __post_init__(self):
        self.host = self.host or os.getenv("OLLAMA_HOST") or "http://host.docker.internal:11434"
        self.model = self.model or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = self.timeout or int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.temperature = self.temperature or float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.max_tokens = self.max_tokens or int(os.getenv("LLM_MAX_TOKENS", "2000"))


# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.available = self._check()

    def _check(self) -> bool:
        try:
            r = requests.get(f"{self.config.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, system: str) -> str:
        if not self.available:
            return ""

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens
            }
        }

        try:
            r = requests.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            if r.status_code == 200:
                return r.json().get("response", "")
        except Exception as e:
            logger.error("Ollama generation failed", exc_info=True)

        return ""


# ============================================================================
# INDICATOR MODEL
# ============================================================================

@dataclass
class RiskIndicator:
    indicator_id: str
    category: str
    description: str
    evidence: str
    confidence: float
    detected_at: str


# ============================================================================
# SAFE LLM PROMPTS
# ============================================================================

class IndicatorPrompts:

    @staticmethod
    def system() -> str:
        return """
You are a compliance analysis assistant.
Your role is strictly limited to extracting factual compliance indicators.

RULES:
- Do NOT assign risk levels
- Do NOT assign compliance scores
- Do NOT state legal violations
- Do NOT recommend actions
"""

    @staticmethod
    def extract(text: str) -> str:
        return f"""
Extract potential compliance indicators from the document below.

DOCUMENT:
{text[:4000]}

Return ONLY valid JSON:

{{
  "indicators": [
    {{
      "indicator_id": "<string>",
      "category": "<GDPR|AML|KYC|CSSF|MIFID|GENERAL>",
      "description": "<what was detected or missing>",
      "evidence": "<exact phrase or absence>",
      "confidence": <0.0-1.0>
    }}
  ]
}}
"""


# ============================================================================
# LLM INDICATOR EXTRACTOR
# ============================================================================

class LLMIndicatorExtractor:

    def __init__(self, client: OllamaClient):
        self.client = client

    def extract(self, text: str) -> List[RiskIndicator]:
        if not self.client.available:
            return []

        response = self.client.generate(
            IndicatorPrompts.extract(text),
            IndicatorPrompts.system()
        )

        try:
            match = re.search(r"\{.*\}", response, re.S)
            if not match:
                return []

            data = json.loads(match.group())
            indicators = []

            for i in data.get("indicators", []):
                indicators.append(
                    RiskIndicator(
                        indicator_id=i.get("indicator_id", "UNKNOWN"),
                        category=i.get("category", "GENERAL"),
                        description=i.get("description", ""),
                        evidence=i.get("evidence", ""),
                        confidence=float(i.get("confidence", 0.5)),
                        detected_at=datetime.utcnow().isoformat()
                    )
                )

            return indicators

        except Exception:
            logger.error("Failed to parse LLM indicators", exc_info=True)
            return []


# ============================================================================
# DETERMINISTIC SCORING ENGINE
# ============================================================================

class IndicatorScoringEngine:
    """
    Converts indicators into a numeric risk score.
    """

    WEIGHTS = {
        "MISSING_KYC": 45,
        "AML_PROCESS_MISSING": 50,
        "SANCTIONS_REFERENCE": 70,
        "GDPR_REFERENCE": 10,
        "DATA_RETENTION_UNCLEAR": 25,
        "AMBIGUOUS_LANGUAGE": 15
    }

    def score(self, indicators: List[RiskIndicator]) -> Dict[str, Any]:
        total = 0.0
        details = []

        for ind in indicators:
            weight = self.WEIGHTS.get(ind.indicator_id, 10)
            s = weight * ind.confidence
            total += s

            details.append({
                "indicator_id": ind.indicator_id,
                "category": ind.category,
                "score": round(s, 2),
                "confidence": ind.confidence,
                "evidence": ind.evidence
            })

        return {
            "risk_score": min(100.0, round(total, 2)),
            "score_details": details
        }


# ============================================================================
# RISK CLASSIFICATION (NO LEGAL CLAIMS)
# ============================================================================

class RiskClassifier:

    def classify(self, score: float) -> Dict[str, Any]:
        if score >= 80:
            return {"level": "HIGH", "manual_review_required": True}
        if score >= 50:
            return {"level": "MEDIUM", "manual_review_required": True}
        return {"level": "LOW", "manual_review_required": False}


# ============================================================================
# MAIN ANALYZER (REGULATOR SAFE)
# ============================================================================

class RegulatorSafeLLMAnalyzer:

    def __init__(self, config: OllamaConfig = None):
        self.client = OllamaClient(config)
        self.extractor = LLMIndicatorExtractor(self.client)
        self.scorer = IndicatorScoringEngine()
        self.classifier = RiskClassifier()

    def analyze(self, text: str) -> Dict[str, Any]:
        start = time.time()

        indicators = self.extractor.extract(text)
        scoring = self.scorer.score(indicators)
        assessment = self.classifier.classify(scoring["risk_score"])

        return {
            "indicators": [i.__dict__ for i in indicators],
            "risk_score": scoring["risk_score"],
            "risk_level": assessment["level"],
            "manual_review_required": assessment["manual_review_required"],
            "explainability": scoring["score_details"],
            "analysis_engine": "llm_indicator_based",
            "model_used": self.client.config.model if self.client.available else None,
            "analysis_duration": round(time.time() - start, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": (
                "This output provides compliance risk indicators only. "
                "It does not constitute a legal determination or regulatory decision."
            )
        }


# ============================================================================
# PUBLIC API
# ============================================================================

def analyze_with_local_llm(text: str, model: str = "llama3.2:3b") -> Dict[str, Any]:
    config = OllamaConfig(model=model)
    analyzer = RegulatorSafeLLMAnalyzer(config)
    return analyzer.analyze(text)


def check_ollama_status() -> Dict[str, Any]:
    client = OllamaClient()
    return {
        "installed": client.available,
        "running": client.available,
        "model": client.config.model,
        "host": client.config.host
    }


__all__ = [
    "RegulatorSafeLLMAnalyzer",
    "analyze_with_local_llm",
    "check_ollama_status"
]