# utils/llm_analyzer_ollama.py
"""
LLM-Powered Compliance Analyzer using Local Ollama
Replaces rule-based analysis with actual AI language model
"""

import logging
import time
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

# ============================================================================
# OLLAMA CONFIGURATION
# ============================================================================

@dataclass
class OllamaConfig:
    """Configuration for Ollama local LLM"""
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"  # or "mistral", "phi3", etc.
    timeout: int = 120
    temperature: float = 0.1  # Low for consistency
    max_tokens: int = 2000

    # Alternative models you can use:
    # - llama3.2:3b (fast, 3GB RAM)
    # - llama3:8b (better, 8GB RAM)
    # - mistral:7b (good balance)
    # - phi3:mini (very fast, 2GB RAM)
    # - mixtral:8x7b (best quality, 26GB RAM)

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

class OllamaClient:
    """Client for local Ollama LLM"""

    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.available = self._check_availability()

        if self.available:
            logger.info(f"✅ Ollama connected: {self.config.model}")
        else:
            logger.warning("⚠️ Ollama not available - using fallback")

    def _check_availability(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from Ollama"""
        try:
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return ""

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.config.host}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []

# ============================================================================
# COMPLIANCE PROMPTS
# ============================================================================

class CompliancePrompts:
    """Structured prompts for compliance analysis"""

    @staticmethod
    def get_system_prompt() -> str:
        """System prompt for compliance analysis"""
        return """You are a Luxembourg financial compliance expert specializing in:
- GDPR (General Data Protection Regulation)
- AML/KYC (Anti-Money Laundering / Know Your Customer)
- CSSF regulations (Luxembourg financial regulator)
- MiFID II (Markets in Financial Instruments Directive)
- Banking compliance and risk management

Your task is to analyze documents for regulatory compliance and provide:
1. Compliance score (0-100)
2. Identified issues with severity levels
3. Specific regulatory violations
4. Actionable recommendations

Be precise, professional, and reference specific regulations."""

    @staticmethod
    def get_analysis_prompt(text: str, doc_type: str, language: str) -> str:
        """Prompt for document analysis"""
        return f"""Analyze this {doc_type} document for regulatory compliance.

DOCUMENT TEXT:
{text[:4000]}  # Limit for context window

ANALYSIS REQUIRED:
1. Overall compliance score (0-100)
2. Document type classification
3. Language: {language}
4. Compliance issues found (with severity: critical/high/medium/low)
5. Specific regulation violations (GDPR, AML, CSSF, etc.)
6. Risk assessment
7. Recommendations for improvement

FORMAT YOUR RESPONSE AS JSON:
{{
  "score": <number 0-100>,
  "document_type": "<type>",
  "language": "<language>",
  "issues": [
    {{
      "rule_id": "<regulation_code>",
      "description": "<issue description>",
      "severity": "<critical|high|medium|low>",
      "regulatory_basis": "<specific regulation>",
      "suggested_action": "<recommendation>",
      "confidence": <0.0-1.0>
    }}
  ],
  "risk_level": "<critical|high|medium|low>",
  "recommendations": ["<recommendation 1>", "<recommendation 2>"],
  "strengths": ["<strength 1>", "<strength 2>"],
  "luxembourg_specific": <true|false>,
  "banking_sector": <true|false>
}}

Provide ONLY the JSON response, no additional text."""

    @staticmethod
    def get_quick_score_prompt(text: str) -> str:
        """Quick scoring prompt for faster analysis"""
        return f"""Rate this document's regulatory compliance on a scale of 0-100.

Document excerpt:
{text[:1000]}

Consider:
- GDPR compliance
- AML/KYC procedures
- Banking regulations
- Documentation quality

Respond with ONLY a number between 0-100."""

# ============================================================================
# LLM-POWERED COMPLIANCE ANALYZER
# ============================================================================

class LLMComplianceAnalyzer:
    """Compliance analyzer using local LLM"""

    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.client = OllamaClient(self.config)
        self.prompts = CompliancePrompts()

        if not self.client.available:
            logger.warning("⚠️ LLM not available, falling back to rules")

    def analyze_document(self, text: str, doc_type: str = "auto",
                         language: str = "auto") -> Dict[str, Any]:
        """
        Main analysis function using LLM
        """
        start_time = time.time()

        if not self.client.available:
            return self._fallback_analysis(text, doc_type, language)

        logger.info(f"🤖 Analyzing with {self.config.model}")

        # Get system and analysis prompts
        system_prompt = self.prompts.get_system_prompt()
        analysis_prompt = self.prompts.get_analysis_prompt(text, doc_type, language)

        # Generate LLM response
        response = self.client.generate(analysis_prompt, system_prompt)

        # Parse LLM response
        result = self._parse_llm_response(response, text, doc_type, language)

        # Add metadata
        result.update({
            'analysis_duration': round(time.time() - start_time, 2),
            'analysis_engine': 'ollama_llm',
            'model_used': self.config.model,
            'llm_powered': True,
            'timestamp': time.time()
        })

        logger.info(f"✅ LLM analysis complete: {result.get('score', 0):.1f}%")

        return result

    def _parse_llm_response(self, response: str, text: str,
                            doc_type: str, language: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # Ensure required fields
                result.setdefault('score', 0.0)
                result.setdefault('issues', [])
                result.setdefault('recommendations', [])
                result.setdefault('document_type', doc_type)
                result.setdefault('language', language)

                # Calculate metrics
                result['total_issues'] = len(result['issues'])
                result['critical_issues'] = len([i for i in result['issues']
                                                 if i.get('severity') == 'critical'])
                result['high_issues'] = len([i for i in result['issues']
                                             if i.get('severity') == 'high'])

                # Generate assessment
                result['overall_assessment'] = self._generate_assessment(result)

                return result
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {response[:500]}")

            # Fallback: extract score manually
            score = self._extract_score_fallback(response, text)
            return {
                'score': score,
                'issues': [],
                'recommendations': ['Review LLM response manually'],
                'document_type': doc_type,
                'language': language,
                'overall_assessment': f"Partial analysis (score: {score:.1f}%)",
                'parse_error': str(e),
                'raw_response': response[:500]
            }

    def _extract_score_fallback(self, response: str, text: str) -> float:
        """Extract score when JSON parsing fails"""
        # Try to find any number that looks like a score
        numbers = re.findall(r'\b(\d{1,3}(?:\.\d{1,2})?)\b', response)
        for num in numbers:
            score = float(num)
            if 0 <= score <= 100:
                return score

        # Ultimate fallback: use rule-based quick estimate
        return self._quick_rule_score(text)

    def _quick_rule_score(self, text: str) -> float:
        """Quick rule-based score estimation"""
        text_lower = text.lower()
        score = 70.0  # Base score

        # Positive indicators
        positive_keywords = [
            'compliance', 'regulation', 'gdpr', 'kyc', 'aml',
            'verification', 'monitoring', 'conformité'
        ]
        for keyword in positive_keywords:
            if keyword in text_lower:
                score += 3.0

        # Negative indicators
        negative_keywords = [
            'non-compliant', 'violation', 'breach', 'manquant',
            'insuffisant', 'incomplete'
        ]
        for keyword in negative_keywords:
            if keyword in text_lower:
                score -= 5.0

        return max(0.0, min(100.0, score))

    def _generate_assessment(self, result: Dict) -> str:
        """Generate human-readable assessment"""
        score = result.get('score', 0)
        critical = result.get('critical_issues', 0)
        high = result.get('high_issues', 0)

        if score >= 95:
            return f"✅ Excellent compliance ({score:.1f}%) - {len(result.get('issues', []))} minor issue(s)"
        elif score >= 85:
            return f"👍 Good compliance ({score:.1f}%) - {high} high priority issue(s)"
        elif score >= 70:
            return f"⚠️ Adequate compliance ({score:.1f}%) - {high + critical} significant issue(s)"
        elif score >= 50:
            return f"⚠️ Poor compliance ({score:.1f}%) - {critical} critical issue(s)"
        else:
            return f"❌ Critical non-compliance ({score:.1f}%) - Immediate action required"

    def _fallback_analysis(self, text: str, doc_type: str, language: str) -> Dict[str, Any]:
        """Fallback when LLM is not available"""
        logger.info("📋 Using rule-based fallback")

        # Simple rule-based analysis
        score = self._quick_rule_score(text)

        return {
            'score': score,
            'issues': [],
            'recommendations': ['Install Ollama for AI-powered analysis'],
            'document_type': doc_type,
            'language': language,
            'overall_assessment': f"Basic analysis ({score:.1f}%) - LLM not available",
            'analysis_engine': 'rule_based_fallback',
            'llm_powered': False
        }

    def quick_score(self, text: str) -> float:
        """Get quick compliance score"""
        if not self.client.available:
            return self._quick_rule_score(text)

        prompt = self.prompts.get_quick_score_prompt(text)
        response = self.client.generate(prompt)

        # Extract number
        numbers = re.findall(r'\b(\d{1,3}(?:\.\d{1,2})?)\b', response)
        if numbers:
            return float(numbers[0])
        return self._quick_rule_score(text)

# ============================================================================
# HYBRID ANALYZER (LLM + Rules)
# ============================================================================

class HybridComplianceAnalyzer:
    """Combines LLM intelligence with rule-based validation"""

    def __init__(self, config: OllamaConfig = None):
        self.llm_analyzer = LLMComplianceAnalyzer(config)
        self.use_llm = self.llm_analyzer.client.available

        # Load rule-based data if available
        try:
            from utils.data_manager import ComplianceDataManager
            self.data_manager = ComplianceDataManager()
            self.has_rules = True
        except:
            self.data_manager = None
            self.has_rules = False

    def analyze(self, text: str, doc_type: str = "auto",
                language: str = "auto", mode: str = "hybrid") -> Dict[str, Any]:
        """
        Analyze with LLM and validate with rules

        Modes:
        - "llm_only": Use only LLM
        - "rules_only": Use only rules
        - "hybrid": LLM + rule validation (best)
        """

        if mode == "llm_only" or (mode == "hybrid" and self.use_llm):
            # Get LLM analysis
            llm_result = self.llm_analyzer.analyze_document(text, doc_type, language)

            if mode == "hybrid" and self.has_rules:
                # Validate with rules
                llm_result = self._validate_with_rules(llm_result, text)
                llm_result['analysis_mode'] = 'hybrid'
            else:
                llm_result['analysis_mode'] = 'llm_only'

            return llm_result

        else:
            # Fallback to rules
            return self._rule_based_analysis(text, doc_type, language)

    def _validate_with_rules(self, llm_result: Dict, text: str) -> Dict[str, Any]:
        """Validate LLM findings with rule-based checks"""

        # Add rule-based validation
        text_lower = text.lower()

        # Check for critical regulatory keywords
        critical_keywords = {
            'gdpr': ['gdpr', 'rgpd', 'data protection', 'protection données'],
            'aml': ['aml', 'kyc', 'anti-blanchiment', 'know your customer'],
            'mifid': ['mifid', 'marchés financiers', 'financial instruments']
        }

        validation_notes = []
        for category, keywords in critical_keywords.items():
            if any(kw in text_lower for kw in keywords):
                validation_notes.append(f"✓ {category.upper()} references found")

        llm_result['validation_notes'] = validation_notes
        llm_result['rules_validated'] = True

        return llm_result

    def _rule_based_analysis(self, text: str, doc_type: str, language: str) -> Dict[str, Any]:
        """Pure rule-based analysis fallback"""

        if self.has_rules and self.data_manager:
            # Use comprehensive rule-based analyzer
            from utils.llm_analyzer import AdvancedComplianceAnalyzer
            analyzer = AdvancedComplianceAnalyzer(self.data_manager)
            result = analyzer.analyze_document_comprehensive(text, doc_type, language)
            result['analysis_mode'] = 'rules_only'
            return result
        else:
            # Simple fallback
            return self.llm_analyzer._fallback_analysis(text, doc_type, language)

# ============================================================================
# PUBLIC API
# ============================================================================

def analyze_with_local_llm(text: str, doc_type: str = "auto",
                           language: str = "auto",
                           model: str = "llama3.2:3b",
                           mode: str = "hybrid") -> Dict[str, Any]:
    """
    Main entry point for LLM-powered compliance analysis

    Args:
        text: Document text to analyze
        doc_type: Type of document (auto-detect if "auto")
        language: Document language (auto-detect if "auto")
        model: Ollama model to use
        mode: "llm_only", "rules_only", or "hybrid"

    Returns:
        Analysis results dictionary
    """
    config = OllamaConfig(model=model)
    analyzer = HybridComplianceAnalyzer(config)
    return analyzer.analyze(text, doc_type, language, mode)

def check_ollama_status() -> Dict[str, Any]:
    """Check if Ollama is installed and running"""
    client = OllamaClient()

    if client.available:
        models = client.list_models()
        return {
            "installed": True,
            "running": True,
            "models": models,
            "recommended_model": "llama3.2:3b" if "llama3.2:3b" in models else models[0] if models else None,
            "host": client.config.host
        }
    else:
        return {
            "installed": False,
            "running": False,
            "models": [],
            "installation_required": True,
            "install_command": "curl -fsSL https://ollama.com/install.sh | sh"
        }

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'LLMComplianceAnalyzer',
    'HybridComplianceAnalyzer',
    'OllamaClient',
    'OllamaConfig',
    'analyze_with_local_llm',
    'check_ollama_status'
]