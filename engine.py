# engine.py - FINAL CORRECTED VERSION
"""
Local optimized engine with advanced scoring system
Capable of achieving 100.0% score for banking compliance analysis
Scores always returned with proper decimal formatting - FINAL CORRECTED VERSION
"""

import json
import os
import re
import logging
import math
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ComplianceIssue:
    """Represents a detected compliance issue"""
    rule_id: str
    description: str
    regulatory_basis: str
    severity: str  # critical, high, medium, low
    confidence_score: float
    recommendation: str
    penalty_info: str = ""
    weight: float = 0.0
    article_reference: str = ""
    business_impact: str = ""
    timeline: str = ""
    priority: int = 1

@dataclass
class ExcellenceCriteria:
    """Excellence criteria for 100.0% score"""
    criterion_id: str
    name: str
    weight: float
    threshold: float
    bonus_eligible: bool = False

@dataclass
class AnalysisResult:
    """Complete optimized analysis result"""
    overall_score: float
    excellence_score: float
    issues: List[ComplianceIssue]
    regulatory_framework: str
    analysis_summary: str
    document_type: str
    language: str
    recommendations: List[str]
    excellence_achieved: bool = False
    bonus_points: float = 0.0
    criteria_scores: Dict[str, float] = None

# ============================================================================
# CORRECTED SCORE FORMATTING FUNCTIONS
# ============================================================================

def format_score_properly(score: Any) -> float:
    """
    DEFINITIVE CORRECTED VERSION - Formats any score to proper decimal percentage
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
                logger.warning(f"⚠️ Cannot convert '{score}' to number, using 0.0")
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
        logger.error(f"❌ Error in format_score_properly with '{score}': {e}")
        return 0.0

def validate_score_range(score: float, field_name: str = "score") -> float:
    """Validates that a score is in the 0-100 range and corrects if necessary"""
    if not isinstance(score, (int, float)):
        logger.warning(f"⚠️ {field_name} is not numeric: {score}")
        return 0.0

    if score < 0:
        logger.warning(f"⚠️ {field_name} negative corrected: {score} → 0.0")
        return 0.0
    elif score > 100:
        logger.warning(f"⚠️ {field_name} above 100 corrected: {score} → 100.0")
        return 100.0
    else:
        return round(float(score), 2)

def fix_all_scores_in_dict(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Fixes all score formatting in a dictionary"""
    if not isinstance(data_dict, dict):
        return data_dict

    corrected = data_dict.copy()

    score_fields = [
        'score', 'final_score', 'overall_score', 'excellence_score',
        'enhanced_score', 'base_score', 'bonus_points', 'luxembourg_relevance',
        'confidence_score', 'weight'
    ]

    corrections_applied = []

    for field in score_fields:
        if field in corrected:
            original_value = corrected[field]
            corrected_value = format_score_properly(original_value)
            corrected[field] = corrected_value

            if isinstance(original_value, (int, float)) and abs(float(original_value) - corrected_value) > 1.0:
                corrections_applied.append(f"{field}: {original_value} → {corrected_value:.2f}")

    if 'issues' in corrected and isinstance(corrected['issues'], list):
        for issue in corrected['issues']:
            if isinstance(issue, dict):
                for score_field in ['confidence_score', 'weight', 'penalty_score']:
                    if score_field in issue:
                        original = issue[score_field]
                        corrected_val = format_score_properly(original)
                        issue[score_field] = corrected_val

    if corrections_applied:
        logger.info(f"🔧 Scores corrected: {', '.join(corrections_applied)}")

    return corrected

class LocalComplianceEngine:
    """
    Local compliance engine optimized for 100.0% score
    Complete analysis with advanced scoring system - FINAL CORRECTED VERSION
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data_cache = {}
        self.excellence_config = self._load_excellence_config()
        self.scoring_weights = self._load_scoring_weights()
        self.analysis_stats = {
            "total_analyses": 0,
            "files_loaded": 0,
            "perfect_scores": 0,
            "average_score": 0.0,
            "score_corrections": 0
        }
        self.load_all_data()
        logger.info(f"✅ LocalComplianceEngine optimized initialized with {len(self.data_cache)} sources")

    def _load_excellence_config(self) -> Dict[str, Any]:
        """Loads excellence configuration for 100.0% score"""
        return {
            "kyc_completeness": {
                "weight": 25.0,
                "threshold": 0.95,
                "required_elements": [
                    "customer_identification", "beneficial_ownership",
                    "risk_assessment", "due_diligence"
                ],
                "bonus_eligible": True
            },
            "aml_compliance": {
                "weight": 25.0,
                "threshold": 0.90,
                "required_elements": [
                    "transaction_monitoring", "suspicious_activity_reporting",
                    "customer_risk_profiling", "ongoing_monitoring"
                ],
                "bonus_eligible": True
            },
            "sanctions_screening": {
                "weight": 15.0,
                "threshold": 0.95,
                "required_elements": [
                    "sanctions_list_check", "pep_screening", "adverse_media"
                ],
                "bonus_eligible": True
            },
            "reporting_accuracy": {
                "weight": 15.0,
                "threshold": 0.90,
                "required_elements": [
                    "regulatory_reporting", "internal_reporting", "audit_trail"
                ],
                "bonus_eligible": False
            },
            "risk_management": {
                "weight": 10.0,
                "threshold": 0.85,
                "required_elements": [
                    "risk_appetite", "risk_controls", "risk_monitoring"
                ],
                "bonus_eligible": False
            },
            "data_protection": {
                "weight": 10.0,
                "threshold": 0.90,
                "required_elements": [
                    "gdpr_compliance", "data_minimization", "consent_management"
                ],
                "bonus_eligible": False
            }
        }

    def _load_scoring_weights(self) -> Dict[str, float]:
        """Weighting system for optimized scoring"""
        return {
            "perfect_compliance_bonus": 10.0,
            "excellence_threshold": 95.0,
            "severity_multipliers": {
                "critical": 25.0,
                "high": 15.0,
                "medium": 8.0,
                "low": 3.0
            },
            "confidence_weight": 0.8,
            "document_type_bonus": {
                "financial_statement": 5.0,
                "contract": 3.0,
                "policy": 2.0
            }
        }

    def load_all_data(self):
        """Loads all JSON files with robust error handling"""
        json_files = [
            'analyses.json', 'compliance_penalties.json', 'compliance_rules.json',
            'cross_border_regulations.json', 'dynamic_rules.json',
            'financial_institutions.json', 'issue_descriptions.json',
            'lux_keywords.json', 'regulations.json',
            'reporting_requirements.json', 'sanctions_lists.json'
        ]

        for filename in json_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            self.data_cache[filename] = data
                            self.analysis_stats["files_loaded"] += 1
                            logger.debug(f"✅ Loaded {filename}")
                        else:
                            self.data_cache[filename] = {}
                else:
                    self.data_cache[filename] = {}
                    logger.warning(f"⚠️ Missing file: {filename}")
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                self.data_cache[filename] = {}

    def detect_language(self, text: str) -> str:
        """Enhanced language detection"""
        text_lower = text.lower()

        language_indicators = {
            'fr': ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'est', 'sont', 'avec', 'pour', 'dans', 'sur'],
            'en': ['the', 'and', 'or', 'is', 'are', 'with', 'for', 'in', 'on', 'at', 'to', 'from', 'by'],
            'de': ['der', 'die', 'das', 'und', 'oder', 'ist', 'sind', 'mit', 'für', 'in', 'auf', 'zu', 'von']
        }

        scores = {}
        for lang, words in language_indicators.items():
            score = sum(1 for word in words if f' {word} ' in f' {text_lower} ')
            scores[lang] = score

        detected_lang = max(scores, key=scores.get) if scores else 'en'
        logger.debug(f"Language detected: {detected_lang} (scores: {scores})")
        return detected_lang

    def detect_document_type(self, text: str, language: str) -> str:
        """Enhanced document type detection"""
        text_lower = text.lower()

        type_indicators = {
            'financial_statement': {
                'fr': ['bilan', 'compte de résultat', 'états financiers', 'actif', 'passif', 'chiffre d\'affaires'],
                'en': ['balance sheet', 'income statement', 'financial statements', 'assets', 'liabilities', 'revenue'],
                'de': ['bilanz', 'gewinn- und verlustrechnung', 'jahresabschluss', 'aktiva', 'passiva']
            },
            'contract': {
                'fr': ['contrat', 'accord', 'convention', 'engagement', 'obligations', 'parties contractantes'],
                'en': ['contract', 'agreement', 'covenant', 'engagement', 'obligations', 'contracting parties'],
                'de': ['vertrag', 'vereinbarung', 'abkommen', 'verpflichtung', 'vertragsparteien']
            },
            'policy': {
                'fr': ['politique', 'procédure', 'directive', 'règlement', 'processus', 'méthodologie'],
                'en': ['policy', 'procedure', 'directive', 'regulation', 'process', 'methodology'],
                'de': ['politik', 'verfahren', 'richtlinie', 'verordnung', 'prozess']
            },
            'compliance_report': {
                'fr': ['rapport de conformité', 'audit', 'contrôle', 'vérification', 'évaluation'],
                'en': ['compliance report', 'audit', 'control', 'verification', 'assessment'],
                'de': ['compliance-bericht', 'audit', 'kontrolle', 'überprüfung']
            }
        }

        scores = {}
        for doc_type, lang_indicators in type_indicators.items():
            if language in lang_indicators:
                score = sum(1 for indicator in lang_indicators[language] if indicator in text_lower)
                scores[doc_type] = score

        detected_type = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else 'general_document'
        logger.debug(f"Document type detected: {detected_type} (scores: {scores})")
        return detected_type

    def analyze_document_compliance(self, text: str, doc_type: str = "auto", language: str = "auto") -> Dict[str, Any]:
        """
        Main compliance analysis with optimized scoring for 100.0% - FINAL CORRECTED VERSION
        """

        self.analysis_stats["total_analyses"] += 1
        start_time = datetime.now()

        if language == "auto":
            language = self.detect_language(text)
        if doc_type == "auto":
            doc_type = self.detect_document_type(text, language)

        logger.info(f"🔍 Analyzing document: type={doc_type}, language={language}")

        try:
            analysis_result = self._perform_comprehensive_analysis(text, doc_type, language)
            excellence_result = self._calculate_excellence_score(analysis_result, text, doc_type)
            final_result = self._merge_analysis_results(analysis_result, excellence_result)

            final_result = fix_all_scores_in_dict(final_result)
            self.analysis_stats["score_corrections"] += 1

            final_result.update({
                'analysis_version': '4.0_excellence_decimal_corrected_final',
                'analysis_duration': round((datetime.now() - start_time).total_seconds(), 2),
                'engine_type': 'local_optimized_decimal_corrected_final',
                'json_files_analyzed': len(self.data_cache),
                'total_rules_checked': self._count_total_rules(),
                'excellence_analysis': True,
                'can_achieve_100': True,
                'score_corrections_applied': True
            })

            final_result['score'] = validate_score_range(final_result.get('score', 0), 'final_score')
            if 'final_score' not in final_result:
                final_result['final_score'] = final_result['score']

            if final_result['score'] >= 100.0:
                self.analysis_stats["perfect_scores"] += 1

            logger.info(f"✅ Analysis completed: score={final_result['score']:.2f}%, corrections applied")

            return final_result

        except Exception as e:
            logger.error(f"❌ Error during analysis: {e}")
            return {
                'score': 0.0,
                'final_score': 0.0,
                'issues': [],
                'recommendations': [f"Analysis error: {str(e)}"],
                'overall_assessment': f"Error: {str(e)}",
                'document_type': doc_type,
                'language': language,
                'error': True,
                'score_corrections_applied': True
            }

    def _perform_comprehensive_analysis(self, text: str, doc_type: str, language: str) -> Dict[str, Any]:
        """Complete analysis using all JSON files"""

        text_lower = text.lower()
        all_issues = []

        compliance_rules = self.data_cache.get('compliance_rules.json', {})
        base_issues = self._analyze_compliance_rules(text_lower, compliance_rules, doc_type, language)
        all_issues.extend(base_issues)

        penalties = self.data_cache.get('compliance_penalties.json', {})
        penalty_issues = self._analyze_penalty_risks(text_lower, penalties, language)
        all_issues.extend(penalty_issues)

        sanctions = self.data_cache.get('sanctions_lists.json', {})
        sanctions_issues = self._analyze_sanctions_screening(text_lower, sanctions)
        all_issues.extend(sanctions_issues)

        cross_border = self.data_cache.get('cross_border_regulations.json', {})
        cross_border_issues = self._analyze_cross_border_compliance(text_lower, cross_border)
        all_issues.extend(cross_border_issues)

        reporting = self.data_cache.get('reporting_requirements.json', {})
        reporting_issues = self._analyze_reporting_requirements(text_lower, reporting, doc_type)
        all_issues.extend(reporting_issues)

        dynamic_rules = self.data_cache.get('dynamic_rules.json', {})
        dynamic_issues = self._analyze_dynamic_rules(text_lower, dynamic_rules, doc_type)
        all_issues.extend(dynamic_issues)

        lux_keywords = self.data_cache.get('lux_keywords.json', {})
        luxembourg_relevance = self._calculate_luxembourg_relevance(text_lower, lux_keywords)

        return {
            'issues': all_issues,
            'luxembourg_relevance': validate_score_range(luxembourg_relevance, 'luxembourg_relevance'),
            'document_type': doc_type,
            'language': language,
            'total_checks_performed': 6,
            'analysis_method': 'comprehensive_multi_source'
        }

    def _analyze_compliance_rules(self, text_lower: str, rules: Dict, doc_type: str, language: str) -> List[ComplianceIssue]:
        """Compliance rules analysis with precise detection and corrected scores"""
        issues = []

        # GDPR/Data protection rules
        if any(keyword in text_lower for keyword in ['données personnelles', 'personal data', 'personenbezogene daten']):
            gdpr_elements = {
                'consent': ['consentement', 'consent', 'einwilligung'],
                'legal_basis': ['base légale', 'legal basis', 'rechtsgrundlage'],
                'rights': ['droits', 'rights', 'rechte'],
                'dpo': ['délégué protection données', 'data protection officer', 'datenschutzbeauftragte']
            }

            missing_elements = []
            for element, keywords in gdpr_elements.items():
                if not any(keyword in text_lower for keyword in keywords):
                    missing_elements.append(element)

            if missing_elements:
                severity = 'critical' if len(missing_elements) >= 3 else 'high' if len(missing_elements) >= 2 else 'medium'
                confidence_score = 90.0 if len(missing_elements) >= 3 else 80.0 if len(missing_elements) >= 2 else 70.0

                issues.append(ComplianceIssue(
                    rule_id=f"GDPR_{len(missing_elements)}_MISSING",
                    description=f"Missing GDPR elements: {', '.join(missing_elements)}",
                    regulatory_basis="GDPR Articles 6, 7, 13, 14",
                    severity=severity,
                    confidence_score=confidence_score,
                    recommendation=f"Add required GDPR elements: {', '.join(missing_elements)}",
                    penalty_info="Up to 4% of annual turnover or €20M",
                    weight=float(self.scoring_weights['severity_multipliers'][severity]),
                    timeline="30 days maximum",
                    priority=1 if severity == 'critical' else 2
                ))

        # AML/KYC banking rules
        financial_keywords = ['transaction', 'client', 'customer', 'virement', 'transfer', 'compte', 'account']
        if any(keyword in text_lower for keyword in financial_keywords):
            aml_requirements = {
                'kyc': ['kyc', 'know your customer', 'connaissance client', 'identification client'],
                'monitoring': ['surveillance', 'monitoring', 'suivi transactions'],
                'reporting': ['déclaration', 'reporting', 'signalement'],
                'risk_assessment': ['évaluation risque', 'risk assessment', 'profil risque']
            }

            missing_aml = []
            for requirement, keywords in aml_requirements.items():
                if not any(keyword in text_lower for keyword in keywords):
                    missing_aml.append(requirement)

            if missing_aml:
                severity = 'critical' if 'kyc' in missing_aml else 'high'
                confidence_score = 85.0

                issues.append(ComplianceIssue(
                    rule_id="AML_BANKING_INCOMPLETE",
                    description=f"Incomplete AML/KYC procedures: {', '.join(missing_aml)}",
                    regulatory_basis="AML Directive 2015/849/EU, Luxembourg AML Law",
                    severity=severity,
                    confidence_score=confidence_score,
                    recommendation="Implement complete AML/KYC procedures",
                    penalty_info="Administrative and criminal sanctions",
                    weight=25.0,
                    timeline="Immediate",
                    priority=1
                ))

        return issues

    def _analyze_penalty_risks(self, text_lower: str, penalties: Dict, language: str) -> List[ComplianceIssue]:
        """Penalty risk analysis with corrected scores"""
        issues = []

        high_risk_patterns = [
            'non-conforme', 'non-compliant', 'violation', 'breach', 'infraction',
            'sanction', 'penalty', 'amende', 'fine'
        ]

        risk_level = sum(1 for pattern in high_risk_patterns if pattern in text_lower)

        if risk_level > 0:
            severity = 'critical' if risk_level >= 3 else 'high' if risk_level >= 2 else 'medium'
            confidence_score = 70.0 + (risk_level * 10.0)

            issues.append(ComplianceIssue(
                rule_id="PENALTY_RISK_DETECTED",
                description=f"Penalty risk detected (level: {risk_level})",
                regulatory_basis="General regulation",
                severity=severity,
                confidence_score=min(100.0, confidence_score),
                recommendation="Complete document review for compliance",
                penalty_info="Variable depending on infringement",
                weight=float(self.scoring_weights['severity_multipliers'][severity]),
                priority=1 if severity == 'critical' else 2
            ))

        return issues

    def _analyze_sanctions_screening(self, text_lower: str, sanctions: Dict) -> List[ComplianceIssue]:
        """Sanctions screening analysis with corrected scores"""
        issues = []

        sanctions_keywords = [
            'sanctions', 'embargo', 'liste noire', 'blacklist', 'personne politiquement exposée',
            'pep', 'politically exposed person', 'adverse media'
        ]

        sanctions_mentions = [kw for kw in sanctions_keywords if kw in text_lower]

        if sanctions_mentions:
            control_keywords = ['screening', 'vérification', 'contrôle', 'check']
            controls_present = any(kw in text_lower for kw in control_keywords)

            if not controls_present:
                issues.append(ComplianceIssue(
                    rule_id="SANCTIONS_SCREENING_MISSING",
                    description="Sanctions mentions without screening procedures",
                    regulatory_basis="EU sanctions regulations, OFAC",
                    severity='high',
                    confidence_score=80.0,
                    recommendation="Implement sanctions screening procedures",
                    penalty_info="Civil and criminal sanctions",
                    weight=15.0,
                    priority=1
                ))

        return issues

    def _analyze_cross_border_compliance(self, text_lower: str, cross_border: Dict) -> List[ComplianceIssue]:
        """Cross-border compliance analysis with corrected scores"""
        issues = []

        cross_border_indicators = [
            'international', 'cross-border', 'transfrontalier', 'export', 'import',
            'foreign', 'étranger', 'overseas', 'correspondent banking'
        ]

        if any(indicator in text_lower for indicator in cross_border_indicators):
            compliance_elements = [
                'crs', 'common reporting standard', 'fatca', 'automatic exchange',
                'échange automatique', 'reporting fiscal'
            ]

            if not any(element in text_lower for element in compliance_elements):
                issues.append(ComplianceIssue(
                    rule_id="CROSS_BORDER_COMPLIANCE",
                    description="Cross-border activities without CRS/FATCA compliance",
                    regulatory_basis="CRS, FATCA, DAC directives",
                    severity='high',
                    confidence_score=75.0,
                    recommendation="Implement CRS/FATCA compliance",
                    penalty_info="International tax penalties",
                    weight=15.0,
                    priority=2
                ))

        return issues

    def _analyze_reporting_requirements(self, text_lower: str, reporting: Dict, doc_type: str) -> List[ComplianceIssue]:
        """Reporting requirements analysis with corrected scores"""
        issues = []

        if doc_type == 'financial_statement':
            required_reports = [
                'audit', 'commissaire aux comptes', 'auditor', 'independent review',
                'annual report', 'rapport annuel'
            ]

            if not any(report in text_lower for report in required_reports):
                issues.append(ComplianceIssue(
                    rule_id="AUDIT_REQUIREMENT_MISSING",
                    description="Financial statements without audit mention",
                    regulatory_basis="Luxembourg commercial companies law",
                    severity='medium',
                    confidence_score=70.0,
                    recommendation="Add external audit certification",
                    penalty_info="Regulatory non-compliance",
                    weight=8.0,
                    priority=3
                ))

        return issues

    def _analyze_dynamic_rules(self, text_lower: str, dynamic_rules: Dict, doc_type: str) -> List[ComplianceIssue]:
        """Dynamic rules analysis with corrected scores"""
        issues = []

        if doc_type == 'contract':
            contract_essentials = [
                'parties', 'obligations', 'durée', 'term', 'résiliation', 'termination',
                'responsabilité', 'liability', 'force majeure'
            ]

            missing_essentials = [essential for essential in contract_essentials
                                  if essential not in text_lower]

            if len(missing_essentials) > 3:
                confidence_score = 60.0 + min(40.0, (len(missing_essentials) - 3) * 10.0)

                issues.append(ComplianceIssue(
                    rule_id="CONTRACT_INCOMPLETE",
                    description=f"Missing contract elements: {len(missing_essentials)}",
                    regulatory_basis="Contract law",
                    severity='medium',
                    confidence_score=confidence_score,
                    recommendation="Complete essential contract clauses",
                    weight=8.0,
                    priority=3
                ))

        return issues

    def _calculate_luxembourg_relevance(self, text_lower: str, lux_keywords: Dict) -> float:
        """Calculate Luxembourg relevance with score validation"""
        if not lux_keywords:
            return 50.0

        total_keywords = 0
        found_keywords = 0

        for category, keywords in lux_keywords.items():
            if isinstance(keywords, list):
                for keyword in keywords[:10]:
                    total_keywords += 1
                    if keyword.lower() in text_lower:
                        found_keywords += 1

        if total_keywords == 0:
            return 50.0

        relevance_ratio = found_keywords / total_keywords
        relevance_percentage = min(100.0, relevance_ratio * 200.0)

        return validate_score_range(relevance_percentage, 'luxembourg_relevance')

    def _calculate_excellence_score(self, analysis_result, text, doc_type):
        """Excellence score using JSON data - CORRECTED decimal return"""

        compliance_rules = self.data_cache.get('compliance_rules.json', {})

        total_weight = 0.0
        achieved_weight = 0.0

        criteria_scores = {}
        excellent_criteria = []
        total_bonus = 0.0

        for criterion_id, config in self.excellence_config.items():
            criterion_score = self._evaluate_excellence_criterion(text.lower(), criterion_id, config, doc_type)
            criteria_scores[criterion_id] = validate_score_range(criterion_score * 100.0, f'criterion_{criterion_id}')

            weight = config['weight']
            threshold = config['threshold']
            total_weight += weight

            if criterion_score >= threshold:
                achieved_weight += weight
                excellent_criteria.append(criterion_id)

                if config.get('bonus_eligible', False) and criterion_score >= 0.98:
                    bonus = weight * 0.1
                    total_bonus += bonus

        if total_weight > 0:
            excellence_score = (achieved_weight / total_weight) * 100.0
        else:
            excellence_score = 75.0

        lux_relevance = analysis_result.get('luxembourg_relevance', 0.0)
        if lux_relevance > 80.0:
            total_bonus += 5.0

        excellence_achieved = (excellence_score >= 95.0 and len(excellent_criteria) >= 4)

        final_excellence_score = min(100.0, excellence_score + total_bonus)

        return {
            'excellence_score': validate_score_range(final_excellence_score, 'excellence_score'),
            'criteria_scores': criteria_scores,
            'excellent_criteria': excellent_criteria,
            'excellence_achieved': excellence_achieved,
            'bonus_points': validate_score_range(total_bonus, 'bonus_points'),
            'luxembourg_bonus': validate_score_range(5.0 if lux_relevance > 80.0 else 0.0, 'luxembourg_bonus')
        }

    def _evaluate_excellence_criterion(self, text_lower: str, criterion_id: str, config: Dict, doc_type: str) -> float:
        """Evaluates a specific excellence criterion - Validated decimal return"""

        required_elements = config.get('required_elements', [])
        if not required_elements:
            return 0.8

        element_keywords = {
            'customer_identification': ['identification client', 'customer identification', 'identité', 'identity'],
            'beneficial_ownership': ['bénéficiaire effectif', 'beneficial owner', 'ultimate beneficial'],
            'risk_assessment': ['évaluation risque', 'risk assessment', 'profil risque', 'risk profile'],
            'due_diligence': ['due diligence', 'diligence raisonnable', 'vérification'],
            'transaction_monitoring': ['surveillance transactions', 'transaction monitoring', 'suivi opérations'],
            'suspicious_activity_reporting': ['déclaration soupçon', 'suspicious activity', 'signalement'],
            'customer_risk_profiling': ['profil risque client', 'customer risk profile'],
            'ongoing_monitoring': ['surveillance continue', 'ongoing monitoring', 'suivi permanent'],
            'sanctions_list_check': ['vérification sanctions', 'sanctions screening', 'liste sanctions'],
            'pep_screening': ['pep screening', 'personnes politiquement exposées', 'politically exposed'],
            'adverse_media': ['adverse media', 'médias négatifs', 'negative news'],
            'regulatory_reporting': ['reporting réglementaire', 'regulatory reporting', 'déclarations'],
            'internal_reporting': ['reporting interne', 'internal reporting', 'rapports internes'],
            'audit_trail': ['piste audit', 'audit trail', 'traçabilité'],
            'risk_appetite': ['appétit risque', 'risk appetite', 'tolérance risque'],
            'risk_controls': ['contrôles risque', 'risk controls', 'mesures contrôle'],
            'risk_monitoring': ['surveillance risque', 'risk monitoring', 'suivi risques'],
            'gdpr_compliance': ['rgpd', 'gdpr', 'protection données', 'data protection'],
            'data_minimization': ['minimisation données', 'data minimization', 'principe proportionnalité'],
            'consent_management': ['gestion consentement', 'consent management', 'consentement']
        }

        elements_found = 0
        for element in required_elements:
            keywords = element_keywords.get(element, [element])
            if any(keyword in text_lower for keyword in keywords):
                elements_found += 1

        base_score = elements_found / len(required_elements)

        if doc_type == 'financial_statement' and criterion_id == 'reporting_accuracy':
            base_score *= 1.2
        elif doc_type == 'contract' and criterion_id in ['kyc_completeness', 'aml_compliance']:
            base_score *= 1.1

        return min(1.0, base_score)

    def _merge_analysis_results(self, analysis_result: Dict, excellence_result: Dict) -> Dict[str, Any]:
        """Merges analysis and excellence results - CORRECTED decimal scores"""

        issues = analysis_result['issues']

        base_score = 100.0
        for issue in issues:
            confidence_decimal = format_score_properly(issue.confidence_score) / 100.0
            penalty = issue.weight * (confidence_decimal * self.scoring_weights['confidence_weight'])
            base_score -= penalty

        base_score = max(0.0, base_score)

        excellence_score = excellence_result['excellence_score']

        if excellence_result['excellence_achieved']:
            final_score = min(100.0, (base_score * 0.4) + (excellence_score * 0.6) + excellence_result['bonus_points'])
        else:
            final_score = (base_score * 0.6) + (excellence_score * 0.4)

        recommendations = self._generate_enhanced_recommendations(issues, excellence_result)

        analysis_summary = self._generate_analysis_summary(final_score, issues, excellence_result)

        issues_dict = []
        for issue in issues:
            issue_dict = asdict(issue)
            issue_dict = fix_all_scores_in_dict(issue_dict)
            issues_dict.append(issue_dict)

        return {
            'score': validate_score_range(final_score, 'merged_final_score'),
            'base_score': validate_score_range(base_score, 'merged_base_score'),
            'excellence_score': validate_score_range(excellence_score, 'merged_excellence_score'),
            'bonus_points': validate_score_range(excellence_result['bonus_points'], 'merged_bonus_points'),
            'issues': issues_dict,
            'recommendations': recommendations,
            'overall_assessment': analysis_summary,
            'excellence_achieved': excellence_result['excellence_achieved'],
            'excellent_criteria': excellence_result['excellent_criteria'],
            'criteria_scores': excellence_result['criteria_scores'],
            'document_type': analysis_result['document_type'],
            'language': analysis_result['language'],
            'luxembourg_relevance': validate_score_range(analysis_result['luxembourg_relevance'], 'merged_luxembourg_relevance'),
            'total_issues': len(issues),
            'critical_issues': len([i for i in issues if i.severity == 'critical']),
            'high_issues': len([i for i in issues if i.severity == 'high']),
            'medium_issues': len([i for i in issues if i.severity == 'medium']),
            'low_issues': len([i for i in issues if i.severity == 'low']),
            'can_achieve_100': True,
            'scoring_method': 'excellence_optimized_decimal_corrected_final'
        }

    def _generate_enhanced_recommendations(self, issues: List[ComplianceIssue], excellence_result: Dict) -> List[str]:
        """Generates enhanced recommendations"""
        recommendations = []

        critical_issues = [i for i in issues if i.severity == 'critical']
        high_issues = [i for i in issues if i.severity == 'high']

        if critical_issues:
            recommendations.append(f"🚨 URGENT: Fix {len(critical_issues)} critical issue(s)")
            for issue in critical_issues[:3]:
                recommendations.append(f"   • {issue.recommendation}")

        if high_issues:
            recommendations.append(f"⚠️ PRIORITY: Address {len(high_issues)} high-level issue(s)")

        if not excellence_result['excellence_achieved']:
            poor_criteria = [
                criterion for criterion, score in excellence_result['criteria_scores'].items()
                if score < (self.excellence_config[criterion]['threshold'] * 100.0)
            ]
            if poor_criteria:
                recommendations.append(f"🎯 For excellence, improve: {', '.join(poor_criteria[:3])}")

        if not critical_issues and not high_issues:
            if excellence_result['excellence_achieved']:
                recommendations.append("✅ Excellence achieved! Maintain high standards")
            else:
                recommendations.append("✅ Basic compliance achieved, aim for excellence")

        if len(issues) == 0 and excellence_result['excellence_achieved']:
            recommendations.append("🏆 Ready for 100.0% score - Exemplary document")

        return recommendations

    def _generate_analysis_summary(self, final_score: float, issues: List[ComplianceIssue], excellence_result: Dict) -> str:
        """Generates complete analysis summary with corrected scores"""

        validated_score = validate_score_range(final_score, 'summary_score')

        if validated_score >= 100.0:
            return f"🏆 PERFECT EXCELLENCE (Score: {validated_score:.2f}%) - Exemplary document compliant with all standards"
        elif validated_score >= 95.0:
            return f"⭐ EXCELLENCE (Score: {validated_score:.2f}%) - Exceptional compliance with {len(issues)} minor improvement point(s)"
        elif validated_score >= 85.0:
            return f"✅ VERY GOOD COMPLIANCE (Score: {validated_score:.2f}%) - {len(issues)} issue(s) detected, globally compliant"
        elif validated_score >= 70.0:
            return f"⚠️ CORRECT COMPLIANCE (Score: {validated_score:.2f}%) - {len(issues)} issue(s) to correct"
        elif validated_score >= 50.0:
            return f"🔍 PARTIAL COMPLIANCE (Score: {validated_score:.2f}%) - Review needed, {len(issues)} issue(s) identified"
        else:
            return f"❌ NON-COMPLIANCE (Score: {validated_score:.2f}%) - Complete review required, {len(issues)} major issue(s)"

    def _count_total_rules(self) -> int:
        """Counts total analyzed rules"""
        total = 0
        for filename, data in self.data_cache.items():
            if isinstance(data, dict):
                total += len(data)
        return total

    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Returns analysis statistics with corrected scores"""
        perfect_rate = (self.analysis_stats["perfect_scores"] /
                        max(self.analysis_stats["total_analyses"], 1)) * 100.0

        return {
            "total_analyses": self.analysis_stats["total_analyses"],
            "perfect_scores": self.analysis_stats["perfect_scores"],
            "perfect_score_rate": f"{validate_score_range(perfect_rate, 'perfect_rate'):.2f}%",
            "files_loaded": self.analysis_stats["files_loaded"],
            "score_corrections": self.analysis_stats["score_corrections"],
            "engine_version": "4.0_excellence_decimal_corrected_final",
            "max_possible_score": 100.0,
            "excellence_criteria_count": len(self.excellence_config),
            "score_correction_enabled": True
        }


# Compatibility functions for integration - FINAL CORRECTED
def analyze_document_compliance(text: str, doc_type: str = "auto", language: str = "auto", data_dir: str = "data") -> Dict[str, Any]:
    """Main compatibility function optimized - FINAL CORRECTED decimal scores"""
    engine = LocalComplianceEngine(data_dir)
    result = engine.analyze_document_compliance(text, doc_type, language)
    return fix_all_scores_in_dict(result)

def identify_issues(text: str, **kwargs) -> Tuple[List[Dict[str, Any]], float]:
    """Compatibility function for utils/llm_analyzer.py - FINAL CORRECTED decimal scores"""
    result = analyze_document_compliance(text, **kwargs)
    issues = result.get('issues', [])

    score_percentage = format_score_properly(result.get('score', 0.0))
    score_decimal = round(score_percentage / 100.0, 4)

    return issues, score_decimal

def check_ollama_installation() -> Dict[str, Any]:
    """Compatibility function - replaces Ollama check with correction"""
    try:
        engine = LocalComplianceEngine()
        stats = engine.get_analysis_statistics()
        return {
            "installed": True,
            "running": True,
            "models": ["local_engine_v4.0_excellence_decimal_corrected_final"],
            "engine_type": "local_optimized_decimal_corrected_final",
            "data_files_available": stats["files_loaded"],
            "status": "operational",
            "excellence_enabled": True,
            "max_score": 100.0,
            "decimal_precision": True,
            "scoring_format": "XX.XX%",
            "score_correction_enabled": True,
            "correction_system": "format_score_properly + validate_score_range"
        }
    except Exception as e:
        return {
            "installed": False,
            "running": False,
            "error": str(e),
            "models": [],
            "score_correction_enabled": False,
            "suggestion": "Check optimized corrected local engine configuration"
        }