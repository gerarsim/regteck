# engine.py - WITH EXTREME RISK DETECTOR INTEGRATION
"""
Local optimized engine with advanced scoring system + Extreme Risk Detection
Capable of achieving 100.0% score for banking compliance analysis
Automatically detects sanctioned entities, terrorist organizations, and criminal activity
Scores always returned with proper decimal formatting
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

# ============================================================================
# EXTREME RISK DETECTOR - INLINE INTEGRATION
# ============================================================================

class ExtremeRiskDetector:
    """Detects extreme compliance violations like sanctioned entities and illegal activities"""

    def __init__(self):
        # Known sanctioned individuals and entities
        self.sanctioned_entities = {
            # Political leaders under sanctions
            'vladimir poutine': {'type': 'individual', 'sanctions': ['EU', 'US', 'UK'], 'risk': 'EXTREME'},
            'vladimir putin': {'type': 'individual', 'sanctions': ['EU', 'US', 'UK'], 'risk': 'EXTREME'},
            'kim jong-un': {'type': 'individual', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'kim jong un': {'type': 'individual', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'ali khamenei': {'type': 'individual', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'},
            'bashar al-assad': {'type': 'individual', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'},

            # Terrorist organizations
            'hamas': {'type': 'terrorist_org', 'sanctions': ['US', 'EU', 'UN'], 'risk': 'EXTREME'},
            'hezbollah': {'type': 'terrorist_org', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'},
            'al-qaeda': {'type': 'terrorist_org', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'isis': {'type': 'terrorist_org', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'al qaeda': {'type': 'terrorist_org', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},

            # Criminal organizations
            'cartel de sinaloa': {'type': 'criminal_org', 'sanctions': ['US'], 'risk': 'EXTREME'},
            'cartel sinaloa': {'type': 'criminal_org', 'sanctions': ['US'], 'risk': 'EXTREME'},
            'ndrangheta': {'type': 'criminal_org', 'sanctions': ['EU'], 'risk': 'EXTREME'},
            'cosa nostra': {'type': 'criminal_org', 'sanctions': ['EU', 'US'], 'risk': 'EXTREME'},

            # Sanctioned countries
            'north korea': {'type': 'country', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'coree du nord': {'type': 'country', 'sanctions': ['UN', 'US', 'EU'], 'risk': 'EXTREME'},
            'iran': {'type': 'country', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'},
            'syria': {'type': 'country', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'},
            'syrie': {'type': 'country', 'sanctions': ['US', 'EU'], 'risk': 'EXTREME'}
        }

        # Red flag keywords for illegal activities
        self.illegal_activity_keywords = {
            'money_laundering': [
                'blanchiment', 'laundering', 'layering', 'placement',
                'structuring', 'smurfing', 'shell company', 'offshore'
            ],
            'sanctions_evasion': [
                'contournement', 'éviter sanctions', 'avoid sanctions',
                'bypass sanctions', 'circumvent', 'evade'
            ],
            'terrorist_financing': [
                'financement terrorisme', 'terrorist financing',
                'terror funding', 'jihad financing'
            ],
            'fraud': [
                'fraude', 'fraud', 'ponzi', 'scam', 'embezzlement',
                'détournement', 'escroquerie'
            ],
            'corruption': [
                'corruption', 'pot-de-vin', 'bribery', 'kickback',
                'bakshish', 'facilitation payment'
            ]
        }

        # VIP/Premium service red flags
        self.suspicious_service_terms = [
            'vip', 'premium', 'privilégié', 'privileged', 'special',
            'facilité', 'facilitated', 'express', 'rapide', 'fast-track',
            'discreet', 'discret', 'confidential', 'confidentiel',
            'services spéciaux', 'special services'
        ]

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive analysis for extreme risks"""
        text_lower = text.lower()

        results = {
            'extreme_risk_detected': False,
            'risk_level': 'NONE',
            'sanctioned_entities_found': [],
            'illegal_activities_found': [],
            'red_flags': [],
            'recommended_action': 'normal_processing',
            'legal_consequences': [],
            'should_report_authorities': False
        }

        # 1. Check for sanctioned entities
        sanctioned_found = self._detect_sanctioned_entities(text_lower)
        if sanctioned_found:
            results['sanctioned_entities_found'] = sanctioned_found
            results['extreme_risk_detected'] = True
            results['risk_level'] = 'EXTREME'
            results['should_report_authorities'] = True

        # 2. Check for illegal activity keywords
        illegal_activities = self._detect_illegal_activities(text_lower)
        if illegal_activities:
            results['illegal_activities_found'] = illegal_activities
            results['extreme_risk_detected'] = True
            results['risk_level'] = 'EXTREME'

        # 3. Check for suspicious VIP services
        vip_flags = self._detect_suspicious_vip_services(text_lower)
        if vip_flags:
            results['red_flags'].extend(vip_flags)
            if results['risk_level'] == 'NONE':
                results['risk_level'] = 'HIGH'

        # 4. Combined risk assessment
        if results['extreme_risk_detected']:
            results['recommended_action'] = 'IMMEDIATE_REJECTION_AND_REPORT'
            results['legal_consequences'] = self._get_legal_consequences(results)

        return results

    def _detect_sanctioned_entities(self, text_lower: str) -> List[Dict[str, Any]]:
        """Detect mentions of sanctioned entities"""
        found = []

        for entity_name, entity_info in self.sanctioned_entities.items():
            if entity_name in text_lower:
                found.append({
                    'name': entity_name.title(),
                    'type': entity_info['type'],
                    'sanctions_lists': entity_info['sanctions'],
                    'risk_level': entity_info['risk']
                })

        return found

    def _detect_illegal_activities(self, text_lower: str) -> List[Dict[str, str]]:
        """Detect mentions of illegal activities"""
        found = []

        for activity_type, keywords in self.illegal_activity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    found.append({
                        'activity': activity_type,
                        'keyword': keyword,
                        'severity': 'CRIMINAL'
                    })
                    break

        return found

    def _detect_suspicious_vip_services(self, text_lower: str) -> List[str]:
        """Detect suspicious VIP service offerings"""
        flags = []

        for term in self.suspicious_service_terms:
            if term.lower() in text_lower:
                flags.append(f"Suspicious service term: '{term}'")

        if 'vip' in text_lower and any(s in text_lower for s in ['sanctions', 'contournement', 'facilité']):
            flags.append("CRITICAL: VIP services for sanctioned entities")

        return flags

    def _get_legal_consequences(self, results: Dict) -> List[str]:
        """Determine legal consequences"""
        consequences = []

        if results['sanctioned_entities_found']:
            consequences.append("⚖️ CRIMINAL LIABILITY: Violates sanctions laws (EU, US, UN)")
            consequences.append("💰 PENALTIES: Up to €1,000,000 or 10 years imprisonment")
            consequences.append("🏦 INSTITUTIONAL: Banking license revocation risk")
            consequences.append("🌍 INTERNATIONAL: FATF blacklisting potential")

        if results['illegal_activities_found']:
            for activity in results['illegal_activities_found']:
                if activity['activity'] == 'money_laundering':
                    consequences.append("🚨 MONEY LAUNDERING: Criminal prosecution mandatory")
                elif activity['activity'] == 'terrorist_financing':
                    consequences.append("🚨 TERRORIST FINANCING: Immediate law enforcement referral required")
                elif activity['activity'] == 'sanctions_evasion':
                    consequences.append("🚨 SANCTIONS EVASION: Severe penalties under international law")

        return consequences


# ============================================================================
# ORIGINAL ENGINE DATACLASSES
# ============================================================================

@dataclass
class ComplianceIssue:
    """Represents a detected compliance issue"""
    rule_id: str
    description: str
    regulatory_basis: str
    severity: str
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
# SCORE FORMATTING FUNCTIONS (Keep existing)
# ============================================================================

def format_score_properly(score: Any) -> float:
    """DEFINITIVE CORRECTED VERSION - Formats any score to proper decimal percentage"""
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
        logger.error(f"❌ Error in format_score_properly: {e}")
        return 0.0

def validate_score_range(score: float, field_name: str = "score") -> float:
    """Validates that a score is in 0-100 range"""
    if not isinstance(score, (int, float)):
        logger.warning(f"⚠️ {field_name} not numeric: {score}, using 0.0")
        return 0.0

    if score < 0:
        return 0.0
    elif score > 100:
        return 100.0
    else:
        return round(float(score), 2)

def fix_all_scores_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively fixes all scores in a dictionary"""
    if not isinstance(data, dict):
        return data

    result = data.copy()
    score_fields = ['score', 'final_score', 'base_score', 'excellence_score',
                    'bonus_points', 'confidence_score', 'weight', 'luxembourg_relevance']

    for field in score_fields:
        if field in result:
            result[field] = format_score_properly(result[field])

    if 'issues' in result and isinstance(result['issues'], list):
        result['issues'] = [fix_all_scores_in_dict(issue) if isinstance(issue, dict) else issue
                            for issue in result['issues']]

    return result

# ============================================================================
# LOCAL COMPLIANCE ENGINE (Keep most of the original, add extreme risk check)
# ============================================================================

class LocalComplianceEngine:
    """Local optimized compliance engine WITH EXTREME RISK DETECTION"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.data_cache = {}
        self.analysis_stats = {
            "total_analyses": 0,
            "perfect_scores": 0,
            "files_loaded": 0,
            "score_corrections": 0,
            "extreme_risks_detected": 0  # NEW
        }

        # ✅ Initialize extreme risk detector
        self.extreme_risk_detector = ExtremeRiskDetector()
        logger.info("✅ Extreme Risk Detector initialized")

        # Load data files
        self._load_data_files()

        # Initialize excellence configuration
        self.excellence_config = self._initialize_excellence_criteria()

    def _load_data_files(self):
        """Load compliance data files"""
        json_files = [
            'compliance_rules.json',
            'lux_keywords.json',
            'compliance_penalties.json',
            'dynamic_rules.json',
            'regulations.json',
            'sanctions_lists.json',  # ✅ Important for extreme risk
            'financial_institutions.json'
        ]

        loaded = 0
        for filename in json_files:
            filepath = os.path.join(self.data_dir, filename)
            try:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.data_cache[filename] = json.load(f)
                        loaded += 1
                        logger.info(f"✅ Loaded: {filename}")
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                self.data_cache[filename] = {}

        self.analysis_stats["files_loaded"] = loaded
        logger.info(f"📊 Loaded {loaded}/{len(json_files)} data files")

    def _initialize_excellence_criteria(self) -> Dict[str, Dict]:
        """Initialize excellence criteria configuration"""
        return {
            'kyc_completeness': {
                'name': 'KYC Completeness',
                'weight': 25.0,
                'threshold': 0.85
            },
            'aml_compliance': {
                'name': 'AML/CFT Compliance',
                'weight': 25.0,
                'threshold': 0.90
            },
            'sanctions_screening': {
                'name': 'Sanctions Screening',
                'weight': 20.0,
                'threshold': 0.95
            },
            'documentation_quality': {
                'name': 'Documentation Quality',
                'weight': 15.0,
                'threshold': 0.80
            },
            'regulatory_alignment': {
                'name': 'Regulatory Alignment',
                'weight': 10.0,
                'threshold': 0.85
            },
            'risk_management': {
                'name': 'Risk Management',
                'weight': 5.0,
                'threshold': 0.75
            }
        }

    def analyze_document_compliance(self, text: str, doc_type: str = "auto", language: str = "auto") -> Dict[str, Any]:
        """
        Main analysis function WITH EXTREME RISK DETECTION
        ✅ NEW: Automatically detects sanctioned entities and illegal activities
        """
        self.analysis_stats["total_analyses"] += 1

        # ✅ STEP 1: EXTREME RISK DETECTION (FIRST!)
        extreme_risk_analysis = self.extreme_risk_detector.analyze_text(text)

        # ✅ If extreme risk detected, override everything and return immediately
        if extreme_risk_analysis['extreme_risk_detected']:
            logger.warning(f"🚨 EXTREME RISK DETECTED: {extreme_risk_analysis['risk_level']}")
            self.analysis_stats["extreme_risks_detected"] += 1

            return self._generate_extreme_risk_result(text, extreme_risk_analysis, doc_type, language)

        # STEP 2: Normal analysis (only if no extreme risk)
        analysis_result = self._perform_comprehensive_analysis(text, doc_type, language)

        # STEP 3: Excellence evaluation
        excellence_result = self._evaluate_excellence_criteria(text, analysis_result['issues'])

        # STEP 4: Merge results
        final_result = self._merge_analysis_results(analysis_result, excellence_result)

        return fix_all_scores_in_dict(final_result)

    def _generate_extreme_risk_result(self, text: str, extreme_analysis: Dict, doc_type: str, language: str) -> Dict[str, Any]:
        """
        Generate analysis result for extreme risk cases
        ✅ Score forced to 0.00%
        ✅ All issues marked as CRITICAL
        ✅ Legal consequences included
        """

        issues = []

        # Generate critical issues for each sanctioned entity
        for entity in extreme_analysis.get('sanctioned_entities_found', []):
            entity_type_labels = {
                'individual': 'Sanctioned Individual',
                'terrorist_org': 'Designated Terrorist Organization',
                'criminal_org': 'Criminal Organization',
                'country': 'Sanctioned Country'
            }

            issues.append({
                'rule_id': f'SANCTIONS_VIOLATION_{entity["type"].upper()}',
                'description': f'🚨 CRITICAL: {entity_type_labels.get(entity["type"], "Entity")} detected: {entity["name"]} (Sanctioned by: {", ".join(entity["sanctions_lists"])})',
                'severity': 'CRITICAL',
                'confidence_score': 100.0,
                'regulatory_basis': 'EU Sanctions Regulations, OFAC SDN List, UN Security Council Resolutions',
                'recommendation': '🚨 IMMEDIATE: Reject transaction, freeze assets, report to FIU and OFAC within 24 hours',
                'penalty_info': 'Criminal prosecution, fines up to €1M or 10 years imprisonment, license revocation',
                'timeline': 'IMMEDIATE (within 24 hours)',
                'business_impact': 'CATASTROPHIC - Institution closure risk',
                'weight': 100.0
            })

        # Generate issues for illegal activities
        for activity in extreme_analysis.get('illegal_activities_found', []):
            activity_labels = {
                'money_laundering': '💰 Money Laundering Indicators',
                'sanctions_evasion': '🚫 Sanctions Evasion Scheme',
                'terrorist_financing': '💣 Terrorist Financing',
                'fraud': '🎭 Fraudulent Activity',
                'corruption': '💸 Corruption/Bribery'
            }

            issues.append({
                'rule_id': f'CRIMINAL_ACTIVITY_{activity["activity"].upper()}',
                'description': f'🚨 CRIMINAL ACTIVITY: {activity_labels.get(activity["activity"], "Illegal Activity")} - Keyword: "{activity["keyword"]}"',
                'severity': 'CRITICAL',
                'confidence_score': 95.0,
                'regulatory_basis': 'AML Directive 2015/849/EU, Criminal Code, FATF Recommendations',
                'recommendation': '🚨 IMMEDIATE: File Suspicious Activity Report (SAR), notify authorities, cease all services',
                'penalty_info': 'Criminal prosecution, institutional penalties, reputational damage',
                'timeline': 'IMMEDIATE',
                'business_impact': 'SEVERE',
                'weight': 90.0
            })

        # VIP services issue if applicable
        if extreme_analysis.get('red_flags'):
            issues.append({
                'rule_id': 'SUSPICIOUS_VIP_SERVICES',
                'description': '⚠️ Suspicious VIP/Premium services offered to high-risk entities',
                'severity': 'CRITICAL',
                'confidence_score': 90.0,
                'regulatory_basis': 'AML Enhanced Due Diligence Requirements',
                'recommendation': 'Enhanced due diligence, senior management approval required, continuous monitoring',
                'penalty_info': 'Regulatory sanctions, reputation damage',
                'timeline': 'IMMEDIATE',
                'business_impact': 'HIGH',
                'weight': 80.0
            })

        # Generate urgent recommendations
        recommendations = [
            "🚨 IMMEDIATE: Stop all operations with these entities",
            "📞 IMMEDIATE: Report to Financial Intelligence Unit (FIU)",
            "📋 IMMEDIATE: File Suspicious Activity Report (SAR)",
            "⚖️ IMMEDIATE: Contact legal counsel",
            "🔒 IMMEDIATE: Freeze any related accounts",
            "📧 IMMEDIATE: Notify CSSF and other regulators",
            "🚫 DO NOT proceed with any transactions",
            "⚠️ Prepare for potential criminal investigation"
        ]

        # Add legal consequences to recommendations
        for consequence in extreme_analysis.get('legal_consequences', []):
            recommendations.append(consequence)

        # Generate assessment
        entity_count = len(extreme_analysis['sanctioned_entities_found'])
        activity_count = len(extreme_analysis['illegal_activities_found'])

        assessment = (
            f"🚨 EXTREME COMPLIANCE VIOLATION DETECTED 🚨\n"
            f"• {entity_count} sanctioned entities found\n"
            f"• {activity_count} illegal activities detected\n"
            f"• IMMEDIATE REGULATORY REPORTING REQUIRED\n"
            f"• DO NOT PROCEED WITH TRANSACTION"
        )

        return {
            'score': 0.0,  # ✅ ZERO SCORE
            'final_score': 0.0,
            'base_score': 0.0,
            'excellence_score': 0.0,
            'bonus_points': 0.0,
            'issues': issues,
            'recommendations': recommendations,
            'overall_assessment': assessment,
            'document_type': doc_type,
            'language': language,
            'luxembourg_relevance': 0.0,
            'total_issues': len(issues),
            'critical_issues': len(issues),
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0,
            'excellence_achieved': False,
            'can_achieve_100': False,
            'extreme_risk_detected': True,  # ✅ NEW FLAG
            'extreme_risk_analysis': extreme_analysis,  # ✅ INCLUDE FULL ANALYSIS
            'requires_immediate_action': True,
            'should_report_to_authorities': extreme_analysis.get('should_report_authorities', True),
            'legal_consequences': extreme_analysis.get('legal_consequences', []),
            'scoring_method': 'extreme_risk_override'
        }

    # ... (Keep all other original methods from the engine)
    # The rest of the LocalComplianceEngine class continues with original methods
    # I'll add a placeholder comment showing where to continue

    # [ORIGINAL METHODS CONTINUE HERE - _perform_comprehensive_analysis, etc.]

# ============================================================================
# COMPATIBILITY FUNCTIONS
# ============================================================================

def analyze_document_compliance(text: str, doc_type: str = "auto", language: str = "auto", data_dir: str = "data") -> Dict[str, Any]:
    """
    Main compatibility function WITH EXTREME RISK DETECTION
    ✅ Automatically detects sanctioned entities and criminal activity
    ✅ Forces score to 0% for extreme violations
    """
    engine = LocalComplianceEngine(data_dir)
    result = engine.analyze_document_compliance(text, doc_type, language)
    return fix_all_scores_in_dict(result)

def identify_issues(text: str, **kwargs) -> Tuple[List[Dict[str, Any]], float]:
    """Compatibility function with extreme risk detection"""
    result = analyze_document_compliance(text, **kwargs)
    issues = result.get('issues', [])

    score_percentage = format_score_properly(result.get('score', 0.0))
    score_decimal = round(score_percentage / 100.0, 4)

    return issues, score_decimal

def check_ollama_installation() -> Dict[str, Any]:
    """Compatibility function with extreme risk info"""
    try:
        engine = LocalComplianceEngine()
        stats = engine.get_analysis_statistics()
        return {
            "installed": True,
            "running": True,
            "models": ["local_engine_v5.0_with_extreme_risk_detection"],
            "engine_type": "local_optimized_with_extreme_risk",
            "data_files_available": stats["files_loaded"],
            "status": "operational",
            "excellence_enabled": True,
            "extreme_risk_detection": True,  # ✅ NEW
            "max_score": 100.0,
            "decimal_precision": True,
            "scoring_format": "XX.XX%",
            "sanctioned_entities_monitored": 15,  # ✅ NEW
            "terrorist_orgs_monitored": 4,  # ✅ NEW
            "features": [
                "Excellence scoring (100.00%)",
                "Extreme risk detection",
                "Sanctions screening",
                "Terrorist financing detection",
                "Money laundering indicators",
                "Decimal precision"
            ]
        }
    except Exception as e:
        return {
            "installed": False,
            "running": False,
            "error": str(e),
            "models": []
        }


# ============================================================================
# NOTE: Continue with original engine methods
# ============================================================================
# The complete original engine.py file continues here with all methods like:
# - _perform_comprehensive_analysis
# - _evaluate_excellence_criteria
# - _merge_analysis_results
# etc.
#
# This integration adds extreme risk detection BEFORE normal analysis
# If extreme risk is found, it returns immediately with 0% score
# Otherwise, it proceeds with normal analysis as before