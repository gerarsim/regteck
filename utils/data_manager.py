# utils/data_manager.py - PRODUCTION-READY VERSION WITH SECURITY
"""
LexAI Compliance Data Manager
Security-hardened with lazy loading, caching, validation, and Luxembourg-specific features

Version: 2.0.1-secure
Date: 2025-12-24
"""

import json
import os
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from functools import lru_cache
import re

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Security limits
MAX_JSON_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CACHE_AGE_MINUTES = 60  # Cache expires after 1 hour
MAX_SEARCH_RESULTS = 100  # Limit search results

# Data file categories
CORE_DATA_FILES = [
    'compliance_rules',
    'lux_keywords',
    'regulations',
    'sanctions_lists'
]

EXTENDED_DATA_FILES = [
    'analyses',
    'compliance_penalties',
    'cross_border_regulations',
    'dynamic_rules',
    'financial_institutions',
    'issue_descriptions',
    'reporting_requirements'
]

RUNTIME_DATA_FILES = [
    'users',
    'sessions',
    'translations'
]

# Luxembourg-specific
LUXEMBOURG_KEYWORDS_CATEGORIES = [
    'aml_kyc',
    'cssf_regulations',
    'banking',
    'insurance',
    'financial_markets',
    'data_protection',
    'cross_border',
    'tax',
    'legal_entities',
    'governance'
]

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Dict[str, Any]
    loaded_at: datetime
    file_path: str
    file_hash: str
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    def is_expired(self, max_age_minutes: int = MAX_CACHE_AGE_MINUTES) -> bool:
        """Check if cache entry is expired"""
        age = datetime.now() - self.loaded_at
        return age > timedelta(minutes=max_age_minutes)

    def access(self) -> None:
        """Record access"""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class SearchResult:
    """Search result with metadata"""
    item_id: str
    data: Dict[str, Any]
    match_score: float
    match_reason: str
    category: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'item_id': self.item_id,
            'data': self.data,
            'match_score': self.match_score,
            'match_reason': self.match_reason,
            'category': self.category
        }


@dataclass
class DataStatistics:
    """Statistics about loaded data"""
    file_name: str
    file_path: str
    record_count: int
    file_size_bytes: int
    last_loaded: datetime
    access_count: int
    has_data: bool
    cache_expired: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'file_name': self.file_name,
            'file_path': self.file_path,
            'record_count': self.record_count,
            'file_size_mb': round(self.file_size_bytes / (1024 * 1024), 2),
            'last_loaded': self.last_loaded.isoformat(),
            'access_count': self.access_count,
            'has_data': self.has_data,
            'cache_expired': cache_expired
        }


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def validate_json_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON file before loading

    Args:
        file_path: Path to JSON file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        path = Path(file_path)

        # Check exists
        if not path.exists():
            return False, f"File not found: {file_path}"

        # Check is file
        if not path.is_file():
            return False, f"Not a file: {file_path}"

        # Check size
        file_size = path.stat().st_size
        if file_size > MAX_JSON_FILE_SIZE:
            return False, f"File too large: {file_size / (1024*1024):.2f}MB > {MAX_JSON_FILE_SIZE / (1024*1024):.2f}MB"

        if file_size == 0:
            return False, "File is empty"

        # Check readable
        if not os.access(path, os.R_OK):
            return False, f"File not readable: {file_path}"

        return True, None

    except Exception as e:
        return False, f"Validation error: {e}"


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of file

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash
    """
    try:
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {e}")
        return ""


def sanitize_search_query(query: str, max_length: int = 200) -> str:
    """
    Sanitize search query to prevent injection attacks

    Args:
        query: Search query
        max_length: Maximum query length

    Returns:
        Sanitized query
    """
    # Remove control characters
    query = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', query)

    # Limit length
    query = query[:max_length]

    # Remove leading/trailing whitespace
    query = query.strip()

    return query


# ============================================================================
# MAIN DATA MANAGER CLASS
# ============================================================================

class ComplianceDataManager:
    """
    Production-ready compliance data manager

    Features:
    - Lazy loading (only loads when needed)
    - Intelligent caching with expiry
    - Security validation
    - File change detection
    - Performance optimized
    - Luxembourg-specific features
    """

    def __init__(self, auto_load: bool = False, cache_age_minutes: int = MAX_CACHE_AGE_MINUTES):
        """
        Initialize data manager

        Args:
            auto_load: Whether to load all data immediately
            cache_age_minutes: Cache expiry time in minutes
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_age_minutes = cache_age_minutes
        self._data_files = self._get_data_file_paths()

        # Statistics
        self._total_loads = 0
        self._total_refreshes = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(f"✅ ComplianceDataManager initialized (lazy loading: {not auto_load})")

        if auto_load:
            self.preload_all()

    def _get_data_file_paths(self) -> Dict[str, str]:
        """
        Get data file paths with fallback

        Returns:
            Dictionary mapping file keys to paths
        """
        try:
            from .config import config
            # Use config if available
            data_files = {}
            for key in config.DATA_FILES.keys():
                if key not in RUNTIME_DATA_FILES:  # Skip runtime files
                    data_files[key] = str(config.get_data_file_path(key))
            return data_files

        except ImportError:
            logger.warning("Config not available, using fallback paths")
            # Fallback - direct paths
            base_dir = Path(__file__).parent.parent / "data"
            return {
                'analyses': str(base_dir / 'analyses.json'),
                'compliance_penalties': str(base_dir / 'compliance_penalties.json'),
                'compliance_rules': str(base_dir / 'compliance_rules.json'),
                'cross_border_regulations': str(base_dir / 'cross_border_regulations.json'),
                'dynamic_rules': str(base_dir / 'dynamic_rules.json'),
                'financial_institutions': str(base_dir / 'financial_institutions.json'),
                'issue_descriptions': str(base_dir / 'issue_descriptions.json'),
                'lux_keywords': str(base_dir / 'lux_keywords.json'),
                'regulations': str(base_dir / 'regulations.json'),
                'reporting_requirements': str(base_dir / 'reporting_requirements.json'),
                'sanctions_lists': str(base_dir / 'sanctions_lists.json')
            }

    def _load_json_file(self, file_path: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Securely load JSON file with validation

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (data, error_message)
        """
        # Validate file
        is_valid, error_msg = validate_json_file(file_path)
        if not is_valid:
            return {}, error_msg

        # Load JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                if not content:
                    return {}, "File is empty"

                data = json.loads(content)

                if not isinstance(data, dict):
                    logger.warning(f"JSON data is not a dictionary: {type(data)}")
                    # Convert to dict if possible
                    if isinstance(data, list):
                        data = {f"item_{i}": item for i, item in enumerate(data)}
                    else:
                        data = {"data": data}

                return data, None

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {e}"
            logger.error(f"Error loading {file_path}: {error_msg}")
            return {}, error_msg

        except Exception as e:
            error_msg = f"Load error: {e}"
            logger.error(f"Error loading {file_path}: {error_msg}")
            return {}, error_msg

    def _get_cached_data(
            self,
            name: str,
            refresh: bool = False,
            check_file_changes: bool = True
    ) -> Dict[str, Any]:
        """
        Get data from cache with lazy loading

        Args:
            name: Data file name
            refresh: Force refresh from file
            check_file_changes: Check if file has changed

        Returns:
            Data dictionary
        """
        # Check if in cache and not expired
        if name in self._cache and not refresh:
            cache_entry = self._cache[name]

            # Check expiry
            if not cache_entry.is_expired(self._cache_age_minutes):
                # Check file changes if requested
                if check_file_changes:
                    file_path = self._data_files.get(name)
                    if file_path:
                        current_hash = calculate_file_hash(file_path)
                        if current_hash != cache_entry.file_hash:
                            logger.info(f"File changed, refreshing: {name}")
                            refresh = True

                if not refresh:
                    cache_entry.access()
                    self._cache_hits += 1
                    return cache_entry.data

        # Cache miss or refresh needed
        self._cache_misses += 1

        # Get file path
        file_path = self._data_files.get(name)
        if not file_path:
            logger.error(f"Unknown data file: {name}")
            return {}

        # Load data
        data, error = self._load_json_file(file_path)

        if error:
            logger.warning(f"Failed to load {name}: {error}")
            # Return cached data if available (stale is better than nothing)
            if name in self._cache:
                logger.info(f"Returning stale cached data for {name}")
                return self._cache[name].data
            return {}

        # Calculate file hash
        file_hash = calculate_file_hash(file_path)

        # Update cache
        self._cache[name] = CacheEntry(
            data=data,
            loaded_at=datetime.now(),
            file_path=file_path,
            file_hash=file_hash
        )

        self._total_loads += 1
        logger.debug(f"✅ Loaded {name} ({len(data)} records)")

        return data

    # ========================================================================
    # DATA ACCESSORS (LAZY LOADED)
    # ========================================================================

    def get_analyses(self, refresh: bool = False) -> Dict[str, Any]:
        """Get analyses data with lazy loading"""
        return self._get_cached_data('analyses', refresh)

    def get_compliance_penalties(self, refresh: bool = False) -> Dict[str, Any]:
        """Get compliance penalties data with lazy loading"""
        return self._get_cached_data('compliance_penalties', refresh)

    def get_compliance_rules(self, refresh: bool = False) -> Dict[str, Any]:
        """Get compliance rules data with lazy loading"""
        return self._get_cached_data('compliance_rules', refresh)

    def get_cross_border_regulations(self, refresh: bool = False) -> Dict[str, Any]:
        """Get cross-border regulations data with lazy loading"""
        return self._get_cached_data('cross_border_regulations', refresh)

    def get_dynamic_rules(self, refresh: bool = False) -> Dict[str, Any]:
        """Get dynamic rules data with lazy loading"""
        return self._get_cached_data('dynamic_rules', refresh)

    def get_financial_institutions(self, refresh: bool = False) -> Dict[str, Any]:
        """Get financial institutions data with lazy loading"""
        return self._get_cached_data('financial_institutions', refresh)

    def get_issue_descriptions(self, refresh: bool = False) -> Dict[str, Any]:
        """Get issue descriptions data with lazy loading"""
        return self._get_cached_data('issue_descriptions', refresh)

    def get_lux_keywords(self, refresh: bool = False) -> Dict[str, Any]:
        """Get Luxembourg keywords data with lazy loading"""
        return self._get_cached_data('lux_keywords', refresh)

    def get_regulations(self, refresh: bool = False) -> Dict[str, Any]:
        """Get regulations data with lazy loading"""
        return self._get_cached_data('regulations', refresh)

    def get_reporting_requirements(self, refresh: bool = False) -> Dict[str, Any]:
        """Get reporting requirements data with lazy loading"""
        return self._get_cached_data('reporting_requirements', refresh)

    def get_sanctions_lists(self, refresh: bool = False) -> Dict[str, Any]:
        """Get sanctions lists data with lazy loading"""
        return self._get_cached_data('sanctions_lists', refresh)

    # ========================================================================
    # SEARCH AND QUERY METHODS
    # ========================================================================

    def search_rules_by_keyword(
            self,
            keyword: str,
            case_sensitive: bool = False,
            max_results: int = MAX_SEARCH_RESULTS
    ) -> List[SearchResult]:
        """
        Search compliance rules by keyword with security

        Args:
            keyword: Search keyword
            case_sensitive: Whether search is case-sensitive
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        # Sanitize query
        keyword = sanitize_search_query(keyword)
        if not keyword:
            return []

        rules = self.get_compliance_rules()
        results: List[SearchResult] = []

        keyword_compare = keyword if case_sensitive else keyword.lower()

        for rule_id, rule_data in rules.items():
            if len(results) >= max_results:
                break

            if isinstance(rule_data, dict):
                # Search in rule content
                rule_text = str(rule_data)
                rule_text_compare = rule_text if case_sensitive else rule_text.lower()

                if keyword_compare in rule_text_compare:
                    # Calculate match score based on frequency
                    match_count = rule_text_compare.count(keyword_compare)
                    match_score = min(1.0, match_count / 10.0)  # Normalize to 0-1

                    results.append(SearchResult(
                        item_id=rule_id,
                        data=rule_data,
                        match_score=match_score,
                        match_reason=f'Keyword "{keyword}" found {match_count} times',
                        category='compliance_rule'
                    ))

        # Sort by match score
        results.sort(key=lambda x: x.match_score, reverse=True)

        return results

    def get_rules_for_document_type(
            self,
            doc_type: str,
            include_general: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get applicable rules for document type

        Args:
            doc_type: Document type
            include_general: Include general rules (applicable to all)

        Returns:
            List of applicable rules
        """
        rules = self.get_compliance_rules()
        applicable_rules = []

        doc_type_lower = doc_type.lower()

        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                # Check applicable document types
                applicable_docs = rule_data.get('applicable_document_types', [])

                if not applicable_docs and include_general:
                    # No specific types means applies to all
                    applicable_rules.append({
                        'rule_id': rule_id,
                        'data': rule_data,
                        'applicability': 'general'
                    })
                elif doc_type_lower in [d.lower() for d in applicable_docs]:
                    applicable_rules.append({
                        'rule_id': rule_id,
                        'data': rule_data,
                        'applicability': 'specific'
                    })

        return applicable_rules

    def get_penalty_info(self, violation_type: str) -> Optional[Dict[str, Any]]:
        """
        Get penalty information with enhanced search

        Args:
            violation_type: Type of violation

        Returns:
            Penalty information or None
        """
        penalties = self.get_compliance_penalties()

        violation_type = sanitize_search_query(violation_type)

        # Direct lookup (case-insensitive)
        for key in penalties.keys():
            if key.lower() == violation_type.lower():
                return penalties[key]

        # Fuzzy search in penalty data
        violation_lower = violation_type.lower()
        for penalty_id, penalty_data in penalties.items():
            if isinstance(penalty_data, dict):
                penalty_str = str(penalty_data).lower()
                if violation_lower in penalty_str:
                    return penalty_data

        return None

    # ========================================================================
    # LUXEMBOURG-SPECIFIC METHODS
    # ========================================================================

    def check_luxembourg_relevance(
            self,
            text: str,
            min_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Check Luxembourg relevance with enhanced analysis

        Args:
            text: Text to analyze
            min_score: Minimum relevance score to include category

        Returns:
            Relevance analysis results
        """
        lux_keywords = self.get_lux_keywords()
        text_lower = text.lower()

        findings = {}
        total_found = 0
        total_possible = 0
        category_scores = {}

        for category, keywords in lux_keywords.items():
            if isinstance(keywords, list):
                found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
                total_found += len(found_keywords)
                total_possible += len(keywords)

                category_score = len(found_keywords) / len(keywords) if keywords else 0
                category_scores[category] = category_score

                if found_keywords and category_score >= min_score:
                    findings[category] = {
                        'keywords_found': found_keywords,
                        'count': len(found_keywords),
                        'total': len(keywords),
                        'score': round(category_score, 3)
                    }

            elif isinstance(keywords, dict):
                # Handle nested structure
                for subcategory, subkeywords in keywords.items():
                    if isinstance(subkeywords, list):
                        found_keywords = [
                            kw for kw in subkeywords
                            if kw.lower() in text_lower
                        ]
                        total_found += len(found_keywords)
                        total_possible += len(subkeywords)

                        subscore = len(found_keywords) / len(subkeywords) if subkeywords else 0
                        full_category = f"{category}_{subcategory}"
                        category_scores[full_category] = subscore

                        if found_keywords and subscore >= min_score:
                            findings[full_category] = {
                                'keywords_found': found_keywords,
                                'count': len(found_keywords),
                                'total': len(subkeywords),
                                'score': round(subscore, 3)
                            }

        # Calculate overall relevance
        relevance_score = total_found / total_possible if total_possible > 0 else 0

        # Determine relevance level
        if relevance_score >= 0.3:
            relevance_level = 'high'
        elif relevance_score >= 0.1:
            relevance_level = 'medium'
        elif relevance_score > 0:
            relevance_level = 'low'
        else:
            relevance_level = 'none'

        return {
            'categories': findings,
            'category_scores': category_scores,
            'total_keywords_found': total_found,
            'total_possible': total_possible,
            'luxembourg_relevance_score': round(relevance_score, 3),
            'relevance_level': relevance_level,
            'is_luxembourg_relevant': total_found > 0,
            'top_categories': sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }

    def check_sanctions_list(
            self,
            entity_name: str,
            strict: bool = False
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Check entity against sanctions lists with enhanced matching

        Args:
            entity_name: Entity name to check
            strict: Use strict matching (exact match required)

        Returns:
            Tuple of (has_match, list_of_matches)
        """
        sanctions = self.get_sanctions_lists()
        matches = []

        entity_name = sanitize_search_query(entity_name)
        entity_lower = entity_name.lower()

        for list_name, entries in sanctions.items():
            if isinstance(entries, list):
                for i, entry in enumerate(entries):
                    if isinstance(entry, str):
                        entry_lower = entry.lower()

                        # Exact match or substring match
                        if strict:
                            is_match = entity_lower == entry_lower
                        else:
                            is_match = entity_lower in entry_lower or entry_lower in entity_lower

                        if is_match:
                            matches.append({
                                'list': list_name,
                                'entry': entry,
                                'entry_index': i,
                                'match_type': 'exact' if entity_lower == entry_lower else 'partial',
                                'confidence': 1.0 if entity_lower == entry_lower else 0.7
                            })

                    elif isinstance(entry, dict):
                        entry_text = str(entry).lower()

                        if entity_lower in entry_text:
                            matches.append({
                                'list': list_name,
                                'entry': entry,
                                'entry_index': i,
                                'match_type': 'object_match',
                                'confidence': 0.8
                            })

            elif isinstance(entries, dict):
                # Handle dictionary-based sanctions lists
                for entry_key, entry_value in entries.items():
                    entry_str = f"{entry_key} {str(entry_value)}".lower()

                    if entity_lower in entry_str:
                        matches.append({
                            'list': list_name,
                            'entry': {entry_key: entry_value},
                            'match_type': 'dict_match',
                            'confidence': 0.8
                        })

        # Sort by confidence
        matches.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return len(matches) > 0, matches

    def get_cssf_applicable_circulars(
            self,
            text: str,
            entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Determine applicable CSSF circulars based on content

        Args:
            text: Document text
            entity_type: Type of entity (e.g., 'bank', 'insurance')

        Returns:
            List of applicable circulars with relevance scores
        """
        text_lower = text.lower()

        # CSSF circular keywords
        circular_patterns = {
            '12/552': {
                'keywords': ['gouvernance', 'governance', 'risk management', 'gestion des risques', 'internal control'],
                'entity_types': ['credit_institution', 'investment_firm', 'bank', 'all']
            },
            '20/750': {
                'keywords': ['climate', 'climat', 'esg', 'environmental', 'environnemental', 'sustainability'],
                'entity_types': ['all']
            },
            '21/773': {
                'keywords': ['aml', 'kyc', 'blanchiment', 'sanctions', 'pep', 'anti-money laundering'],
                'entity_types': ['all']
            },
            '08/356': {
                'keywords': ['customer', 'client', 'protection', 'complaint', 'réclamation', 'consumer'],
                'entity_types': ['bank', 'insurance', 'credit_institution']
            }
        }

        applicable = []

        for circular, info in circular_patterns.items():
            # Check keywords
            keyword_matches = [kw for kw in info['keywords'] if kw in text_lower]

            # Check entity type
            entity_applicable = (
                    not entity_type or
                    'all' in info['entity_types'] or
                    entity_type in info['entity_types']
            )

            if keyword_matches and entity_applicable:
                relevance_score = len(keyword_matches) / len(info['keywords'])

                applicable.append({
                    'circular': f'CSSF {circular}',
                    'circular_number': circular,
                    'keywords_matched': keyword_matches,
                    'match_count': len(keyword_matches),
                    'relevance_score': round(relevance_score, 3),
                    'entity_applicable': entity_applicable
                })

        # Sort by relevance
        applicable.sort(key=lambda x: x['relevance_score'], reverse=True)

        return applicable

    # ========================================================================
    # CACHE AND STATISTICS
    # ========================================================================

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about loaded data

        Returns:
            Statistics dictionary
        """
        stats = {}

        for name in self._data_files.keys():
            file_path = self._data_files[name]

            # Get cache entry if loaded
            cache_entry = self._cache.get(name)

            if cache_entry:
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0

                data_stats = DataStatistics(
                    file_name=name,
                    file_path=file_path,
                    record_count=len(cache_entry.data),
                    file_size_bytes=file_size,
                    last_loaded=cache_entry.loaded_at,
                    access_count=cache_entry.access_count,
                    has_data=len(cache_entry.data) > 0,
                    cache_expired=cache_entry.is_expired(self._cache_age_minutes)
                )

                stats[name] = data_stats.to_dict()
            else:
                # Not loaded yet
                file_exists = Path(file_path).exists()
                file_size = Path(file_path).stat().st_size if file_exists else 0

                stats[name] = {
                    'file_name': name,
                    'file_path': file_path,
                    'loaded': False,
                    'exists': file_exists,
                    'file_size_mb': round(file_size / (1024 * 1024), 2) if file_exists else 0
                }

        # Summary statistics
        loaded_files = len(self._cache)
        total_records = sum(
            s['record_count'] for s in stats.values()
            if 'record_count' in s
        )

        stats['_summary'] = {
            'total_files': len(self._data_files),
            'loaded_files': loaded_files,
            'files_with_data': sum(
                1 for s in stats.values()
                if isinstance(s, dict) and s.get('has_data', False)
            ),
            'total_records': total_records,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': round(
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0,
                3
            ),
            'total_loads': self._total_loads,
            'total_refreshes': self._total_refreshes,
            'data_directory': str(Path(list(self._data_files.values())[0]).parent) if self._data_files else 'unknown'
        }

        return stats

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information

        Returns:
            Cache information dictionary
        """
        return {
            'cached_files': list(self._cache.keys()),
            'cache_size': len(self._cache),
            'cache_age_minutes': self._cache_age_minutes,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': round(
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0 else 0,
                3
            ),
            'expired_entries': sum(
                1 for entry in self._cache.values()
                if entry.is_expired(self._cache_age_minutes)
            )
        }

    def clear_cache(self, file_name: Optional[str] = None) -> None:
        """
        Clear cache

        Args:
            file_name: Specific file to clear, or None for all
        """
        if file_name:
            if file_name in self._cache:
                del self._cache[file_name]
                logger.info(f"🗑️  Cleared cache for {file_name}")
        else:
            self._cache.clear()
            logger.info("🗑️  Cleared all cache")

    def refresh_all_data(self) -> None:
        """Refresh all cached data from files"""
        for name in list(self._cache.keys()):
            self._get_cached_data(name, refresh=True, check_file_changes=False)

        self._total_refreshes += 1
        logger.info("🔄 All data refreshed from files")

    def preload_all(self, core_only: bool = False) -> None:
        """
        Preload data files

        Args:
            core_only: Only load core files
        """
        files_to_load = CORE_DATA_FILES if core_only else self._data_files.keys()

        for name in files_to_load:
            if name in self._data_files:
                self._get_cached_data(name)

        logger.info(f"📦 Preloaded {len(files_to_load)} data files")

    # ========================================================================
    # BACKWARD COMPATIBILITY
    # ========================================================================

    def get_rules(self, regulation_type: str = "all") -> List[Dict]:
        """
        Get rules for backward compatibility

        Args:
            regulation_type: Type of regulation or "all"

        Returns:
            List of rules
        """
        try:
            rules = self.get_compliance_rules()

            if regulation_type == "all":
                return list(rules.values())
            else:
                # Filter by regulation type
                filtered_rules = []
                regulation_lower = regulation_type.lower()

                for rule_data in rules.values():
                    if isinstance(rule_data, dict):
                        rule_regulation = rule_data.get('regulation', '').lower()
                        rule_category = rule_data.get('category', '').lower()

                        if (regulation_lower in rule_regulation or
                                regulation_lower in rule_category):
                            filtered_rules.append(rule_data)

                return filtered_rules

        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_data_manager(
        auto_load: bool = False,
        cache_age_minutes: int = MAX_CACHE_AGE_MINUTES
) -> ComplianceDataManager:
    """
    Factory function to create data manager

    Args:
        auto_load: Whether to preload all data
        cache_age_minutes: Cache expiry time

    Returns:
        Configured data manager instance
    """
    return ComplianceDataManager(
        auto_load=auto_load,
        cache_age_minutes=cache_age_minutes
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'ComplianceDataManager',
    'CacheEntry',
    'SearchResult',
    'DataStatistics',
    'create_data_manager',
    'validate_json_file',
    'sanitize_search_query',
    'MAX_JSON_FILE_SIZE',
    'MAX_CACHE_AGE_MINUTES',
    'CORE_DATA_FILES',
    'EXTENDED_DATA_FILES'
]