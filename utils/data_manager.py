# utils/data_manager.py - SIMPLE VERSION FOR YOUR DATA
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ComplianceDataManager:
    """Simple data manager that works with your actual data files"""
    
    def __init__(self):
        self._cache = {}
        self._cache_timestamps = {}
        self._data_files = self._get_your_data_files()
        self._initialize_cache()
    
    def _get_your_data_files(self) -> Dict[str, str]:
        """Get your actual data file paths"""
        try:
            from .config import config
            # Use config if available
            data_files = {}
            for key in config.DATA_FILES.keys():
                if key not in ['users', 'sessions', 'translations']:  # Skip runtime files
                    data_files[key] = str(config.get_data_file_path(key))
            return data_files
        except ImportError:
            # Fallback - direct paths to your files
            return {
                'analyses': 'data/analyses.json',
                'compliance_penalties': 'data/compliance_penalties.json',
                'compliance_rules': 'data/compliance_rules.json',
                'cross_border_regulations': 'data/cross_border_regulations.json',
                'dynamic_rules': 'data/dynamic_rules.json',
                'financial_institutions': 'data/financial_institutions.json',
                'issue_descriptions': 'data/issue_descriptions.json',
                'lux_keywords': 'data/lux_keywords.json',
                'regulations': 'data/regulations.json',
                'reporting_requirements': 'data/reporting_requirements.json',
                'sanctions_lists': 'data/sanctions_lists.json'
            }
    
    def _initialize_cache(self):
        """Load your data files into cache"""
        for name, path in self._data_files.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            self._cache[name] = json.loads(content)
                        else:
                            self._cache[name] = {}
                    self._cache_timestamps[name] = datetime.now()
                    logger.debug(f"✅ Loaded {name} from {path}")
                except Exception as e:
                    logger.warning(f"❌ Failed to load {name}: {e}")
                    self._cache[name] = {}
            else:
                logger.warning(f"⚠️ Data file not found: {path}")
                self._cache[name] = {}
    
    def _get_cached_data(self, name: str, refresh: bool = False) -> Dict[str, Any]:
        """Get cached data with optional refresh"""
        if refresh or name not in self._cache:
            file_path = self._data_files.get(name)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            self._cache[name] = json.loads(content)
                        else:
                            self._cache[name] = {}
                    self._cache_timestamps[name] = datetime.now()
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
        
        return self._cache.get(name, {})
    
    # Main accessors for your data files
    def get_analyses(self, refresh: bool = False) -> Dict[str, Any]:
        """Get analyses.json data"""
        return self._get_cached_data('analyses', refresh)
    
    def get_compliance_penalties(self, refresh: bool = False) -> Dict[str, Any]:
        """Get compliance_penalties.json data"""
        return self._get_cached_data('compliance_penalties', refresh)
    
    def get_compliance_rules(self, refresh: bool = False) -> Dict[str, Any]:
        """Get compliance_rules.json data"""
        return self._get_cached_data('compliance_rules', refresh)
    
    def get_cross_border_regulations(self, refresh: bool = False) -> Dict[str, Any]:
        """Get cross_border_regulations.json data"""
        return self._get_cached_data('cross_border_regulations', refresh)
    
    def get_dynamic_rules(self, refresh: bool = False) -> Dict[str, Any]:
        """Get dynamic_rules.json data"""
        return self._get_cached_data('dynamic_rules', refresh)
    
    def get_financial_institutions(self, refresh: bool = False) -> Dict[str, Any]:
        """Get financial_institutions.json data"""
        return self._get_cached_data('financial_institutions', refresh)
    
    def get_issue_descriptions(self, refresh: bool = False) -> Dict[str, Any]:
        """Get issue_descriptions.json data"""
        return self._get_cached_data('issue_descriptions', refresh)
    
    def get_lux_keywords(self, refresh: bool = False) -> Dict[str, Any]:
        """Get lux_keywords.json data"""
        return self._get_cached_data('lux_keywords', refresh)
    
    def get_regulations(self, refresh: bool = False) -> Dict[str, Any]:
        """Get regulations.json data"""
        return self._get_cached_data('regulations', refresh)
    
    def get_reporting_requirements(self, refresh: bool = False) -> Dict[str, Any]:
        """Get reporting_requirements.json data"""
        return self._get_cached_data('reporting_requirements', refresh)
    
    def get_sanctions_lists(self, refresh: bool = False) -> Dict[str, Any]:
        """Get sanctions_lists.json data"""
        return self._get_cached_data('sanctions_lists', refresh)
    
    # Analysis methods using your data
    def search_rules_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Search your compliance rules by keyword"""
        rules = self.get_compliance_rules()
        results = []
        
        keyword_lower = keyword.lower()
        
        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                # Search in rule content
                rule_text = str(rule_data).lower()
                if keyword_lower in rule_text:
                    results.append({
                        'rule_id': rule_id,
                        'data': rule_data,
                        'match_reason': 'keyword_found'
                    })
        
        return results
    
    def get_rules_for_document_type(self, doc_type: str) -> List[Dict[str, Any]]:
        """Get applicable rules for document type"""
        rules = self.get_compliance_rules()
        applicable_rules = []
        
        for rule_id, rule_data in rules.items():
            if isinstance(rule_data, dict):
                # Check if rule applies to document type
                applicable_docs = rule_data.get('applicable_document_types', [])
                if not applicable_docs or doc_type in applicable_docs:
                    applicable_rules.append({
                        'rule_id': rule_id,
                        'data': rule_data
                    })
        
        return applicable_rules
    
    def get_penalty_info(self, violation_type: str) -> Optional[Dict[str, Any]]:
        """Get penalty information from your data"""
        penalties = self.get_compliance_penalties()
        
        # Direct lookup
        if violation_type in penalties:
            return penalties[violation_type]
        
        # Search in penalty data
        for penalty_id, penalty_data in penalties.items():
            if isinstance(penalty_data, dict):
                if violation_type.lower() in str(penalty_data).lower():
                    return penalty_data
        
        return None
    
    def check_luxembourg_relevance(self, text: str) -> Dict[str, Any]:
        """Check Luxembourg relevance using your lux_keywords.json"""
        lux_keywords = self.get_lux_keywords()
        text_lower = text.lower()
        
        findings = {}
        total_found = 0
        total_possible = 0
        
        for category, keywords in lux_keywords.items():
            if isinstance(keywords, list):
                found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
                total_found += len(found_keywords)
                total_possible += len(keywords)
                
                if found_keywords:
                    findings[category] = {
                        'keywords_found': found_keywords,
                        'count': len(found_keywords),
                        'total': len(keywords)
                    }
            elif isinstance(keywords, dict):
                # Handle nested structure
                for subcategory, subkeywords in keywords.items():
                    if isinstance(subkeywords, list):
                        found_keywords = [kw for kw in subkeywords if kw.lower() in text_lower]
                        total_found += len(found_keywords)
                        total_possible += len(subkeywords)
                        
                        if found_keywords:
                            findings[f"{category}_{subcategory}"] = {
                                'keywords_found': found_keywords,
                                'count': len(found_keywords),
                                'total': len(subkeywords)
                            }
        
        relevance_score = total_found / total_possible if total_possible > 0 else 0
        
        return {
            'categories': findings,
            'total_keywords_found': total_found,
            'total_possible': total_possible,
            'luxembourg_relevance_score': relevance_score,
            'is_luxembourg_relevant': total_found > 0
        }
    
    def check_sanctions_list(self, entity_name: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check entity against your sanctions lists"""
        sanctions = self.get_sanctions_lists()
        matches = []
        entity_lower = entity_name.lower()
        
        for list_name, entries in sanctions.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, str) and entity_lower in entry.lower():
                        matches.append({
                            'list': list_name,
                            'entry': entry,
                            'match_type': 'string'
                        })
                    elif isinstance(entry, dict):
                        entry_text = str(entry).lower()
                        if entity_lower in entry_text:
                            matches.append({
                                'list': list_name,
                                'entry': entry,
                                'match_type': 'object'
                            })
        
        return len(matches) > 0, matches
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about your loaded data"""
        stats = {}
        
        for name, data in self._cache.items():
            stats[name] = {
                'file_name': self._data_files.get(name, 'unknown'),
                'record_count': len(data) if isinstance(data, dict) else 0,
                'last_loaded': self._cache_timestamps.get(name),
                'has_data': len(data) > 0 if isinstance(data, (dict, list)) else False
            }
        
        # Summary
        total_records = sum(s['record_count'] for s in stats.values())
        files_with_data = sum(1 for s in stats.values() if s['has_data'])
        
        stats['_summary'] = {
            'total_files': len(stats),
            'files_with_data': files_with_data,
            'total_records': total_records,
            'data_directory': 'data/',
            'all_files_loaded': files_with_data == len(self._data_files)
        }
        
        return stats
    
    def refresh_all_data(self) -> None:
        """Refresh all data from files"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._initialize_cache()
        logger.info("🔄 All data refreshed from your files")
    
    # Backward compatibility method
    def get_rules(self, regulation_type: str = "all") -> List[Dict]:
        """Get rules for backward compatibility"""
        try:
            rules = self.get_compliance_rules()
            if regulation_type == "all":
                return list(rules.values())
            else:
                # Filter by regulation type
                filtered_rules = []
                for rule_data in rules.values():
                    if isinstance(rule_data, dict):
                        rule_regulation = rule_data.get('regulation', '').lower()
                        if regulation_type.lower() in rule_regulation:
                            filtered_rules.append(rule_data)
                return filtered_rules
        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return []