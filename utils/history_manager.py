# utils/history_manager.py - Optimized history management

import streamlit as st
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class HistoryManager:
    """Optimized analysis history management with pagination and filtering"""
    
    MAX_MEMORY_ITEMS = 50  # Keep only recent items in memory
    ITEMS_PER_PAGE = 10
    
    def __init__(self, session_manager):
        self.session_manager = session_manager
    
    def add_analysis(self, analysis_data: Dict[str, Any]) -> None:
        """Add analysis with memory optimization"""
        history = self.get_history()
        
        # Add timestamp if not present
        if 'timestamp' not in analysis_data:
            analysis_data['timestamp'] = datetime.now().isoformat()
        
        # Add to beginning of list (most recent first)
        history.insert(0, analysis_data)
        
        # Optimize memory usage
        if len(history) > self.MAX_MEMORY_ITEMS:
            # Keep recent items in session, save older ones to persistent storage
            recent_items = history[:self.MAX_MEMORY_ITEMS]
            older_items = history[self.MAX_MEMORY_ITEMS:]
            
            self._save_to_persistent_storage(older_items)
            self.session_manager.set('analysis_history', recent_items)
        else:
            self.session_manager.set('analysis_history', history)
        
        # Set as current analysis
        self.session_manager.set('current_analysis', analysis_data)
        logger.info(f"Analysis added. Total in memory: {len(self.get_history())}")
    
    def get_history(self, include_persistent: bool = False) -> List[Dict[str, Any]]:
        """Get analysis history with optional persistent storage inclusion"""
        memory_history = self.session_manager.get('analysis_history', [])
        
        if include_persistent:
            persistent_history = self._load_from_persistent_storage()
            # Merge and deduplicate by timestamp
            all_history = memory_history + persistent_history
            seen_timestamps = set()
            unique_history = []
            
            for item in all_history:
                timestamp = item.get('timestamp', '')
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_history.append(item)
            
            # Sort by timestamp (most recent first)
            unique_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return unique_history
        
        return memory_history
    
    def get_paginated_history(self, page: int = 1, items_per_page: int = None) -> Tuple[List[Dict], int, int]:
        """Get paginated history"""
        if items_per_page is None:
            items_per_page = self.ITEMS_PER_PAGE
        
        history = self.get_history(include_persistent=True)
        total_items = len(history)
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Ensure page is within bounds
        page = max(1, min(page, total_pages))
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        page_items = history[start_idx:end_idx]
        
        return page_items, page, total_pages
    
    def filter_history(self, 
                      date_from: Optional[datetime] = None,
                      date_to: Optional[datetime] = None,
                      min_score: Optional[float] = None,
                      max_score: Optional[float] = None,
                      doc_type: Optional[str] = None,
                      has_issues: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Filter history with multiple criteria"""
        history = self.get_history(include_persistent=True)
        filtered = []
        
        for item in history:
            # Date filtering
            if date_from or date_to:
                try:
                    item_date = datetime.fromisoformat(item.get('timestamp', ''))
                    if date_from and item_date < date_from:
                        continue
                    if date_to and item_date > date_to:
                        continue
                except:
                    continue
            
            # Score filtering
            score = item.get('score', 0)
            if min_score is not None and score < min_score:
                continue
            if max_score is not None and score > max_score:
                continue
            
            # Document type filtering
            if doc_type and item.get('doc_type', '') != doc_type:
                continue
            
            # Issues filtering
            if has_issues is not None:
                item_has_issues = len(item.get('issues', [])) > 0
                if has_issues != item_has_issues:
                    continue
            
            filtered.append(item)
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from history"""
        history = self.get_history(include_persistent=True)
        
        if not history:
            return {
                'total_analyses': 0,
                'average_score': 0,
                'total_issues': 0,
                'critical_issues': 0,
                'score_trend': [],
                'doc_types': {},
                'issues_by_severity': {},
                'monthly_activity': {}
            }
        
        # Basic stats
        total_analyses = len(history)
        scores = [item.get('score', 0) for item in history]
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Issues stats
        all_issues = []
        for item in history:
            all_issues.extend(item.get('issues', []))
        
        total_issues = len(all_issues)
        critical_issues = len([i for i in all_issues if i.get('severity') == 'critical'])
        
        # Issues by severity
        issues_by_severity = {}
        for issue in all_issues:
            severity = issue.get('severity', 'medium')
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
        
        # Document types
        doc_types = {}
        for item in history:
            doc_type = item.get('doc_type', 'Unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Monthly activity
        monthly_activity = {}
        for item in history:
            try:
                date = datetime.fromisoformat(item.get('timestamp', ''))
                month_key = date.strftime('%Y-%m')
                monthly_activity[month_key] = monthly_activity.get(month_key, 0) + 1
            except:
                continue
        
        # Score trend (last 10 analyses)
        recent_scores = [(item.get('timestamp', ''), item.get('score', 0)) 
                        for item in history[:10]]
        recent_scores.reverse()  # Chronological order
        
        return {
            'total_analyses': total_analyses,
            'average_score': average_score,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'score_trend': recent_scores,
            'doc_types': doc_types,
            'issues_by_severity': issues_by_severity,
            'monthly_activity': monthly_activity
        }
    
    def export_history(self, format: str = 'json') -> str:
        """Export history in various formats"""
        history = self.get_history(include_persistent=True)
        
        if format == 'json':
            return json.dumps({
                'export_date': datetime.now().isoformat(),
                'total_analyses': len(history),
                'analyses': history
            }, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            if not history:
                return "No data to export"
            
            # Flatten data for CSV
            csv_data = []
            for item in history:
                row = {
                    'timestamp': item.get('timestamp', ''),
                    'doc_type': item.get('doc_type', ''),
                    'score': item.get('score', 0),
                    'num_issues': len(item.get('issues', [])),
                    'critical_issues': len([i for i in item.get('issues', []) 
                                          if i.get('severity') == 'critical']),
                    'detected_language': item.get('detected_language', ''),
                    'file_name': item.get('file_name', '')
                }
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            return df.to_csv(index=False)
        
        return "Unsupported format"
    
    def clear_history(self, keep_recent: int = 0) -> None:
        """Clear history, optionally keeping recent items"""
        if keep_recent > 0:
            history = self.get_history()
            recent_items = history[:keep_recent]
            self.session_manager.set('analysis_history', recent_items)
        else:
            self.session_manager.set('analysis_history', [])
        
        # Clear persistent storage too
        self._clear_persistent_storage()
        logger.info(f"History cleared, kept {keep_recent} recent items")
    
    def _save_to_persistent_storage(self, items: List[Dict[str, Any]]) -> None:
        """Save items to persistent storage (implement based on your storage backend)"""
        # This is a placeholder - implement based on your needs
        # Could save to file, database, etc.
        try:
            from utils.paths import get_data_path
            storage_path = get_data_path('history_archive.json')
            
            existing_data = []
            try:
                with open(storage_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except:
                pass
            
            # Merge and deduplicate
            all_items = existing_data + items
            seen_timestamps = set()
            unique_items = []
            
            for item in all_items:
                timestamp = item.get('timestamp', '')
                if timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_items.append(item)
            
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(unique_items, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save to persistent storage: {e}")
    
    def _load_from_persistent_storage(self) -> List[Dict[str, Any]]:
        """Load items from persistent storage"""
        try:
            from utils.paths import get_data_path
            storage_path = get_data_path('history_archive.json')
            
            with open(storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _clear_persistent_storage(self) -> None:
        """Clear persistent storage"""
        try:
            from utils.paths import get_data_path
            storage_path = get_data_path('history_archive.json')
            
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            logger.error(f"Failed to clear persistent storage: {e}")