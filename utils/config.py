# utils/config.py - SIMPLE VERSION THAT WORKS
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Simple configuration for LexAI that works with your data/ folder"""
    
    # Application settings
    APP_VERSION = "2.0.0"
    DEFAULT_LANGUAGE = "fr"
    
    # File size limits
    MAX_DOCUMENT_SIZE = 10_000_000  # 10MB
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    # Base directory
    BASE_DIR = Path(__file__).parent.parent
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get data directory"""
        data_dir = Path(os.environ.get("DATA_DIR", cls.BASE_DIR / "data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @classmethod
    def get_assets_dir(cls) -> Path:
        """Get assets directory"""
        assets_dir = Path(os.environ.get("ASSETS_DIR", cls.BASE_DIR / "assets"))
        assets_dir.mkdir(parents=True, exist_ok=True)
        return assets_dir
    
    @classmethod
    def get_logs_dir(cls) -> Path:
        """Get logs directory"""
        logs_dir = Path(os.environ.get("LOGS_DIR", cls.BASE_DIR / "logs"))
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    @classmethod
    def get_models_dir(cls) -> Path:
        """Get models directory"""
        models_dir = Path(os.environ.get("MODEL_CACHE_DIR", cls.BASE_DIR / "models_cache"))
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir
    
    @classmethod
    def get_data_path(cls, filename: str = "", subfolder: Optional[str] = None) -> Path:
        """Get path in data directory"""
        data_dir = cls.get_data_dir()
        if subfolder:
            data_dir = data_dir / subfolder
            data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / filename if filename else data_dir
    
    @classmethod
    def get_asset_path(cls, filename: str = "", subfolder: Optional[str] = None) -> Path:
        """Get path in assets directory"""
        assets_dir = cls.get_assets_dir()
        if subfolder:
            assets_dir = assets_dir / subfolder
            assets_dir.mkdir(parents=True, exist_ok=True)
        return assets_dir / filename if filename else assets_dir
    
    @classmethod
    def get_log_path(cls, filename: str = "", subfolder: Optional[str] = None) -> Path:
        """Get path in logs directory"""
        logs_dir = cls.get_logs_dir()
        if subfolder:
            logs_dir = logs_dir / subfolder
            logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir / filename if filename else logs_dir
    
    @classmethod
    def get_model_path(cls, filename: str = "", subfolder: Optional[str] = None) -> Path:
        """Get path in models directory"""
        models_dir = cls.get_models_dir()
        if subfolder:
            models_dir = models_dir / subfolder
            models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / filename if filename else models_dir
    
    # Your actual data files (matching your screenshot)
    DATA_FILES = {
        'analyses': 'analyses.json',
        'compliance_penalties': 'compliance_penalties.json', 
        'compliance_rules': 'compliance_rules.json',
        'cross_border_regulations': 'cross_border_regulations.json',
        'dynamic_rules': 'dynamic_rules.json',
        'financial_institutions': 'financial_institutions.json',
        'issue_descriptions': 'issue_descriptions.json',
        'lux_keywords': 'lux_keywords.json',
        'regulations': 'regulations.json',
        'reporting_requirements': 'reporting_requirements.json',
        'sanctions_lists': 'sanctions_lists.json',
        # Runtime files
        'users': 'users.json',
        'sessions': 'sessions.json',
        'translations': 'translations.json'
    }
    
    @classmethod
    def get_data_file_path(cls, file_key: str) -> Path:
        """Get path to specific data file"""
        if file_key not in cls.DATA_FILES:
            raise ValueError(f"Unknown data file key: {file_key}. Available: {list(cls.DATA_FILES.keys())}")
        return cls.get_data_path(cls.DATA_FILES[file_key])
    
    @classmethod
    def get_all_data_file_paths(cls) -> Dict[str, Path]:
        """Get all data file paths"""
        return {key: cls.get_data_file_path(key) for key in cls.DATA_FILES}
    
    # Supported settings
    SUPPORTED_FILE_TYPES = ['pdf', 'docx', 'txt']
    SUPPORTED_LANGUAGES = {
        "fr": "Français",
        "en": "English", 
        "de": "Deutsch",
        "es": "Español"
    }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'default_model': os.environ.get("LLM_MODEL", "google/flan-t5-small"),
            'model_cache_dir': str(cls.get_models_dir()),
            'fallback_model': "google/flan-t5-small"
        }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, Any]:
        """Validate environment and return status"""
        status = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'directories': {},
            'data_files': {}
        }
        
        # Check directories
        directories = {
            'data': cls.get_data_dir(),
            'assets': cls.get_assets_dir(),
            'logs': cls.get_logs_dir(),
            'models': cls.get_models_dir()
        }
        
        for name, path in directories.items():
            exists = path.exists()
            writable = exists and os.access(path, os.W_OK)
            status['directories'][name] = {
                'path': str(path),
                'exists': exists,
                'writable': writable
            }
            if not exists:
                status['warnings'].append(f"Directory {name} does not exist: {path}")
            elif not writable:
                status['warnings'].append(f"Directory {name} is not writable: {path}")
        
        # Check your actual data files
        for key in cls.DATA_FILES.keys():
            file_path = cls.get_data_file_path(key)
            exists = file_path.exists()
            status['data_files'][key] = {
                'path': str(file_path),
                'exists': exists
            }
            if not exists and key in ['users', 'sessions', 'translations']:
                # Only warn for runtime files
                status['warnings'].append(f"Runtime file {key} missing: {file_path}")
        
        return status

# Global config instance
config = Config()

# For backward compatibility
DATA_DIR = config.get_data_dir()
ASSETS_DIR = config.get_assets_dir()
LOGS_DIR = config.get_logs_dir()
MODEL_CACHE_DIR = config.get_models_dir()

# Simple function for backward compatibility
def get_data_path(filename: str = "", subfolder: Optional[str] = None) -> str:
    """Simple function to get data paths"""
    return str(config.get_data_path(filename, subfolder))