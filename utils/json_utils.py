# utils/json_utils.py - CORRECTED VERSION WITH ALL IMPORTS
import json
import logging
import fcntl
import platform
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

def load_json(
    path: Union[str, Path], 
    default: Optional[Dict[str, Any]] = None,
    create_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Load JSON data from file with comprehensive error handling
    
    Args:
        path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        create_if_missing: Whether to create file with default value if missing
        
    Returns:
        Dictionary with JSON data or default value
    """
    if default is None:
        default = {}
    
    path = Path(path)
    
    # Check if file exists
    if not path.exists():
        if create_if_missing:
            logger.info(f"Creating missing JSON file: {path}")
            return save_json(path, default) and default or default
        else:
            logger.warning(f"JSON file not found: {path}")
            return default
    
    # Try to load JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"Empty JSON file: {path}")
                if create_if_missing:
                    return save_json(path, default) and default or default
                return default
            
            data = json.loads(content)
            logger.debug(f"Successfully loaded JSON: {path}")
            return data
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        if create_if_missing:
            logger.info(f"Recreating corrupted JSON file: {path}")
            return save_json(path, default) and default or default
        return default
    
    except Exception as e:
        logger.error(f"Error reading JSON file {path}: {e}")
        return default

def save_json_safe(path: Union[str, Path], data: Dict[str, Any]) -> bool:
    """Save JSON with file locking to prevent race conditions"""
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    
    try:
        # Create parent directory
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        with open(temp_path, 'w', encoding='utf-8') as f:
            # Add file locking on Unix systems
            if platform.system() != 'Windows':
                try:
                    if hasattr(fcntl, 'flock'):
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (ImportError, AttributeError, OSError):
                    pass  # Skip locking if not available
            
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic move to final location
        temp_path.replace(path)
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        # Clean up temp file
        try:
            if temp_path.exists():
                temp_path.unlink()
        except:
            pass
        return False

def save_json(
    path: Union[str, Path], 
    data: Dict[str, Any],
    backup: bool = True,
    indent: int = 2
) -> bool:
    """
    Save JSON data to file with error handling and optional backup
    
    Args:
        path: Path to JSON file
        data: Data to save
        backup: Whether to create backup before overwriting
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    
    try:
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create backup if file exists and backup is requested
        if backup and path.exists():
            backup_path = path.with_suffix(f"{path.suffix}.backup")
            try:
                import shutil
                shutil.copy2(path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
        
        # Use safe saving method
        return save_json_safe(path, data)
        
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        return False

def merge_json_files(
    *file_paths: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Merge multiple JSON files into one dictionary
    
    Args:
        *file_paths: Paths to JSON files to merge
        output_path: Optional path to save merged result
        
    Returns:
        Merged dictionary
    """
    merged = {}
    
    for file_path in file_paths:
        data = load_json(file_path, default={}, create_if_missing=False)
        if data:
            merged.update(data)
            logger.debug(f"Merged data from: {file_path}")
    
    if output_path:
        save_json(output_path, merged)
        logger.info(f"Saved merged JSON to: {output_path}")
    
    return merged

def validate_json_schema(
    data: Dict[str, Any], 
    required_keys: List[str],
    optional_keys: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate JSON data against a simple schema
    
    Args:
        data: JSON data to validate
        required_keys: List of required keys
        optional_keys: List of optional keys
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if optional_keys is None:
        optional_keys = []
    
    errors = []
    
    # Check required keys
    for key in required_keys:
        if key not in data:
            errors.append(f"Missing required key: {key}")
        elif data[key] is None:
            errors.append(f"Required key has null value: {key}")
    
    # Check for unexpected keys
    allowed_keys = set(required_keys + optional_keys)
    for key in data.keys():
        if key not in allowed_keys:
            errors.append(f"Unexpected key: {key}")
    
    return len(errors) == 0, errors

def cleanup_json_backups(
    directory: Union[str, Path],
    max_backups: int = 5
) -> int:
    """
    Clean up old JSON backup files
    
    Args:
        directory: Directory to clean
        max_backups: Maximum number of backups to keep per file
        
    Returns:
        Number of files cleaned up
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    backup_files = list(directory.glob("*.json.backup"))
    backup_groups = {}
    
    # Group backups by original filename
    for backup_file in backup_files:
        original_name = backup_file.name.replace('.backup', '')
        if original_name not in backup_groups:
            backup_groups[original_name] = []
        backup_groups[original_name].append(backup_file)
    
    cleaned_count = 0
    
    # Keep only the newest backups for each file
    for original_name, backups in backup_groups.items():
        if len(backups) > max_backups:
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old backups
            for old_backup in backups[max_backups:]:
                try:
                    old_backup.unlink()
                    cleaned_count += 1
                    logger.debug(f"Removed old backup: {old_backup}")
                except Exception as e:
                    logger.warning(f"Could not remove backup {old_backup}: {e}")
    
    return cleaned_count

def get_json_file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a JSON file
    
    Args:
        path: Path to JSON file
        
    Returns:
        Dictionary with file information
    """
    path = Path(path)
    
    info = {
        'path': str(path),
        'exists': path.exists(),
        'size_bytes': 0,
        'record_count': 0,
        'is_valid_json': False,
        'last_modified': None,
        'encoding': 'unknown'
    }
    
    if not path.exists():
        return info
    
    try:
        # File statistics
        stat = path.stat()
        info['size_bytes'] = stat.st_size
        info['last_modified'] = stat.st_mtime
        
        # Try to load and analyze JSON
        data = load_json(path, create_if_missing=False)
        if data is not None:
            info['is_valid_json'] = True
            if isinstance(data, dict):
                info['record_count'] = len(data)
            elif isinstance(data, list):
                info['record_count'] = len(data)
        
        # Detect encoding
        with open(path, 'rb') as f:
            raw_data = f.read(1024)  # Read first 1KB
            try:
                raw_data.decode('utf-8')
                info['encoding'] = 'utf-8'
            except UnicodeDecodeError:
                try:
                    raw_data.decode('latin-1')
                    info['encoding'] = 'latin-1'
                except UnicodeDecodeError:
                    info['encoding'] = 'unknown'
    
    except Exception as e:
        logger.error(f"Error analyzing JSON file {path}: {e}")
        info['error'] = str(e)
    
    return info

# Convenience functions for common data files
def load_users() -> Dict[str, Any]:
    """Load users.json"""
    try:
        from .config import config
        return load_json(config.get_data_file_path('users'))
    except ImportError:
        # Fallback if config not available
        return load_json("data/users.json")

def save_users(users_data: Dict[str, Any]) -> bool:
    """Save users.json"""
    try:
        from .config import config
        return save_json(config.get_data_file_path('users'), users_data)
    except ImportError:
        # Fallback if config not available
        return save_json("data/users.json", users_data)

def load_sessions() -> Dict[str, Any]:
    """Load sessions.json"""
    try:
        from .config import config
        return load_json(config.get_data_file_path('sessions'))
    except ImportError:
        # Fallback if config not available
        return load_json("data/sessions.json")

def save_sessions(sessions_data: Dict[str, Any]) -> bool:
    """Save sessions.json"""
    try:
        from .config import config
        return save_json(config.get_data_file_path('sessions'), sessions_data)
    except ImportError:
        # Fallback if config not available
        return save_json("data/sessions.json", sessions_data)

def load_translations() -> Dict[str, Any]:
    """Load translations.json"""
    try:
        from .config import config
        return load_json(config.get_data_file_path('translations'))
    except ImportError:
        # Fallback if config not available
        return load_json("data/translations.json")

def save_translations(translations_data: Dict[str, Any]) -> bool:
    """Save translations.json"""
    try:
        from .config import config
        return save_json(config.get_data_file_path('translations'), translations_data)
    except ImportError:
        # Fallback if config not available
        return save_json("data/translations.json", translations_data)