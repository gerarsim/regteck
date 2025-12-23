#!/usr/bin/env python3
"""
Fixed Ollama/Compliance Engine Online Checker
Works with data/ subdirectory structure
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class EngineHealthChecker:
    """Fixed health checker for compliance engine with correct paths"""

    def __init__(self, base_dir: str = None):
        """
        Initialize with base directory
        If None, uses parent directory (project root)
        """
        if base_dir is None:
            # Get parent directory of utils/ (which is project root)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)

        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')

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
        """
        Check if compliance engine is properly installed and operational
        This is the fixed version of check_ollama_installation
        """
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
            # Add project root to path if needed
            if self.base_dir not in sys.path:
                sys.path.insert(0, self.base_dir)

            from engine import LocalComplianceEngine, analyze_document_compliance

            result["installed"] = True
            result["engine_type"] = "local_json_based_decimal_corrected"

            # Try to instantiate and test
            try:
                engine = LocalComplianceEngine(self.data_dir)
                stats = engine.get_analysis_statistics()

                result["running"] = True
                result["models"] = ["local_engine_v4.0_decimal_corrected"]
                result["features"] = [
                    "Excellence metrics",
                    "100.00% scoring capability",
                    "Advanced weighting",
                    "Banking sector optimization",
                    "Decimal precision (XX.XX%)",
                    "Luxembourg-specific analysis"
                ]
                result["statistics"] = stats

            except Exception as e:
                result["running"] = False
                result["errors"] = [f"Engine initialization failed: {str(e)}"]
                logger.warning(f"Engine init error: {e}")

        except ImportError as e:
            result["installed"] = False
            result["errors"] = [f"Engine module not found: {str(e)}"]
            result["recommendations"] = [
                "Ensure engine.py is present in project root",
                "Check Python import paths",
                "Verify engine.py is not corrupted"
            ]

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

        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            data_info["errors"] = [f"Data directory not found: {self.data_dir}"]
            for filename in self.required_files:
                data_info["missing"].append(filename)
            return data_info

        for filename in self.required_files:
            # Look in data/ subdirectory
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                data_info["found"] += 1

                # Validate JSON
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
        """
        Validate JSON file
        Returns: (is_valid, error_message, size_mb, record_count)
        """
        try:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Count records
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
        found_ratio = data_files["found"] / data_files["total_required"]
        valid_ratio = len(data_files["valid"]) / data_files["total_required"]

        # Determine status
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

        # Add recommendations
        if data_files["missing"]:
            result["recommendations"].append(
                f"Add missing files to data/: {', '.join(data_files['missing'][:3])}{'...' if len(data_files['missing']) > 3 else ''}"
            )

        if data_files["corrupted"]:
            result["recommendations"].append(
                f"Fix corrupted files: {', '.join(data_files['corrupted'])}"
            )

        if not result["running"] and result["installed"]:
            result["recommendations"].append(
                "Check engine.py initialization - review error logs"
            )

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
            "data_summary": f"{len(full_status['data_files']['valid'])}/{full_status['data_files']['total_required']} files valid",
            "decimal_precision": True,
            "scoring_format": "XX.XX%",
            "score_correction_enabled": True
        }


# Main functions for compatibility
def check_ollama_installation(base_dir: str = None) -> Dict[str, Any]:
    """
    Fixed check_ollama_installation function
    Actually checks local compliance engine, not Ollama
    """
    checker = EngineHealthChecker(base_dir)
    return checker.get_compact_status()


def check_compliance_engine_online(base_dir: str = None) -> Dict[str, Any]:
    """Full health check with detailed information"""
    checker = EngineHealthChecker(base_dir)
    return checker.check_engine_installation()


def get_engine_status_summary(base_dir: str = None) -> str:
    """Get human-readable status summary"""
    checker = EngineHealthChecker(base_dir)
    status = checker.check_engine_installation()

    status_emoji = {
        "operational": "✅",
        "warning": "⚠️",
        "degraded": "⚠️",
        "error": "❌",
        "critical": "🚫",
        "unknown": "❓"
    }

    emoji = status_emoji.get(status["status"], "❓")

    summary = f"{emoji} Status: {status['status'].upper()}\n"
    summary += f"Engine: {'Running' if status['running'] else 'Not Running'}\n"
    summary += f"Data Files: {len(status['data_files']['valid'])}/{status['data_files']['total_required']} valid\n"

    if status["warnings"]:
        summary += "\nWarnings:\n"
        for warning in status["warnings"]:
            summary += f"  • {warning}\n"

    if status["recommendations"]:
        summary += "\nRecommendations:\n"
        for rec in status["recommendations"]:
            summary += f"  • {rec}\n"

    return summary


# Backward compatibility - export what was in old version
__all__ = [
    'check_ollama_installation',
    'check_compliance_engine_online',
    'get_engine_status_summary',
    'EngineHealthChecker'
]


if __name__ == "__main__":
    """Run diagnostics when executed directly"""
    print("=" * 80)
    print("LexAI Compliance Engine - Online Check & Diagnostics")
    print("=" * 80)

    # Determine base directory (parent of utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    print(f"\n📁 Project directory: {base_dir}")
    print(f"📁 Data directory: {os.path.join(base_dir, 'data')}")

    # Run checks
    checker = EngineHealthChecker(base_dir)

    print("\n" + "=" * 80)
    print("COMPACT STATUS (for UI)")
    print("=" * 80)
    compact = checker.get_compact_status()
    print(json.dumps(compact, indent=2))

    print("\n" + "=" * 80)
    print("DETAILED STATUS")
    print("=" * 80)
    detailed = checker.check_engine_installation()

    print(f"\n📊 Overall Status: {detailed['status'].upper()}")
    print(f"🕐 Timestamp: {detailed['timestamp']}")

    print(f"\n🔧 Engine:")
    print(f"  Installed: {detailed['installed']}")
    print(f"  Running: {detailed['running']}")
    print(f"  Type: {detailed['engine_type']}")

    if detailed['models']:
        print(f"  Models: {', '.join(detailed['models'])}")

    if detailed['features']:
        print(f"\n  Features:")
        for feature in detailed['features']:
            print(f"    • {feature}")

    print(f"\n📁 Data Files:")
    df = detailed['data_files']
    print(f"  Required: {df['total_required']}")
    print(f"  Found: {df['found']}")
    print(f"  Valid: {len(df['valid'])}")
    print(f"  Total Size: {df['total_size_mb']:.2f} MB")

    if df['missing']:
        print(f"\n  ❌ Missing ({len(df['missing'])}):")
        for file in df['missing']:
            print(f"    • {file}")

    if df['invalid']:
        print(f"\n  ⚠️  Invalid ({len(df['invalid'])}):")
        for file in df['invalid']:
            error = df['details'][file].get('error', 'Unknown error')
            print(f"    • {file}: {error}")

    if df['valid']:
        print(f"\n  ✅ Valid ({len(df['valid'])}):")
        for file in df['valid'][:5]:  # Show first 5
            details = df['details'][file]
            print(f"    • {file}: {details['size_mb']:.2f} MB, {details['records']} records")
        if len(df['valid']) > 5:
            print(f"    ... and {len(df['valid']) - 5} more")

    if detailed['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in detailed['warnings']:
            print(f"  • {warning}")

    if detailed['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in detailed['recommendations']:
            print(f"  • {rec}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(get_engine_status_summary(base_dir))

    print("=" * 80)