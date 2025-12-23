#!/bin/bash
# entrypoint.sh - LexAI startup script with Ollama support
set -euo pipefail

# ============================================================================
# CONFIGURATION
# ============================================================================
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

# Colors for logs
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# ============================================================================
# OLLAMA CONNECTION CHECK
# ============================================================================
check_ollama_connection() {
    log_step "Checking Ollama LLM connection..."

    local ollama_host="${OLLAMA_HOST:-http://localhost:11434}"
    local ollama_model="${OLLAMA_MODEL:-llama3.2:3b}"

    log_info "Ollama Host: $ollama_host"
    log_info "Ollama Model: $ollama_model"

    # Check if USE_LLM_ANALYZER is enabled
    if [[ "${USE_LLM_ANALYZER:-false}" != "true" ]]; then
        log_info "LLM analyzer disabled - skipping Ollama check"
        return 0
    fi

    # Test Ollama connectivity with retry
    local max_retries=3
    local retry=0

    while [[ $retry -lt $max_retries ]]; do
        if curl -sf --connect-timeout 5 "$ollama_host/api/tags" > /dev/null 2>&1; then
            log_info "Ollama API is reachable"

            # Check if model is available
            local response=$(curl -sf "$ollama_host/api/tags" 2>/dev/null)
            if echo "$response" | grep -q "models"; then
                log_info "Ollama is responding with model list"

                # Test if configured model exists
                if echo "$response" | grep -q "$ollama_model"; then
                    log_info "Model '$ollama_model' is available"
                    export LLM_AVAILABLE=true
                    return 0
                else
                    log_warn "Model '$ollama_model' not found"
                    log_info "Pull it with: ollama pull $ollama_model"
                fi
            fi
            break
        else
            retry=$((retry + 1))
            if [[ $retry -lt $max_retries ]]; then
                log_warn "Ollama not responding, retry $retry/$max_retries..."
                sleep 2
            fi
        fi
    done

    log_warn "Cannot reach Ollama at $ollama_host"
    log_info "Troubleshooting tips:"
    log_info "  1. Ensure Ollama is running: systemctl status ollama"
    log_info "  2. Check if Ollama listens on all interfaces:"
    log_info "     Edit /etc/systemd/system/ollama.service"
    log_info "     Add: Environment=\"OLLAMA_HOST=0.0.0.0\""
    log_info "  3. Verify from host: curl http://localhost:11434/api/tags"

    export LLM_AVAILABLE=false
    return 1
}

# ============================================================================
# LOCAL ENGINE SETUP
# ============================================================================
setup_local_engine() {
    log_step "Setting up Local Compliance Engine..."

    # Check if engine.py exists
    if [[ -f "/app/engine.py" ]]; then
        log_info "engine.py found"

        # Test local engine with timeout to prevent hanging
        if timeout 10 python -c "
import sys
sys.path.insert(0, '/app')
try:
    from engine import LocalComplianceEngine
    engine = LocalComplianceEngine('/app/data')
    print(f'Local engine initialized with {len(engine.data_cache)} data files')
except Exception as e:
    print(f'Local engine error: {e}')
    exit(1)
" 2>/dev/null; then
            log_info "Local compliance engine operational"
            export USE_LOCAL_ENGINE=true
            export COMPLIANCE_ENGINE=local
            return 0
        else
            log_warn "Local engine test failed - using fallback"
        fi
    else
        log_warn "engine.py not found - using fallback analysis"
    fi

    # Fallback mode
    export USE_LOCAL_ENGINE=false
    export COMPLIANCE_ENGINE=fallback
    return 1
}

# ============================================================================
# DATA FILES VERIFICATION
# ============================================================================
verify_data_files() {
    log_step "Verifying compliance data files..."

    local data_dir="/app/data"

    # Check if data directory exists
    if [[ ! -d "$data_dir" ]]; then
        log_warn "Data directory not found: $data_dir"
        log_info "Will use basic fallback analysis"
        return 0
    fi

    # Count JSON files
    local json_count=0
    if ls "$data_dir"/*.json >/dev/null 2>&1; then
        json_count=$(ls "$data_dir"/*.json 2>/dev/null | wc -l)
    fi

    log_info "JSON data files found: $json_count"

    # List some files for verification
    if [[ $json_count -gt 0 ]]; then
        log_info "Sample data files:"
        ls "$data_dir"/*.json 2>/dev/null | head -3 | while read -r file; do
            filename=$(basename "$file")
            size=$(stat -c%s "$file" 2>/dev/null || echo "?")
            log_info "   - $filename ($size bytes)"
        done
    fi

    if [[ $json_count -ge 3 ]]; then
        log_info "Sufficient data files for compliance analysis"
    else
        log_warn "Limited data files - will use basic analysis"
    fi

    return 0
}

# ============================================================================
# APPLICATION VERIFICATION
# ============================================================================
verify_app() {
    log_step "Verifying application components..."

    # Check critical files
    local critical_files=("/app/streamlit_app.py")
    for file in "${critical_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Critical file missing: $file"
            exit 1
        fi
    done

    # Test Python imports with timeout
    log_info "Testing core imports..."
    if timeout 15 python -c "
import sys
sys.path.insert(0, '/app')
try:
    import streamlit
    print('Streamlit OK')

    # Test local engine import (non-blocking)
    try:
        from engine import LocalComplianceEngine
        print('Local engine import OK')
    except ImportError:
        print('Local engine not available - will use fallback')
    except Exception as e:
        print(f'Local engine issue: {e}')

    print('Core application ready')
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
" 2>/dev/null; then
        log_info "Application verification complete"
        return 0
    else
        log_warn "Some imports failed - continuing with fallbacks"
        return 0
    fi
}

# ============================================================================
# RUNTIME SETUP
# ============================================================================
setup_runtime() {
    log_step "Setting up runtime environment..."

    # Create essential directories
    mkdir -p /app/logs /app/data

    # Set essential Streamlit configuration
    export STREAMLIT_SERVER_ADDRESS=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
    export STREAMLIT_SERVER_PORT=${STREAMLIT_SERVER_PORT:-8501}
    export STREAMLIT_SERVER_HEADLESS=${STREAMLIT_SERVER_HEADLESS:-true}
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

    # Set application variables
    export PYTHONPATH=/app
    export APP_VERSION=${APP_VERSION:-2.0.0}
    export DATA_DIR=${DATA_DIR:-/app/data}

    log_info "Runtime configured"
}

# ============================================================================
# CLEANUP HANDLER
# ============================================================================
cleanup() {
    log_info "Shutting down LexAI..."
    exit 0
}

# ============================================================================
# MAIN SCRIPT
# ============================================================================
main() {
    log_step "Starting LexAI Compliance Analysis Platform"
    echo "========================================================"

    # Display basic system info
    log_info "Python: $(python --version 2>&1)"
    log_info "Working directory: $(pwd)"

    # Setup runtime environment
    setup_runtime

    # Verify application
    verify_app

    # Verify data files (non-blocking)
    verify_data_files

    # Check Ollama connection (non-blocking)
    if check_ollama_connection; then
        log_info "Mode: LLM-powered analysis (Ollama)"
    else
        # Setup local compliance engine (non-blocking)
        if setup_local_engine; then
            log_info "Mode: Local Compliance Engine"
        else
            log_info "Mode: Fallback Analysis"
        fi
    fi

    # Setup signal handlers
    trap cleanup SIGTERM SIGINT

    # Display connection info
    log_step "Starting Streamlit server..."
    log_info "Web Interface: http://localhost:${STREAMLIT_SERVER_PORT}"
    log_info "Ollama Host: ${OLLAMA_HOST:-not configured}"
    log_info "Engine Type: ${COMPLIANCE_ENGINE:-fallback}"

    echo "========================================================"
    log_info "Starting web interface..."

    # Start Streamlit
    exec streamlit run /app/streamlit_app.py \
        --server.address "$STREAMLIT_SERVER_ADDRESS" \
        --server.port "$STREAMLIT_SERVER_PORT" \
        --server.headless "$STREAMLIT_SERVER_HEADLESS" \
        --browser.gatherUsageStats false \
        --logger.level error
}

# Entry point
main "$@"