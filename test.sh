#!/usr/bin/env bash

# Test automation script for MNIST digit recognition project
# Provides various testing commands

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment is active
check_venv() {
    if [[ -z "$VIRTUAL_ENV" ]] && [[ ! -d ".venv" ]]; then
        print_warning "No virtual environment detected."
        print_warning "Consider running: source .envrc"
        echo ""
    fi
}

# Install test dependencies
install_deps() {
    print_header "Installing Test Dependencies"

    if command -v pip &> /dev/null; then
        pip install -q pytest pytest-cov
        print_success "Test dependencies installed"
    else
        print_error "pip not found. Please ensure Python is properly installed."
        exit 1
    fi
}

# Run unit tests only
test_unit() {
    print_header "Running Unit Tests"
    python run_tests.py --quick
}

# Run integration tests only
test_integration() {
    print_header "Running Integration Tests"
    python run_tests.py --test test_integration
}

# Run all tests
test_all() {
    print_header "Running All Tests"
    python run_tests.py
}

# Run tests with pytest
test_pytest() {
    print_header "Running Tests with Pytest"

    if command -v pytest &> /dev/null; then
        pytest tests/ -v
    else
        print_warning "pytest not installed. Installing..."
        install_deps
        pytest tests/ -v
    fi
}

# Run tests with coverage
test_coverage() {
    print_header "Running Tests with Coverage"

    if command -v pytest &> /dev/null; then
        pytest tests/ --cov=. --cov-report=html --cov-report=term
        print_success "Coverage report generated in htmlcov/"
    else
        print_warning "pytest not installed. Installing..."
        install_deps
        pytest tests/ --cov=. --cov-report=html --cov-report=term
    fi
}

# Quick syntax check
test_syntax() {
    print_header "Checking Python Syntax"

    python_files=($(find . -maxdepth 1 -name "*.py"))

    if [[ ${#python_files[@]} -eq 0 ]]; then
        print_warning "No Python files found for syntax check"
        return
    fi

    for file in "${python_files[@]}"; do
        if python -m py_compile "$file"; then
            print_success "$file syntax OK"
        else
            print_error "$file has syntax errors"
            exit 1
        fi
    done
}

# Check code style (if available)
test_style() {
    print_header "Checking Code Style"

    if command -v flake8 &> /dev/null; then
        flake8 *.py tests/ --max-line-length=100 --ignore=E501,W503
        print_success "Code style check passed"
    elif command -v pycodestyle &> /dev/null; then
        pycodestyle *.py tests/ --max-line-length=100
        print_success "Code style check passed"
    else
        print_warning "No code style checker found (flake8 or pycodestyle)"
        print_warning "Skipping style check"
    fi
}

# Clean up test artifacts
clean() {
    print_header "Cleaning Test Artifacts"

    # Remove pytest cache
    rm -rf .pytest_cache
    rm -rf tests/__pycache__
    rm -rf __pycache__

    # Remove coverage files
    rm -rf htmlcov/
    rm -f .coverage

    # Remove temporary test files
    find . -name "*.pyc" -delete
    find . -name "*~" -delete

    print_success "Test artifacts cleaned"
}

# Show test status/summary
status() {
    print_header "Test Environment Status"

    echo "Project Directory: $PROJECT_DIR"
    echo "Python Version: $(python --version 2>&1)"
    echo "Virtual Environment: ${VIRTUAL_ENV:-"Not activated"}"
    echo ""

    echo "Available test files:"
    for test_file in tests/test_*.py; do
        if [[ -f "$test_file" ]]; then
            echo "  - $(basename "$test_file")"
        fi
    done
    echo ""

    echo "Testing tools:"
    if command -v pytest &> /dev/null; then
        echo "  ✅ pytest: $(pytest --version 2>&1 | head -1)"
    else
        echo "  ❌ pytest: Not installed"
    fi

    if python -c "import coverage" 2>/dev/null; then
        echo "  ✅ coverage: Available"
    else
        echo "  ❌ coverage: Not available"
    fi
}

# Show help
show_help() {
    echo "Test automation script for MNIST digit recognition project"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  unit          Run unit tests only (fast)"
    echo "  integration   Run integration tests only"
    echo "  all           Run all tests"
    echo "  pytest        Run tests using pytest"
    echo "  coverage      Run tests with coverage report"
    echo "  syntax        Check Python syntax"
    echo "  style         Check code style"
    echo "  install       Install test dependencies"
    echo "  clean         Clean test artifacts"
    echo "  status        Show test environment status"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 unit           # Quick unit tests"
    echo "  $0 coverage       # Full test coverage"
    echo "  $0 syntax         # Just check syntax"
}

# Main command handling
main() {
    # Check environment
    check_venv

    case "${1:-help}" in
        "unit")
            test_unit
            ;;
        "integration")
            test_integration
            ;;
        "all")
            test_all
            ;;
        "pytest")
            test_pytest
            ;;
        "coverage")
            test_coverage
            ;;
        "syntax")
            test_syntax
            ;;
        "style")
            test_style
            ;;
        "install")
            install_deps
            ;;
        "clean")
            clean
            ;;
        "status")
            status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
