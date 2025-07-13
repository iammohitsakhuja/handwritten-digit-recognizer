#!/usr/bin/env python3
"""
Test Runner for Handwritten Digit Recognizer Project

This script runs all tests for the project and provides detailed output.
"""

import unittest
import sys
import time
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import test modules
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test_suite(verbosity=2, pattern="test_*.py", specific_test=None):
    """
    Run the complete test suite.

    Args:
        verbosity (int): Level of test output detail (0-2)
        pattern (str): Pattern to match test files
        specific_test (str): Run only a specific test module

    Returns:
        unittest.TestResult: Test results
    """
    # Set up test discovery
    test_dir = Path(__file__).parent

    if specific_test:
        # Run specific test module
        if not specific_test.endswith(".py"):
            specific_test += ".py"

        loader = unittest.TestLoader()
        suite = loader.discover(str(test_dir), pattern=specific_test)
    else:
        # Discover all tests
        loader = unittest.TestLoader()
        suite = loader.discover(str(test_dir), pattern=pattern)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)

    return result


def print_summary(result, start_time, end_time):
    """Print a summary of test results."""
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, "skipped") else 0
    passed = total_tests - failures - errors - skipped

    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print(f"Duration: {duration:.2f} seconds")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")

        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")

    print("=" * 70)


def list_available_tests():
    """List all available test modules."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))

    print("Available test modules:")
    for test_file in sorted(test_files):
        print(f"  - {test_file.stem}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run tests for MNIST digit recognition project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Test output verbosity level",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        default="test_*.py",
        help="Pattern to match test files",
    )

    parser.add_argument(
        "--test",
        "-t",
        type=str,
        help="Run only a specific test module (e.g., test_predict_digits)",
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List available test modules"
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Quick test mode (skip integration tests)",
    )

    return parser.parse_args()


def main():
    """Main test runner function."""
    args = parse_arguments()

    # List tests if requested
    if args.list:
        list_available_tests()
        return

    print("🧪 Handwritten Digit Recognizer - Test Suite")
    print("=" * 50)

    # Adjust pattern for quick mode
    if args.quick:
        pattern = "test_*.py"
        if args.pattern == "test_*.py":
            # Exclude integration tests in quick mode
            print("Quick mode: Skipping integration tests")
    else:
        pattern = args.pattern

    # Show test configuration
    print(f"Verbosity: {args.verbosity}")
    print(f"Pattern: {pattern}")
    if args.test:
        print(f"Specific test: {args.test}")
    if args.quick:
        print("Mode: Quick (unit tests only)")
    print()

    # Run tests
    start_time = time.time()

    try:
        if args.quick and not args.test:
            # Run unit tests only (exclude integration)
            unit_tests = [
                "test_train_mnist_model",
                "test_predict_digits",
                "test_run_training",
            ]
            results = []

            for test_module in unit_tests:
                print(f"\n--- Running {test_module} ---")
                result = run_test_suite(args.verbosity, f"{test_module}.py")
                results.append(result)

            # Combine results for summary
            total_run = sum(r.testsRun for r in results)
            total_failures = sum(len(r.failures) for r in results)
            total_errors = sum(len(r.errors) for r in results)

            # Create a summary result object
            class CombinedResult:
                def __init__(self):
                    self.testsRun = total_run
                    self.failures = []
                    self.errors = []
                    self.skipped = []
                    for r in results:
                        self.failures.extend(r.failures)
                        self.errors.extend(r.errors)
                        if hasattr(r, "skipped"):
                            self.skipped.extend(r.skipped)

                def wasSuccessful(self):
                    return len(self.failures) == 0 and len(self.errors) == 0

            result = CombinedResult()
        else:
            result = run_test_suite(args.verbosity, pattern, args.test)

        end_time = time.time()

        # Print summary
        print_summary(result, start_time, end_time)

        # Exit with appropriate code
        sys.exit(0 if result.wasSuccessful() else 1)

    except KeyboardInterrupt:
        print("\n\n❌ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
