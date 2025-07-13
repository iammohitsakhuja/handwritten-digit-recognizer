#!/usr/bin/env python3
"""
Integration tests for MNIST digit recognition project

These tests verify that different components work together correctly.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import subprocess
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIntegration(unittest.TestCase):
    """Integration tests for the MNIST project."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.project_root = Path(__file__).parent.parent

    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_script_help_commands(self):
        """Test that all scripts respond to --help flag."""
        scripts = [
            "scripts/train_mnist_model.py",
            "scripts/predict_digits.py",
            "scripts/run_training.py",
        ]

        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                try:
                    # Run script with --help flag
                    result = subprocess.run(
                        [sys.executable, str(script_path), "--help"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.project_root,
                    )

                    # Check that help was displayed (exit code 0 and output contains 'usage')
                    self.assertEqual(
                        result.returncode,
                        0,
                        f"{script} --help failed with stderr: {result.stderr}",
                    )
                    self.assertIn(
                        "usage",
                        result.stdout.lower(),
                        f"{script} --help doesn't show usage information",
                    )

                except subprocess.TimeoutExpired:
                    self.fail(f"{script} --help timed out")
                except Exception as e:
                    self.fail(f"Failed to run {script} --help: {e}")

    def test_shell_script_help(self):
        """Test that the shell script responds to --help."""
        shell_script = self.project_root / "mnist.sh"
        if shell_script.exists():
            try:
                result = subprocess.run(
                    ["bash", str(shell_script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.project_root,
                )

                self.assertEqual(
                    result.returncode,
                    0,
                    f"mnist.sh --help failed with stderr: {result.stderr}",
                )
                self.assertIn(
                    "Usage",
                    result.stdout,
                    "mnist.sh --help doesn't show usage information",
                )

            except subprocess.TimeoutExpired:
                self.fail("mnist.sh --help timed out")
            except Exception as e:
                self.fail(f"Failed to run mnist.sh --help: {e}")

    def test_predict_script_no_model(self):
        """Test predict script behavior when no model exists."""
        script_path = self.project_root / "scripts/predict_digits.py"
        if script_path.exists():
            try:
                # Run predict script with demo flag and non-existent model
                result = subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--demo",
                        "--model-path",
                        "nonexistent/model.pkl",
                        "--no-plots",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=self.project_root,
                )

                # Should fail gracefully (non-zero exit code)
                self.assertNotEqual(
                    result.returncode,
                    0,
                    "predict_digits.py should fail when model doesn't exist",
                )

            except subprocess.TimeoutExpired:
                self.fail("predict_digits.py timed out")
            except Exception as e:
                self.fail(f"Failed to run predict_digits.py: {e}")

    def test_project_structure(self):
        """Test that required high-level project structure exists."""
        # Check for essential directories
        required_directories = [
            "scripts",
            "src",
            "tests",
            "docs",
        ]

        for dir_name in required_directories:
            dir_path = self.project_root / dir_name
            self.assertTrue(
                dir_path.exists() and dir_path.is_dir(),
                f"Required directory {dir_name} is missing",
            )

        # Check that scripts directory contains Python files
        scripts_dir = self.project_root / "scripts"
        if scripts_dir.exists():
            python_files = list(scripts_dir.glob("*.py"))
            self.assertGreater(
                len(python_files), 0, "scripts directory should contain Python files"
            )

        # Check that src directory contains Python package
        src_dir = self.project_root / "src"
        if src_dir.exists():
            python_packages = [
                d
                for d in src_dir.iterdir()
                if d.is_dir() and (d / "__init__.py").exists()
            ]
            self.assertGreater(
                len(python_packages),
                0,
                "src directory should contain at least one Python package",
            )

        # Check that tests directory contains test files
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            self.assertGreater(
                len(test_files), 0, "tests directory should contain test files"
            )

        # Check for essential top-level files
        essential_files = [
            "requirements.txt",
            "mnist.sh",
        ]

        for file_name in essential_files:
            file_path = self.project_root / file_name
            self.assertTrue(
                file_path.exists(), f"Essential file {file_name} is missing"
            )

    def test_scripts_are_executable(self):
        """Test that Python scripts have executable permissions."""
        scripts = [
            "scripts/train_mnist_model.py",
            "scripts/predict_digits.py",
            "scripts/run_training.py",
            "mnist.sh",
        ]

        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                # Check if file is executable
                self.assertTrue(
                    os.access(script_path, os.X_OK), f"{script} is not executable"
                )

    def test_imports_work(self):
        """Test that all modules can be imported without errors."""
        # Test that we can import the main modules
        try:
            import scripts.train_mnist_model
            import scripts.predict_digits
            import scripts.run_training
        except ImportError as e:
            self.fail(f"Failed to import modules: {e}")

    def test_requirements_file_format(self):
        """Test that requirements.txt is properly formatted."""
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            with open(req_file, "r") as f:
                lines = f.readlines()

            # Should have some requirements
            self.assertGreater(len(lines), 0, "requirements.txt is empty")

            # Check for essential packages
            content = "".join(lines).lower()
            essential_packages = ["torch", "fastai", "matplotlib", "numpy"]

            for package in essential_packages:
                self.assertIn(
                    package,
                    content,
                    f"Essential package {package} not found in requirements.txt",
                )


class TestEnvironmentSetup(unittest.TestCase):
    """Test environment setup and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent

    def test_python_version_file(self):
        """Test .python-version file exists and is readable."""
        version_file = self.project_root / ".python-version"
        if version_file.exists():
            with open(version_file, "r") as f:
                version = f.read().strip()

            # Should be a valid version format (e.g., "3.13.5")
            parts = version.split(".")
            self.assertGreaterEqual(len(parts), 2, "Invalid Python version format")

            # First part should be 3 (Python 3.x)
            self.assertEqual(parts[0], "3", "Should use Python 3")

    def test_envrc_file(self):
        """Test .envrc file exists and contains virtual environment setup."""
        envrc_file = self.project_root / ".envrc"
        if envrc_file.exists():
            with open(envrc_file, "r") as f:
                content = f.read()

            # Should contain virtual environment activation
            self.assertIn(
                "venv", content.lower(), ".envrc should set up virtual environment"
            )
            self.assertIn(
                "activate", content.lower(), ".envrc should activate environment"
            )


if __name__ == "__main__":
    unittest.main()
