"""
test_project_integrity.py
Tests for project configuration: requirements.txt, .gitignore, file structure.
"""
import os
import sys
import pytest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


class TestRequirementsTxt:
    def _parse_requirements(self, root):
        path = os.path.join(root, "requirements.txt")
        assert os.path.isfile(path), "requirements.txt not found"
        packages = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # strip version specifier — get bare package name
                    pkg = line.split(">=")[0].split("==")[0].split("<=")[0].strip().lower()
                    packages.append(pkg)
        return packages

    def test_pandas_listed(self, root):
        assert "pandas" in self._parse_requirements(root)

    def test_numpy_listed(self, root):
        assert "numpy" in self._parse_requirements(root)

    def test_scikit_learn_listed(self, root):
        pkgs = self._parse_requirements(root)
        assert "scikit-learn" in pkgs

    def test_joblib_listed(self, root):
        assert "joblib" in self._parse_requirements(root)

    def test_streamlit_listed(self, root):
        assert "streamlit" in self._parse_requirements(root)

    def test_plotly_listed(self, root):
        assert "plotly" in self._parse_requirements(root)

    def test_shap_listed(self, root):
        assert "shap" in self._parse_requirements(root)

    def test_matplotlib_listed(self, root):
        assert "matplotlib" in self._parse_requirements(root)

    def test_seaborn_listed(self, root):
        assert "seaborn" in self._parse_requirements(root)

    def test_no_duplicate_packages(self, root):
        pkgs = self._parse_requirements(root)
        assert len(pkgs) == len(set(pkgs)), (
            f"Duplicate packages in requirements.txt: "
            f"{[p for p in pkgs if pkgs.count(p) > 1]}"
        )


class TestGitignore:
    def _read_gitignore(self, root):
        path = os.path.join(root, ".gitignore")
        assert os.path.isfile(path), ".gitignore not found"
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_venv_excluded(self, root):
        content = self._read_gitignore(root)
        assert ".venv/" in content or "venv/" in content, \
            ".gitignore must exclude virtual environment directories"

    def test_pycache_excluded(self, root):
        content = self._read_gitignore(root)
        assert "__pycache__" in content

    def test_pyc_files_excluded(self, root):
        content = self._read_gitignore(root)
        assert "*.pyc" in content

    def test_env_secrets_excluded(self, root):
        content = self._read_gitignore(root)
        assert ".env" in content, ".gitignore must exclude .env secrets file"

    def test_kaggle_json_excluded(self, root):
        content = self._read_gitignore(root)
        assert "kaggle.json" in content

    def test_model_joblib_excluded(self, root):
        """Large model binaries should be excluded from git."""
        content = self._read_gitignore(root)
        assert "*.joblib" in content, \
            ".gitignore must exclude *.joblib model files"

    def test_ds_store_excluded(self, root):
        content = self._read_gitignore(root)
        assert ".DS_Store" in content


class TestProjectStructure:
    """Verify all required source and page files exist in the correct locations."""
    REQUIRED_FILES = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
        "src/prediction.py",
        "src/setup_model.py",
        "src/train_model.py",
        "src/data_preprocessing.py",
        "src/eda.py",
        "src/inspect_dataset.py",
        "pages/1_Marine_Water_Quality_Prediction.py",
        "pages/2_Data_Analytics.py",
        "pages/3_Model_Performance.py",
        "pages/4_About_Project.py",
        "data/marine_water_quality.csv",
        "data/split_metadata.json",
    ]

    @pytest.mark.parametrize("relative_path", REQUIRED_FILES)
    def test_required_file_exists(self, root, relative_path):
        full_path = os.path.join(root, relative_path.replace("/", os.sep))
        assert os.path.isfile(full_path), (
            f"Required file missing: {relative_path}"
        )

    def test_pages_directory_exists(self, root):
        assert os.path.isdir(os.path.join(root, "pages"))

    def test_src_directory_exists(self, root):
        assert os.path.isdir(os.path.join(root, "src"))

    def test_models_directory_exists(self, root):
        assert os.path.isdir(os.path.join(root, "models"))

    def test_reports_directory_exists(self, root):
        assert os.path.isdir(os.path.join(root, "reports"))

    def test_data_directory_exists(self, root):
        assert os.path.isdir(os.path.join(root, "data"))


class TestSyntax:
    """All Python source and page files must be syntactically valid."""
    import ast, pathlib

    @staticmethod
    def _py_files(root):
        import pathlib
        files = []
        for pattern in ["*.py", "src/*.py", "pages/*.py", "tests/*.py"]:
            files.extend(pathlib.Path(root).glob(pattern))
        return files

    def test_all_python_files_parse(self, root):
        import ast, pathlib
        errors = []
        for p in self._py_files(root):
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as e:
                errors.append(f"{p.name} line {e.lineno}: {e.msg}")
        assert not errors, "Syntax errors found:\n" + "\n".join(errors)
