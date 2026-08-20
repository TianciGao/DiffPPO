from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RELEASE_COMMIT = "31dac8148a84204b9db506909edd8fb92822fcba"
RELEASE_TREE = "3ca1845dfbc46316c37236b49cee9d64ee2e3678"
EXPECTED_MAIN = "5cc5b6d74fcaa05c95cd25640036baa500e69c36"
REPOSITORY = Path(__file__).resolve().parents[3]


def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        check=check,
        capture_output=True,
        text=True,
    )


def test_release_commit_and_tree_are_exact() -> None:
    assert git("rev-parse", f"{RELEASE_COMMIT}^{{tree}}").stdout.strip() == RELEASE_TREE
    assert git("merge-base", "--is-ancestor", EXPECTED_MAIN, "HEAD", check=False).returncode == 0


@pytest.mark.parametrize("path", ["src/ppo_dap", "tests"])
def test_core_and_existing_tests_match_release_bytes(path: str) -> None:
    assert (
        git("rev-parse", f"{RELEASE_COMMIT}:{path}").stdout.strip()
        == git("rev-parse", f"HEAD:{path}").stdout.strip()
    )
    assert git("diff", "--quiet", RELEASE_COMMIT, "--", path, check=False).returncode == 0


def test_main_to_release_drift_is_public_documentation_only() -> None:
    changed = git("diff", "--name-only", RELEASE_COMMIT, EXPECTED_MAIN).stdout.splitlines()
    assert set(changed) == {
        "CITATION.cff",
        "README.md",
        "RELEASE_NOTES.md",
        "docs/ALGORITHM.md",
        "docs/IMPLEMENTATION.md",
        "docs/THEORY_CONFORMANCE.md",
        "docs/releases/v0.1.0.md",
    }
