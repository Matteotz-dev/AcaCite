import subprocess
from pathlib import Path

from app.ingestion.repository import discover_repository


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_repository_discovery_honors_gitignore_and_reports_state(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (root / "tracked.py").write_text("def tracked():\n    pass\n", encoding="utf-8")
    (root / "ignored.py").write_text("secret = True\n", encoding="utf-8")
    git(root, "add", ".gitignore", "tracked.py")
    git(root, "commit", "-qm", "fixture")
    snapshot = discover_repository(root, (tmp_path,))
    assert snapshot.commit
    assert snapshot.branch
    assert snapshot.dirty is False
    assert tuple(path.name for path in snapshot.files) == ("tracked.py",)

    (root / "new.md").write_text("# New\n", encoding="utf-8")
    assert discover_repository(root, (tmp_path,)).dirty is True
