from pathlib import Path

from src.path_security import resolve_within_base


def test_resolve_allows_paths_within_workspace(tmp_path: Path):
    child = tmp_path / "nested" / "file.txt"
    resolved = resolve_within_base(tmp_path, child.relative_to(tmp_path))
    assert resolved == child.resolve()


def test_resolve_rejects_parent_escape(tmp_path: Path):
    try:
        resolve_within_base(tmp_path, Path('..') / 'outside.txt')
    except PermissionError:
        return
    raise AssertionError("Expected PermissionError for parent-path escape")
