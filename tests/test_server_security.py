from pathlib import Path

from src.path_security import is_within_base


def test_is_within_base_accepts_real_child(tmp_path: Path):
    base = tmp_path / "agent"
    child = base / "docs" / "note.txt"
    child.parent.mkdir(parents=True)
    child.write_text("ok", encoding="utf-8")
    assert is_within_base(base, child)


def test_is_within_base_rejects_prefix_sibling(tmp_path: Path):
    base = tmp_path / "agent"
    sibling = tmp_path / "agent-evil" / "note.txt"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("oops", encoding="utf-8")
    assert not is_within_base(base, sibling)
