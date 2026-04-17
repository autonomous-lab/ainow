from pathlib import Path


def is_within_base(base: Path, target: Path) -> bool:
    """Return True when target stays inside base after resolution."""
    base = Path(base).resolve()
    target = Path(target).resolve()
    try:
        return target.is_relative_to(base)
    except AttributeError:
        try:
            from os.path import commonpath
            return commonpath([str(base), str(target)]) == str(base)
        except ValueError:
            return False


def resolve_within_base(base: Path, rel_path: str | Path) -> Path:
    """Resolve rel_path under base and reject path traversal."""
    base = Path(base).resolve()
    target = Path(rel_path)
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    if not is_within_base(base, target):
        raise PermissionError(f"Path escapes workspace: {rel_path}")
    return target
