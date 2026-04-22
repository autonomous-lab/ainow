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
    """Resolve rel_path under base and reject path traversal.

    Raises PermissionError with an actionable message that tells the model
    what the allowed base is and suggests a valid alternative under it, so
    small models don't spin retrying the same out-of-sandbox path.
    """
    base = Path(base).resolve()
    target = Path(rel_path)
    if not target.is_absolute():
        target = base / target
    target = target.resolve()
    if not is_within_base(base, target):
        # Suggest a safe alternative: preserve the original filename, put it
        # at the root of the workspace. The model often just wants *a* path
        # to write to; it rarely cares about the parent directory.
        suggested = (base / Path(str(rel_path)).name).as_posix()
        raise PermissionError(
            f"Path escapes workspace: '{rel_path}' is outside your allowed "
            f"working directory. You can only read/write inside '{base}'. "
            f"Retry with a relative path (e.g. '{Path(str(rel_path)).name}') "
            f"or an absolute path under that directory (e.g. '{suggested}')."
        )
    return target
