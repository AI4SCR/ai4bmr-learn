from __future__ import annotations

from pathlib import Path


def set_permissions(path: str | Path, permission: int, recursive: bool = True) -> None:
    """Set filesystem permissions on a path.

    Args:
        path: File or directory path.
        permission: Unix mode bits, e.g. 0o770.
        recursive: Apply to all children when `path` is a directory.
    """
    target = Path(path).expanduser().resolve()
    assert target.exists(), f"Path does not exist: {target}"
    assert permission >= 0, "permission must be >= 0"

    if target.is_dir() and recursive:
        for child in target.rglob("*"):
            child.chmod(permission)
    target.chmod(permission)
