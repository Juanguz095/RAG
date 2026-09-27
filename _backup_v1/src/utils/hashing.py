import hashlib
from pathlib import Path


def compute_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()


def compute_file_hash_from_path(file_path: str | Path) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
