"""Streaming SHA-256 for file content."""

from hashlib import sha256
from pathlib import Path
from typing import BinaryIO


def sha256_stream(stream: BinaryIO) -> str:
    """Hash bytes from the current position to EOF without closing the stream."""
    digest = sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path) -> str:
    """Hash the bytes of a file."""
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)
