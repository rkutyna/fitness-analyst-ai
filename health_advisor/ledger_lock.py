"""The lock that keeps a call-ledger append invisible to a reader until it is whole.

The writer (``deepdive_mcp._CallLedger.append``) hands the kernel one complete
line in one ``write(2)`` on an ``O_APPEND`` descriptor. That orders writers
against each other, but on Linux it does not make the write atomic with respect
to a reader: a buffered ``read(2)`` of a regular file does not wait for a
``write(2)`` in progress on the same file, and returns whatever part of it has
been copied in so far. Measured on ext4, overlayfs and tmpfs (#116): an
unlocked reader saw a partial last line, always ending on a page boundary,
while the writer had issued exactly one full-length write per record. macOS
did not show it. Every ledger reader treats one undecodable line as "the whole
ledger is unreadable" (#103), so that half-line blanks every citation of the
turn.

So the append and every read of the file hold a ``flock(2)``: the writer an
exclusive lock for the duration of its write, a reader a shared one for the
duration of its read. ``flock`` locks belong to the open file description, so
they exclude threads in one process as well as the separate MCP server process
that writes the ledger on the codex path. Closing the file releases the lock.
"""
from __future__ import annotations

import fcntl


def lock_for_append(fd: int) -> None:
    """Block until no reader or other writer holds the ledger at ``fd``."""
    fcntl.flock(fd, fcntl.LOCK_EX)


def lock_for_read(file) -> None:
    """Block until no append to the open ledger ``file`` is in progress.

    Call it before the first read; the lock lasts until the file is closed.
    """
    fcntl.flock(file.fileno(), fcntl.LOCK_SH)
