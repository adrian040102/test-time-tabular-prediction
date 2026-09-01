"""Canonical classification of deterministic experiment skips.

Only the four explicit emitters below represent intentional N/A outcomes.
Bag8 preserves an inner emitter verbatim behind its fixed subprocess prefix.
Unknown or embedded ``*_skip:`` text is a genuine error and must remain
retryable.
"""

from __future__ import annotations


DETERMINISTIC_SKIP_PREFIXES = (
    "tabpfn_skip:",
    "tabicl_skip:",
    "joint_tsne_skip:",
    "pseudo_label_skip:",
)
BAG8_SUBPROCESS_ERROR_PREFIX = "bag-8 subprocess error: "


def is_deterministic_skip_error(error: object) -> bool:
    """Return whether *error* starts with an approved skip emitter.

    The match is deliberately anchored.  Treating arbitrary ``_skip:``
    substrings as terminal can suppress retries for unrelated failures.
    """
    if not isinstance(error, str):
        return False
    if error.startswith(DETERMINISTIC_SKIP_PREFIXES):
        return True
    if not error.startswith(BAG8_SUBPROCESS_ERROR_PREFIX):
        return False
    inner = error[len(BAG8_SUBPROCESS_ERROR_PREFIX):]
    return inner.startswith(DETERMINISTIC_SKIP_PREFIXES)
