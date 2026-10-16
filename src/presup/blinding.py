"""The user turn with the claim sentence redacted, for the blinded challenge judge.

Whether a response reads as a challenge depends on how firmly the user put the claim, so every
response is also judged with the claim sentence replaced by a token (chal_blind_*). Within an
item x format the redacted turn is then the same for NO_CLAIM and the four NAI_* cells, and for
the four AI_* cells: the judge can't see the framing but still sees the request.
"""

from __future__ import annotations

from collections import defaultdict

# the judge template doesn't explain the redaction, so the token says it itself
REDACTION_TOKEN = "[The user's own statement about the claim has been removed.]"

BLIND_COLUMNS = ("context_sentence", "request")


def terminal_punctuation(record: dict) -> str:
    """The mark that ends the user's turn ("?" for conv, "." for instr)."""
    text = (record.get("prompt_text") or "").strip()
    if text and text[-1] in ".?!":
        return text[-1]
    return "?" if record.get("format") == "conv" else "."


def redacted_user_turn(record: dict) -> str:
    """Context sentence, redaction token, request.

    Rebuilt from the context_sentence and request columns rather than by cutting prompt_text.
    """
    missing = [c for c in BLIND_COLUMNS if not str(record.get(c) or "").strip()]
    if missing:
        raise SystemExit(
            f"cannot build the blinded user turn for {record.get('item_id')} "
            f"{record.get('format')} {record.get('condition')}: missing {missing}.\n"
            "The blinded judge needs the context_sentence and request columns."
        )
    # NO_CLAIM gets the token too, so its judge prompt matches the NAI cells
    context = str(record["context_sentence"]).strip().rstrip(".")
    request = str(record["request"]).strip().rstrip(".?!")
    return f"{context}. {REDACTION_TOKEN} {request}{terminal_punctuation(record)}"


def check_blinding(records: list[dict]) -> list[str]:
    """List of failures, empty if the blinding holds.

    Within an item x format the redacted turn should be the same for every cell that shares a
    request, and differ between the NAI and AI requests.
    """
    failures: list[str] = []
    blocks: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for record in records:
        blocks[(record["item_id"], record["format"])][record["condition"]] = record

    for (item, fmt), block in sorted(blocks.items()):
        groups: dict[str, dict[str, str]] = defaultdict(dict)
        for condition, record in block.items():
            groups[str(record["request"]).strip()][condition] = redacted_user_turn(record)
        for request, turns in groups.items():
            distinct = set(turns.values())
            if len(distinct) > 1:
                failures.append(
                    f"{item} {fmt}: the redacted turn differs within one request group "
                    f"({sorted(turns)}), so encoding is not blinded:\n      "
                    + "\n      ".join(sorted(distinct))
                )
        if len(groups) < 2:
            failures.append(
                f"{item} {fmt}: only {len(groups)} distinct request group(s); the AI and NAI "
                "requests must differ or at-issueness is blinded away too"
            )
        collapsed = {next(iter(turns.values())) for turns in groups.values()}
        if len(collapsed) != len(groups):
            failures.append(
                f"{item} {fmt}: two request groups produced the same redacted turn, so the "
                "at-issueness factor is invisible to the blinded judge"
            )
    return failures


def blinding_summary(records: list[dict]) -> str:
    """One line: how many prompts collapse to how many distinct blinded turns."""
    blocks: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in records:
        blocks[(record["item_id"], record["format"])].add(redacted_user_turn(record))
    cells = len(records)
    distinct = sum(len(turns) for turns in blocks.values())
    return (f"{cells} prompts collapse to {distinct} distinct blinded user turns across "
            f"{len(blocks)} item x format blocks "
            f"({distinct / max(len(blocks), 1):.0f} per block = the at-issueness levels)")
