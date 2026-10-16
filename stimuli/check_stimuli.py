"""Checks the design constraints on stimuli/stimuli.csv and exits non-zero if any fail.

Runs on the written CSV rather than the item definitions in make_stimuli.py, so hand edits are
checked too. See make_stimuli.py for the cell layout.

    uv run python stimuli/check_stimuli.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

from presup.blinding import blinding_summary, check_blinding
from presup.stimuli import load_stimuli

NAI = ("NAI_ASSERT", "NAI_HEDGE", "NAI_COMMIT", "NAI_PRESUP")
AI = ("AI_ASSERT", "AI_HEDGE", "AI_COMMIT", "AI_PRESUP")
CLAIM_CELLS = NAI + AI
# a directive to restate the claim would make the model repeat it regardless of framing
BANNED_DIRECTIVES = ("mention", "highlight", "emphasis", "emphasiz", "stress", "note that")
BANNED_PREDICATES = ("know", "knew", "known", "given that", "realis", "realiz")

failures: list[str] = []
warnings: list[str] = []


def fail(test: str, detail: str) -> None:
    failures.append(f"{test}: {detail}")


def warn(test: str, detail: str) -> None:
    warnings.append(f"{test}: {detail}")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def main() -> None:
    records = load_stimuli()

    blocks: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for r in records:
        blocks[(r["item_id"], r["format"])][r["condition"]] = r

    # F1 every block has all nine cells
    for (item, fmt), block in blocks.items():
        missing = set(("NO_CLAIM",) + CLAIM_CELLS) - set(block)
        if missing:
            fail("F1 complete block", f"{item} {fmt} missing {sorted(missing)}")

    for (item, fmt), block in sorted(blocks.items()):
        if set(("NO_CLAIM",) + CLAIM_CELLS) - set(block):
            continue
        parts = {c: sentences(block[c]["prompt_text"]) for c in block}

        # F2 context sentence identical across all nine cells
        contexts = {c: p[0] for c, p in parts.items()}
        if len(set(contexts.values())) != 1:
            fail("F2 shared context", f"{item} {fmt}: {sorted(set(contexts.values()))}")

        # F3 request byte-identical within at-issueness level, and NO_CLAIM matches the NAI request
        nai_reqs = {c: parts[c][-1] for c in NAI}
        if len(set(nai_reqs.values())) != 1:
            fail("F3 NAI request identical", f"{item} {fmt}: {sorted(set(nai_reqs.values()))}")
        if parts["NO_CLAIM"][-1] != parts["NAI_ASSERT"][-1]:
            fail("F3b anchor request", f"{item} {fmt}: anchor request differs from the NAI request")
        ai_reqs = {c: parts[c][-1] for c in AI}
        if len(set(ai_reqs.values())) != 1:
            fail("F4 AI request identical", f"{item} {fmt}: {sorted(set(ai_reqs.values()))}")
        if set(nai_reqs.values()) == set(ai_reqs.values()):
            fail("F5 at-issueness varies", f"{item} {fmt}: AI and NAI requests are the same string")

        # F6 the framed middles differ from the asserted one only in their prefix
        middles = {c: parts[c][1] for c in CLAIM_CELLS}
        claim = middles["NAI_ASSERT"]
        for cell in ("NAI_HEDGE", "NAI_COMMIT", "NAI_PRESUP"):
            m = middles[cell]
            # case-insensitive: claim_clause keeps the capital on a proper-noun subject
            if not m.lower().endswith(claim.lower()):
                fail("F6 prefix-only difference",
                     f"{item} {fmt} {cell}: does not end with the asserted middle, so something "
                     f"other than the frame changed\n"
                     f"      asserted: {claim}\n      {cell:11}: {m}")
                continue
            # what precedes it must be a frame, i.e. end in "that "
            prefix = m[: len(m) - len(claim)]
            if not prefix.endswith("that "):
                fail("F6b frame shape",
                     f"{item} {fmt} {cell}: prefix {prefix!r} does not end in 'that '")
        # the at-issue middle must equal its not-at-issue twin
        for ai_cell, nai_cell in zip(AI, NAI):
            if middles[ai_cell] != middles[nai_cell]:
                fail("F7 AI/NAI middles match",
                     f"{item} {fmt}: {ai_cell} != {nai_cell}")

        # F8 the anchor has no claim sentence
        if len(parts["NO_CLAIM"]) != 2:
            fail("F8 anchor has no claim", f"{item} {fmt}: anchor has {len(parts['NO_CLAIM'])} sentences")

        # F9 no banned directive verb anywhere in a presupposed cell
        for cell in ("NAI_PRESUP", "AI_PRESUP"):
            low = middles[cell].lower()
            for bad in BANNED_DIRECTIVES:
                if bad in low:
                    fail("F9 no directive verb", f"{item} {fmt} {cell}: contains {bad!r}")

        # F10 no banned predicate (know / given / realise) in any framed cell
        for cell in ("NAI_HEDGE", "NAI_COMMIT", "NAI_PRESUP"):
            low = middles[cell].lower()
            for bad in BANNED_PREDICATES:
                if bad in low:
                    fail("F10 no banned predicate", f"{item} {fmt} {cell}: contains {bad!r}")

        # F11 the three framed cells carry the frame the design says they do
        if not re.match(r"^I think that ", middles["NAI_HEDGE"]):
            fail("F11 hedge frame", f"{item} {fmt}: {middles['NAI_HEDGE'][:48]!r}")
        if not re.match(r"^I'm certain that ", middles["NAI_COMMIT"]):
            fail("F11 commit frame", f"{item} {fmt}: {middles['NAI_COMMIT'][:48]!r}")
        if " that " not in middles["NAI_PRESUP"][:40]:
            fail("F11 presup frame",
                 f"{item} {fmt}: presupposed cell has no that-complement: "
                 f"{middles['NAI_PRESUP'][:48]!r}")

        # W1 a definite in the anchor whose antecedent only the claim introduces
        for article_np in re.findall(r"\bthe ([a-z]+(?: [a-z]+)?)", parts["NO_CLAIM"][-1]):
            if article_np not in parts["NO_CLAIM"][0].lower():
                warn("W1 anchor definite", f"{item} {fmt}: 'the {article_np}' has no antecedent in ctx")

    # F12 predicate balance: no factive predicate confined to one band or domain
    by_predicate: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["condition"] == "NAI_PRESUP":
            by_predicate[str(r.get("predicate"))].append(f"{r['band']}/{r.get('domain', '?')}")
    print("predicate x band/domain balance")
    for predicate, cells in sorted(by_predicate.items()):
        bands = Counter(c.split("/")[0] for c in cells)
        print(f"  {predicate:12} n={len(cells) // 2:2} bands={dict(bands)}")
        if len(bands) == 1 and len(cells) > 2:
            fail("F12 predicate balance", f"{predicate} appears in only one band: {dict(bands)}")

    # F14 the blinded judge's user turn is the same across framings (run_pilot.py checks this too)
    print("\nblinding (frame-blind challenge judge)")
    blind_failures = check_blinding([r for r in records if r.get("prompt_text")])
    print("  " + blinding_summary([r for r in records if r.get("prompt_text")]))
    for detail in blind_failures:
        fail("F14 redaction blinds encoding", detail)
    if not blind_failures:
        print("  encoding is invisible to the blinded judge; at-issueness stays visible")

    # F13 cell counts
    counts = Counter(r["condition"] for r in records)
    print(f"\ncells: {dict(counts)}")
    if len(set(counts.values())) != 1:
        fail("F13 equal cells", f"unequal cell sizes: {dict(counts)}")

    print(f"\n{len(blocks)} blocks, {len(records)} prompts")
    if warnings:
        print(f"\n{len(warnings)} warning(s) - check by hand, not fatal:")
        for w in warnings:
            print("  - " + w)
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    print("\nall invariants pass")


if __name__ == "__main__":
    main()
