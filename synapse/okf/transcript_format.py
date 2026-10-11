"""Turn-level parsing for session transcripts (contract: docs/transcript-format.md).

v2 transcripts carry the ``<!-- synapse-transcript v2 -->`` marker and number turns
``T1..Tn``; legacy v1 transcripts use plain ``## User`` / ``## Assistant`` headings
and get turn numbers assigned in order of appearance. One parser serves both, so
the distiller, the evidence verifier, and incremental distillation share a single
turn model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

V2_MARKER = "<!-- synapse-transcript v2 -->"

_V2_TURN_PATTERN = re.compile(r"^## T(\d+) (User|Assistant)\s*$")
_V1_TURN_PATTERN = re.compile(r"^## (User|Assistant)\s*$")
_TOOL_PATTERN = re.compile(r"^### Tool T(\d+)\.(\d+) (.+?)\s*$")

# Accepted turn roles.
ROLE_USER = "User"
ROLE_ASSISTANT = "Assistant"


@dataclass(slots=True)
class TranscriptTurn:
    """One numbered conversation turn with its attached tool sections."""

    number: int
    role: str  # "User" | "Assistant"
    body: str
    # Raw lines of any `### Tool T<n>.<m> <name>` blocks attached to this turn.
    tool_sections: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"T{self.number}"


@dataclass(slots=True)
class ParsedTranscript:
    """Turn-structured view of a transcript body."""

    version: int  # 2 when the v2 marker is present, else 1
    turns: list[TranscriptTurn] = field(default_factory=list)

    def turn(self, number: int) -> TranscriptTurn | None:
        for turn in self.turns:
            if turn.number == number:
                return turn
        return None

    @property
    def last_turn_number(self) -> int:
        return self.turns[-1].number if self.turns else 0

    def turns_after(self, last_distilled_turn: int) -> list[TranscriptTurn]:
        """Turns with a number strictly greater than the watermark."""

        return [turn for turn in self.turns if turn.number > last_distilled_turn]


def parse_transcript(content: str) -> ParsedTranscript:
    """Parse a transcript body into numbered turns.

    v2: headings are ``## T<n> User`` / ``## T<n> Assistant``; ``n`` is trusted
    (server-assigned by the bridge). Legacy v1: ``## User`` / ``## Assistant``
    blocks get ``T1..Tn`` in order of appearance. Level-3 ``### Tool …`` blocks
    attach to the enclosing turn. Lines before the first turn heading are
    ignored (v2 marker, preamble).
    """

    is_v2 = V2_MARKER in content
    turn_pattern = _V2_TURN_PATTERN if is_v2 else _V1_TURN_PATTERN

    parsed = ParsedTranscript(version=2 if is_v2 else 1)
    current: TranscriptTurn | None = None
    body_buffer: list[str] = []
    tool_index: int | None = None  # index into current.tool_sections being appended to

    def flush() -> None:
        nonlocal current, body_buffer, tool_index
        if current is not None:
            current.body = "\n".join(body_buffer).strip()
            parsed.turns.append(current)
        current = None
        body_buffer = []
        tool_index = None

    for line in content.splitlines():
        turn_match = turn_pattern.match(line)
        if turn_match is not None:
            flush()
            if is_v2:
                number = int(turn_match.group(1))
                role = turn_match.group(2)
            else:
                number = len(parsed.turns) + 1
                role = turn_match.group(1)
            current = TranscriptTurn(number=number, role=role, body="")
            continue
        if current is None:
            continue  # preamble before the first turn
        if _TOOL_PATTERN.match(line) is not None or (tool_index is None and line.startswith("### ")):
            current.tool_sections.append(line)
            tool_index = len(current.tool_sections) - 1
        elif tool_index is not None:
            current.tool_sections[tool_index] += "\n" + line
        else:
            body_buffer.append(line)

    flush()
    return parsed
