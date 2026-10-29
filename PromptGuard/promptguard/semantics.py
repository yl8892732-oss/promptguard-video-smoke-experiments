"""Small deterministic English parser and exact tokenizer offset alignment.

The paper does not specify a parser. This implementation uses a small verb /
relation lexicon; callers may supply explicit spans for general prompts.
No language model or extra grounding network is downloaded.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SemanticUnit:
    label: str
    spans: tuple[tuple[int, int], ...]


VERBS = set("fly flies flying walk walks walking run runs running jump jumps jumping sit sits sitting stand stands standing ride rides riding drive drives driving hold holds holding carry carries carrying play plays playing dance dances dancing swim swims swimming move moves moving chase chases chasing look looks looking eat eats eating wave waves waving roll rolls rolling turn turns turning float floats floating strum strums strumming".split())
RELATIONS = set("above below under over beside behind near inside outside across through toward towards around on in with next between".split())
BREAKS = set("a an the and while as is are was were to of at by slowly quickly gently then".split())


def parse_prompt(prompt: str) -> list[SemanticUnit]:
    words = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?|[,.;!?]", prompt))
    if not words:
        raise ValueError("the built-in parser expects an English prompt; use explicit spans")
    chunks = []
    pending = []
    for i, word in enumerate(words):
        text = word.group().lower()
        if text in VERBS | RELATIONS | BREAKS or not text[0].isalpha():
            if pending:
                chunks.append(pending)
                pending = []
        else:
            pending.append(i)
    if pending:
        chunks.append(pending)
    units = []
    for chunk in chunks:
        a, b = words[chunk[0]].start(), words[chunk[-1]].end()
        units.append(SemanticUnit(prompt[a:b], ((a, b),)))
    for i, word in enumerate(words):
        text = word.group().lower()
        left = next((c for c in reversed(chunks) if c[-1] < i), None)
        right = next((c for c in chunks if c[0] > i), None)
        if left and text in VERBS | RELATIONS:
            subject = words[left[-1]]
            spans = [(subject.start(), subject.end()), (word.start(), word.end())]
            if text in RELATIONS and right:
                obj = words[right[-1]]
                spans.append((obj.start(), obj.end()))
            if text in VERBS or right:
                units.append(SemanticUnit(" ".join(prompt[a:b] for a, b in spans), tuple(spans)))
    if not units:
        units = [SemanticUnit(prompt, ((0, len(prompt)),))]
    # Deduplicate equivalent sets of spans without making separate requirements
    # compete for quota simply because the parser emitted a duplicate.
    return list({unit.spans: unit for unit in units}.values())


def align_units(prompt, units, tokenizer, max_length=226, expected_ids=None):
    """Map char spans to token positions, including noncontiguous relations.

    Fails on truncation / incompatible tokenization instead of silently using
    incorrect text keys. Tokenizer must support offset_mapping (fast tokenizer).
    """
    full = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    if len(full["input_ids"]) > max_length:
        raise ValueError("prompt would be truncated; shorten it before semantic routing")
    encoded = tokenizer(prompt, return_offsets_mapping=True, padding="max_length",
                        max_length=max_length, truncation=True, add_special_tokens=True)
    if expected_ids is not None and list(encoded["input_ids"]) != list(expected_ids):
        raise ValueError("semantic tokenizer IDs differ from the model tokenizer IDs")
    result = []
    for unit in units:
        if not unit.spans or any(a < 0 or b <= a or b > len(prompt) for a, b in unit.spans):
            raise ValueError(f"invalid character spans for {unit.label!r}")
        indices = sorted({i for i, (a, b) in enumerate(encoded["offset_mapping"])
                          if b > a and any(a < end and b > start for start, end in unit.spans)})
        if not indices:
            raise ValueError(f"no tokens aligned for {unit.label!r}")
        result.append(indices)
    return result
