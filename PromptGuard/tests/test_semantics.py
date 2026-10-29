import pytest
from promptguard.semantics import parse_prompt, align_units, SemanticUnit


def test_example_semantics():
    prompt = "a small red bird flies above a blue car"
    units = parse_prompt(prompt)
    assert [u.label for u in units] == ["small red bird", "blue car", "bird flies", "bird above car"]
    assert " ".join(prompt[a:b] for a, b in units[-1].spans) == "bird above car"


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        import re
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\w+", text)] + [(0, 0)]
        ids = list(range(1, len(offsets)+1))
        if kwargs.get("padding"):
            length = kwargs["max_length"]
            ids, offsets = ids[:length], offsets[:length]
            offsets += [(0, 0)] * (length - len(ids))
            ids += [0] * (length - len(ids))
        return {"input_ids": ids, "offset_mapping": offsets}


def test_offset_alignment_noncontiguous_and_no_special_tokens():
    prompt = "a small red bird flies above a blue car"
    result = align_units(prompt, parse_prompt(prompt), FakeTokenizer(), 12)
    assert result == [[1, 2, 3], [7, 8], [3, 4], [3, 5, 8]]


def test_truncation_and_invalid_spans_fail():
    with pytest.raises(ValueError, match="truncated"):
        align_units("red bird flies", parse_prompt("red bird flies"), FakeTokenizer(), 2)
    with pytest.raises(ValueError, match="invalid character"):
        align_units("bird", [SemanticUnit("bad", ((-1, 2),))], FakeTokenizer())
    with pytest.raises(ValueError, match="differ"):
        align_units("bird", parse_prompt("bird"), FakeTokenizer(), 2, expected_ids=[99, 99])
