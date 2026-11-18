import pytest

from rigcode.tools.editmatch import EditError, apply_edit, find_omission_placeholder, strip_line_numbers, unescape

SRC = "def f(x):\n    if x:\n        return 1\n    return 2\n"


def test_exact_match():
    out = apply_edit(SRC, "        return 1", "        return 10")
    assert out.content == "def f(x):\n    if x:\n        return 10\n    return 2\n"
    assert out.strategy == "exact"
    assert out.first_line == 3


def test_ambiguous_match_reports_lines():
    src = "a = 1\nb = 2\na = 1\n"
    with pytest.raises(EditError) as e:
        apply_edit(src, "a = 1", "a = 3")
    assert "2 locations" in str(e.value)
    assert "lines 1, 3" in str(e.value)


def test_replace_all():
    out = apply_edit("a = 1\nb = 2\na = 1\n", "a = 1", "a = 3", replace_all=True)
    assert out.count == 2
    assert out.content == "a = 3\nb = 2\na = 3\n"


def test_unicode_punctuation_tier():
    src = 'msg = "it\u2019s ok"\n'
    out = apply_edit(src, 'msg = "it\'s ok"', 'msg = "fine"')
    assert out.content == 'msg = "fine"\n'
    assert out.strategy == "unicode-punctuation"


def test_trailing_whitespace_tier():
    out = apply_edit("x = 1   \ny = 2\n", "x = 1\ny = 2", "x = 1\ny = 3")
    assert out.content == "x = 1\ny = 3\n"
    assert out.strategy == "trailing-whitespace"


def test_indentation_offset_tier():
    src = "class A:\n    def f(self):\n        return 1\n"
    out = apply_edit(src, "def f(self):\n    return 1", "def f(self):\n    return 2")
    assert out.content == "class A:\n    def f(self):\n        return 2\n"
    assert out.strategy == "indentation-offset"


def test_copied_line_number_prefixes():
    out = apply_edit("a = 1\nb = 2\n", "     1\ta = 1\n     2\tb = 2", "     1\ta = 1\n     2\tb = 3")
    assert out.content == "a = 1\nb = 3\n"
    assert out.strategy.startswith("line-number-prefixes+")


def test_double_escaped_strings():
    out = apply_edit("a = 1\nb = 2\n", "a = 1\\nb = 2", "a = 1\\nb = 3")
    assert out.content == "a = 1\nb = 3\n"
    assert out.strategy.startswith("unescaped+")


def test_placeholder_rejected():
    with pytest.raises(EditError, match="placeholder"):
        apply_edit(SRC, "    return 2", "    # ... rest of the code ...\n    return 3")


def test_omission_detector():
    assert find_omission_placeholder("// ... existing code ...")
    assert find_omission_placeholder("# rest of the code remains unchanged")
    assert find_omission_placeholder("x = 1  # set x") is None
    assert find_omission_placeholder("print('...')") is None
    assert find_omission_placeholder("# ... rest ...", old="# ... rest ...") is None


def test_no_match_shows_closest_region():
    with pytest.raises(EditError) as e:
        apply_edit(SRC, "    if x:\n        return 11", "    pass")
    msg = str(e.value)
    assert "most similar" in msg
    assert "lines 2-3" in msg


def test_already_applied_hint():
    with pytest.raises(EditError, match="already be applied"):
        apply_edit(SRC, "return 99", "        return 1\n    return 2")


def test_identical_and_empty_old():
    with pytest.raises(EditError, match="identical"):
        apply_edit(SRC, "x", "x")
    with pytest.raises(EditError, match="empty"):
        apply_edit(SRC, "", "y")


def test_helpers():
    assert strip_line_numbers("  10\tfoo\n  11\tbar") == "foo\nbar"
    assert strip_line_numbers("foo\n  11\tbar") is None
    assert unescape("a\\nb") == "a\nb"
    assert unescape("plain") is None
