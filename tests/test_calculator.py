import pytest

from cfa_agent.tools import CALCULATOR_SCHEMA, calculator


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2 + 2", "4"),
        ("1234.5 * 0.21", "259.245"),
        ("(180 - 150) / 150 * 100", "20.0"),
        ("2 ** 10", "1024"),
    ],
)
def test_evaluates_arithmetic(expr, expected):
    assert calculator(expr) == expected


def test_division_by_zero_explains():
    assert "zero" in calculator("1 / 0").lower()


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "price",
    ],
)
def test_rejects_non_arithmetic(expr):
    # Review Focus #4
    assert calculator(expr).startswith("Error")


def test_rejects_huge_exponent_instead_of_hanging():
    # Valid arithmetic, but evaluating it would freeze the Colab kernel mid-lecture.
    assert calculator("2 ** 10000000").startswith("Error")


def test_schema_matches_function_name():
    assert CALCULATOR_SCHEMA["function"]["name"] == "calculator"
