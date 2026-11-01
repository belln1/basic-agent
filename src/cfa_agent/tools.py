"""The tools the agent can call.

Each tool is a plain function that takes simple arguments and returns a **string**.
That return type is not an accident: whatever a tool returns is handed straight back
to the model as the text of a message, so a tool that raises would end the
conversation instead of informing it. Every path here returns a string, errors
included.

Next to each function is its schema -- the description the model reads when deciding
whether to call it. The schemas are written out by hand rather than generated,
because writing one is the exercise.
"""

import ast

# Arithmetic only. Anything outside this list is rejected rather than evaluated,
# which is why this tool cannot be talked into running `__import__('os')`.
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.UnaryOp,
    ast.BinOp,
    ast.USub,
    ast.UAdd,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)

# `2 ** 10000000` is perfectly good arithmetic and would hang the notebook.
_MAX_EXPONENT = 1000

_BINARY_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _evaluate(node):
    """Walk the validated tree and compute its value."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and right > _MAX_EXPONENT:
            raise ValueError(f"exponent above {_MAX_EXPONENT} is too large to compute")
        return _BINARY_OPS[type(node.op)](left, right)
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. "1234.5 * 0.21"."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return f"Error: {expression!r} is not a valid arithmetic expression."

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return (
                f"Error: {expression!r} contains something that is not arithmetic. "
                "Pass numbers and the operators + - * / // % ** only."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return f"Error: {node.value!r} is not a number."
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return f"Error: {node.value!r} is not a number."

    try:
        return str(_evaluate(tree))
    except ZeroDivisionError:
        return "Error: cannot divide by zero."
    except ValueError as exc:
        return f"Error: {exc}."


CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate one arithmetic expression and return the result. "
            "Use this for any ratio, growth rate, or multiple you need to compute "
            "rather than doing the arithmetic yourself. Pass numbers and the "
            "operators + - * / // % ** only -- no variable names, no function calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic to evaluate, e.g. '(180 - 150) / 150 * 100'",
                }
            },
            "required": ["expression"],
        },
    },
}
