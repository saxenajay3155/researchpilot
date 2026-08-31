import ast
import operator
from langchain.tools import tool
import re

#Calculator

# Operations we explicitly allow
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calculate(node):

    # Numbers
    if isinstance(node, ast.Constant):

        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")


    # + - * / ** %
    if isinstance(node, ast.BinOp):

        operation = OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unsupported operation."
            )

        left = _calculate(node.left)
        right = _calculate(node.right)

        return operation(
            left,
            right
        )


    # Negative / positive numbers
    if isinstance(node, ast.UnaryOp):

        operation = OPERATORS.get(
            type(node.op)
        )

        if operation is None:
            raise ValueError(
                "Unsupported unary operation."
            )

        return operation(
            _calculate(node.operand)
        )


    raise ValueError(
        "Invalid mathematical expression."
    )


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Use this for arithmetic such as differences,
    percentages, growth rates, ratios, addition,
    subtraction, multiplication, and division.

    The expression must contain only numbers,
    parentheses, and +, -, *, /, **, %.

    Do not use functions such as round(), min(),
    max(), abs(), or comparison operators.
    """

    # Remove thousands separators:
    # 49,908 -> 49908
    # 1,78,650 -> 178650
    expression = re.sub(
        r"(?<=\d),(?=\d)",
        "",
        expression
    )

    tree = ast.parse(
        expression,
        mode="eval"
    )

    return float(
        _calculate(tree.body)
    )

tools = {"calculator":calculator}