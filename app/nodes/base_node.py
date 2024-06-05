from abc import ABC, abstractmethod
from typing import Optional, List
import re

from loguru import logger


class BaseNode(ABC):
    def __init__(self,
                 node_name: str,
                 node_type: str,
                 input: str,
                 output: List[str],
                 min_input_len: int = 1,
                 model_config: Optional[dict] = None
                 ):
        self.node_name = node_name
        self.input = input   # 这个节点需要什么
        self.output = output # 这个节点会输出什么
        self.min_input_len = min_input_len # 最少需要什么
        self.model_config = model_config

        if node_type not in ["node", "conditional_node"]:
            raise ValueError(
                f"node_type must be 'node' or 'conditional_node', got '{node_type}'")
        self.node_type = node_type
        self.logger = logger

    @abstractmethod
    def execute(self, state: dict) -> dict:
        pass

    def get_input_keys(self, state: dict) -> List[str]:
        """Use the _parse_input_keys method to identify which state keys are
        needed based on the input attribute
        """
        try:
            input_keys = self._parse_input_keys(state, self.input)
            self._validate_input_keys(input_keys)
            return input_keys
        except ValueError as e:
            raise ValueError(
                f"Error parsing input keys for {self.node_name}: {str(e)}")

    def _validate_input_keys(self, input_keys):
        if len(input_keys) < self.min_input_len:
            raise ValueError(
                f"{self.node_name} requires at least {self.min_input_len} input keys, got {len(input_keys)}.")

    def _parse_input_keys(self, state: dict, expression: str) -> List[str]:
        # Check for empty expression
        if not expression:
            raise ValueError("Empty expression.")

        # Check for adjacent state keys without an operator between them
        pattern = r'\b(' + '|'.join(re.escape(key) for key in state.keys()) + \
                  r')(\b\s*\b)(' + '|'.join(re.escape(key)
                                            for key in state.keys()) + r')\b'
        if re.search(pattern, expression):
            raise ValueError(
                "Adjacent state keys found without an operator between them.")

        # Remove spaces
        expression = expression.replace(" ", "")

        # Check for operators with empty adjacent tokens or at the start/end
        if expression[0] in '&|' or expression[-1] in '&|' \
                or '&&' in expression or '||' in expression or \
                '&|' in expression or '|&' in expression:
            raise ValueError("Invalid operator usage.")

        # Check for balanced parentheses and valid operator placement
        open_parentheses = close_parentheses = 0
        for i, char in enumerate(expression):
            if char == '(':
                open_parentheses += 1
            elif char == ')':
                close_parentheses += 1
            # Check for invalid operator sequences
            if char in "&|" and i + 1 < len(expression) and expression[i + 1] in "&|":
                raise ValueError(
                    "Invalid operator placement: operators cannot be adjacent.")

        # Check for missing or balanced parentheses
        if open_parentheses != close_parentheses:
            raise ValueError(
                "Missing or unbalanced parentheses in expression.")

        # Helper function to evaluate an expression without parentheses
        def evaluate_simple_expression(exp):
            # Split the expression by the OR operator and process each segment
            for or_segment in exp.split('|'):
                # Check if all elements in an AND segment are in state
                and_segment = or_segment.split('&')
                if all(elem.strip() in state for elem in and_segment):
                    return [elem.strip() for elem in and_segment if elem.strip() in state]
            return []

        # Helper function to evaluate expressions with parentheses
        def evaluate_expression(expression):
            while '(' in expression:
                start = expression.rfind('(')
                end = expression.find(')', start)
                sub_exp = expression[start + 1:end]
                # Replace the evaluated part with a placeholder and then evaluate it
                sub_result = evaluate_simple_expression(sub_exp)
                # For simplicity in handling, join sub-results with OR to reprocess them later
                expression = expression[:start] + \
                             '|'.join(sub_result) + expression[end + 1:]
            return evaluate_simple_expression(expression)

        result = evaluate_expression(expression)

        if not result:
            raise ValueError("No state keys matched the expression.")

        # Remove redundant state keys from the result, without changing their order
        final_result = []
        for key in result:
            if key not in final_result:
                final_result.append(key)

        return final_result
