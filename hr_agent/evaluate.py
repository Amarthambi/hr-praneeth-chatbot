"""Performance measures from the HR policy case study."""

from __future__ import annotations

import math


def task_success_rate(successful_tasks: int, total_tasks: int) -> float:
    """Percentage of questions handled successfully."""
    _require_positive(total_tasks, "total_tasks")
    _require_non_negative(successful_tasks, "successful_tasks")
    if successful_tasks > total_tasks:
        raise ValueError("successful_tasks cannot exceed total_tasks")
    return successful_tasks / total_tasks * 100


def failure_rate(failed_tasks: int, total_tasks: int) -> float:
    """Percentage of questions that were not completed successfully."""
    _require_positive(total_tasks, "total_tasks")
    _require_non_negative(failed_tasks, "failed_tasks")
    if failed_tasks > total_tasks:
        raise ValueError("failed_tasks cannot exceed total_tasks")
    return failed_tasks / total_tasks * 100


def hallucination_rate(incorrect_or_unsupported_claims: int, claims_checked: int) -> float:
    """Percentage of checked claims that the policy documents do not support."""
    _require_positive(claims_checked, "claims_checked")
    _require_non_negative(incorrect_or_unsupported_claims, "incorrect_or_unsupported_claims")
    if incorrect_or_unsupported_claims > claims_checked:
        raise ValueError("incorrect claims cannot exceed claims_checked")
    return incorrect_or_unsupported_claims / claims_checked * 100


def cost_per_task(
    input_tokens: int,
    input_price: float,
    output_tokens: int,
    output_price: float,
) -> float:
    """Token cost for one question: (input tokens x input price) + (output tokens x output price)."""
    _require_non_negative(input_tokens, "input_tokens")
    _require_non_negative(output_tokens, "output_tokens")
    if input_price < 0 or output_price < 0:
        raise ValueError("token prices must be zero or positive")
    return input_tokens * input_price + output_tokens * output_price


def chained_reliability(single_step_success_rate: float, steps: int) -> float:
    """Approximate reliability of n sequential steps with the same success rate.

    single_step_success_rate is a fraction from 0 to 1, not a percentage.
    """
    if not 0 <= single_step_success_rate <= 1:
        raise ValueError("single_step_success_rate must be between 0 and 1")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return single_step_success_rate ** steps


def successes_required(target_percent: float, total_tasks: int) -> int:
    """Smallest success count that reaches the target percentage."""
    _require_positive(total_tasks, "total_tasks")
    if not 0 <= target_percent <= 100:
        raise ValueError("target_percent must be between 0 and 100")
    return math.ceil(target_percent * total_tasks / 100 - 1e-9)


def max_incorrect_claims(max_rate_percent: float, claims_checked: int) -> int:
    """Largest error count that stays within the maximum hallucination rate."""
    _require_non_negative(claims_checked, "claims_checked")
    if max_rate_percent < 0:
        raise ValueError("max_rate_percent must be zero or positive")
    return math.floor(max_rate_percent * claims_checked / 100 + 1e-9)


def format_percent(value: float) -> str:
    rounded = round(value, 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}%"
    return f"{rounded:.1f}%"


def pdf_scenarios() -> list[dict[str, object]]:
    """Worked results for the six practical scenarios in the case study."""
    failed = 200 - 184
    success = task_success_rate(184, 200)
    failure = failure_rate(failed, 200)
    required = successes_required(95, 200)
    additional = required - 184
    errors = hallucination_rate(8, 250)
    allowed = max_incorrect_claims(2, 500)
    cost = cost_per_task(2000, 0.001, 500, 0.002)
    return [
        {
            "title": "1. Baseline success rate",
            "detail": "200 employee questions, 184 resolved successfully.",
            "lines": [
                f"Task success rate = 184 / 200 x 100 = {format_percent(success)}",
                f"Unsuccessful questions = 200 - 184 = {failed}",
                f"Failure rate = {failed} / 200 x 100 = {format_percent(failure)}",
            ],
        },
        {
            "title": "2. Improving the success rate",
            "detail": "Current result is 184 successes out of the same 200 questions (92%). Target is 95%.",
            "lines": [
                f"Successes required for 95% = {required}",
                f"Additional successes needed = {required} - 184 = {additional}",
            ],
        },
        {
            "title": "3. Checking factual accuracy",
            "detail": "250 policy claims checked, 8 incorrect when compared with the handbook.",
            "lines": [
                f"Hallucination rate = 8 / 250 x 100 = {format_percent(errors)}",
            ],
        },
        {
            "title": "4. Setting an error limit",
            "detail": "Maximum hallucination rate is 2% across 500 checked claims.",
            "lines": [
                f"Maximum incorrect claims = {allowed}",
            ],
        },
        {
            "title": "5. Calculating processing cost",
            "detail": "2,000 input tokens at 0.001 each, and 500 output tokens at 0.002 each.",
            "lines": [
                f"Cost = (2000 x 0.001) + (500 x 0.002) = {cost:.2f}",
            ],
        },
        {
            "title": "6. Interpreting the results",
            "detail": "Recorded results: task success 92%, failure 8%, hallucination 3.2%, maximum hallucination 2%.",
            "lines": [
                "Meets the target: none. The only numeric limit in this scenario is a hallucination rate of at most 2%.",
                "Does not meet the target: hallucination rate, 3.2% against a maximum of 2%.",
                "Improvement before wider use: tighten retrieval, answer only from policy text, and refuse or escalate any claim the documents do not support.",
            ],
        },
    ]


def _require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")
