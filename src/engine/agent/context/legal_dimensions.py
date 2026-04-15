"""
Shared legal review dimensions and prompt normalization helpers.
"""

from __future__ import annotations

import re

_LEGAL_TYPO_MAP = {
    "considerafton": "consideration",
    "quicly": "quickly",
    "ther board": "the board",
}

_LEGAL_ALLOCATION_GROUPS = [
    ["indemn", "indemnity"],
    ["escrow", "holdback"],
    ["reps", "warrant", "representation", "warranty"],
    ["disclosure schedule"],
    ["insurance", "r&w insurance", "representation and warranty insurance"],
    ["cap", "basket", "survival"],
]

_LEGAL_EXECUTION_GROUPS = [
    ["signing", "sign", "closing", "close"],
    ["pre-close", "pre close", "interim covenant", "interim operating"],
    ["consent", "approval", "condition precedent"],
    ["timeline", "weeks", "days", "rapid", "quickly"],
]

_LEGAL_TAX_EXECUTION_GROUPS = [
    ["irs", "section 368", "368", "tax-free reorganization"],
    ["carryover basis", "built-in gain", "basis"],
    ["seller tax", "shareholder tax", "capital gain"],
]

_LEGAL_REGULATORY_EXECUTION_GROUPS = [
    ["hsr", "merger control", "antitrust", "foreign investment", "cfius"],
    ["regulatory remediation", "remediation", "compliance cure", "cure plan"],
    ["consent", "approval", "notification", "condition precedent"],
]

_LEGAL_EMPLOYEE_TRANSFER_GROUPS = [
    ["employee", "employees", "labor", "employment"],
    ["works council", "consultation", "collective", "benefit"],
    ["transfer", "tupe", "retention", "service credit"],
]


def normalize_legal_task_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    for wrong, right in _LEGAL_TYPO_MAP.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def legal_allocation_groups() -> list[list[str]]:
    return [list(group) for group in _LEGAL_ALLOCATION_GROUPS]


def legal_execution_groups() -> list[list[str]]:
    return [list(group) for group in _LEGAL_EXECUTION_GROUPS]


def legal_tax_execution_groups() -> list[list[str]]:
    return [list(group) for group in _LEGAL_TAX_EXECUTION_GROUPS]


def legal_regulatory_execution_groups() -> list[list[str]]:
    return [list(group) for group in _LEGAL_REGULATORY_EXECUTION_GROUPS]


def legal_employee_transfer_groups() -> list[list[str]]:
    return [list(group) for group in _LEGAL_EMPLOYEE_TRANSFER_GROUPS]
