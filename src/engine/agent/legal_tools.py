"""Built-in legal advisory tools for the active engine."""

from __future__ import annotations

import json

from langchain_core.tools import tool


def _json_envelope(tool_name: str, facts: dict) -> dict:
    return {
        "type": tool_name,
        "facts": facts,
        "assumptions": {},
        "source": {"tool": tool_name},
        "quality": {
            "is_synthetic": False,
            "is_estimated": False,
            "cache_hit": False,
            "missing_fields": [],
        },
        "errors": [],
    }


@tool
def legal_playbook_retrieval(query: str, deal_size_hint: str = "", urgency: str = "") -> str:
    """Return high-level playbook guidance for legal-finance structure analysis."""
    facts = {
        "playbook_points": [
            "Separate structure choice from risk-allocation mechanics; both must be explicit.",
            "When equity, rollover, or non-cash consideration is contemplated, analyze economics and tax treatment against inherited liability risk.",
            "Cross-border regulatory gaps require closing-condition, covenant, and remediation sequencing detail.",
            "Accelerated timelines demand a narrower diligence plan, not a thinner one.",
        ],
        "query": query,
        "deal_size_hint": deal_size_hint,
        "urgency": urgency,
    }
    return json.dumps(_json_envelope("legal_playbook_retrieval", facts), ensure_ascii=True)


@tool
def transaction_structure_checklist(
    consideration_preference: str = "",
    liability_goal: str = "",
    urgency: str = "",
) -> str:
    """Return structure options and liability-allocation checklist items."""
    structures = [
        {
            "structure": "asset purchase with selective assumed liabilities",
            "tradeoff": "best liability ring-fencing but can trigger tax leakage and transfer-consent friction",
            "timeline_dependency": "works best if assets and contracts can be cleanly separated on an accelerated timeline",
        },
        {
            "structure": "carve-out sale of clean IP or business line",
            "tradeoff": "isolates contaminated operations but requires pre-close separation work and diligence clarity",
            "timeline_dependency": "depends on whether the seller can separate the clean perimeter before signing or closing",
        },
        {
            "structure": "reverse triangular merger with aggressive indemnity and escrow package",
            "tradeoff": "supports equity consideration more naturally but exposes the buyer to more inherited liability risk",
            "timeline_dependency": "requires stronger closing conditions, indemnity, and remediation covenants if diligence is compressed",
        },
        {
            "structure": "hybrid structure with contingent consideration and pre-close remediation covenants",
            "tradeoff": "balances economics and risk sharing but adds negotiation and milestone complexity",
            "timeline_dependency": "depends on whether the parties can define measurable cure and release triggers quickly",
        },
    ]
    facts = {
        "candidate_structures": structures,
        "allocation_mechanics": [
            "specific indemnities for known compliance gaps",
            "escrow or holdback with milestone release mechanics",
            "caps, baskets, and survival periods for reps and warranties",
            "disclosure schedules and compliance remediation covenants",
        ],
        "consideration_preference": consideration_preference,
        "liability_goal": liability_goal,
        "urgency": urgency,
    }
    return json.dumps(_json_envelope("transaction_structure_checklist", facts), ensure_ascii=True)


@tool
def regulatory_execution_checklist(jurisdictions_json: str = "[]", regulatory_gaps: str = "") -> str:
    """Return regulatory execution and closing-condition checklist items."""
    try:
        jurisdictions = json.loads(jurisdictions_json) if jurisdictions_json else []
    except Exception:
        jurisdictions = [jurisdictions_json] if jurisdictions_json else []
    facts = {
        "jurisdictions": jurisdictions,
        "execution_dependencies": [
            "map approvals, filings, and consultation obligations by jurisdiction",
            "separate pre-closing cure items from post-closing remediation covenants",
            "decide which compliance issues become closing conditions versus price protections",
            "track workforce-transfer, consultation, third-party consent, and other timing constraints for each jurisdiction",
        ],
        "regulatory_gaps": regulatory_gaps,
    }
    return json.dumps(_json_envelope("regulatory_execution_checklist", facts), ensure_ascii=True)


@tool
def tax_structure_checklist(consideration_preference: str = "", cross_border: bool = False) -> str:
    """Return tax execution checklist items for structure selection."""
    facts = {
        "tax_structure_comparison": [
            "identify who receives the tax benefit under each structure",
            "identify required elections, qualification conditions, or rollover mechanics",
            "spell out what breaks the intended tax treatment",
            "compare equity or rollover treatment against asset deal step-up and leakage tradeoffs",
        ],
        "consideration_preference": consideration_preference,
        "cross_border": bool(cross_border),
    }
    return json.dumps(_json_envelope("tax_structure_checklist", facts), ensure_ascii=True)
