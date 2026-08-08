"""A LangChain agent facade for CURA AI chat routing."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents import create_agent

from .tools import CURAI_TOOLS, liver_prediction_tool, mri_analysis_tool, navigation_payload, xray_fracture_tool

SYSTEM_PROMPT = """You are CURA AI Medical Assistant. You have exactly four tools: liver prediction navigation, X-ray fracture navigation, MRI navigation, and grounded medical-document RAG. Never diagnose from chat alone. Use a prediction-navigation tool only when a user explicitly wants a prediction, assessment, test, or image analysis. For questions about precautions, prevention, symptoms, causes, treatment, diet, care, or general condition information, use grounded RAG only and do not return a navigation tool. Include that this is not medical advice when discussing medical matters."""


WORKFLOW_REQUEST_PATTERNS = (
    # Direct commands: "analyze my MRI", "classify this X-ray".
    r"\b(?:analy[sz]e|classify|predict|assess|evaluate|detect|check|test|screen)\s+"
    r"(?:me\b|my\b|this\b|these\b|the attached\b|an? (?:uploaded|attached)\b)",
    # Polite or conversational requests: "can you analyze ...", "I want to test ...".
    r"\b(?:can|could|would|will) you (?:please )?"
    r"(?:analy[sz]e|classify|predict|assess|evaluate|detect|check|test|screen)\b",
    r"\bi (?:want|need|would like) (?:you )?to (?:analy[sz]e|classify|predict|assess|evaluate|detect|check|test|screen)\b",
    r"\bhelp me (?:analy[sz]e|classify|predict|assess|evaluate|detect|check|test|screen)\b",
    # Explicit workflow language and personal diagnostic questions.
    r"\b(?:run|start|open|use) (?:the |a |an )?(?:ai |ml )?"
    r"(?:(?:liver|xray|mri|brain) )?(?:analysis|prediction|assessment|classifier|model|tool)\b",
    r"\bupload\b.{0,40}\b(?:image|scan|xray|mri|report|result|results|labs?|values?)\b",
    r"\bdo i have\b",
    r"\b(?:check|test|screen) (?:me |myself )?for\b",
    # Short imperative requests commonly entered in the chat box.
    r"^(?:please )?(?:predict|assess|detect|classify)\b",
)


def _has_workflow_request(question: str) -> bool:
    """Return True only for an explicit request to run an analysis workflow."""
    return any(re.search(pattern, question) for pattern in WORKFLOW_REQUEST_PATTERNS)


def _navigation_target(question: str) -> str | None:
    """Select a workflow only for an explicit prediction or upload request."""
    normalized = " ".join(question.lower().replace("x-ray", "xray").split())

    # A condition or imaging term identifies the subject, but is never enough
    # on its own to open a tool. The user must also explicitly request analysis.
    if not _has_workflow_request(normalized):
        return None
    if any(term in normalized for term in ("mri", "brain scan", "brain tumor", "brain tumour", "glioma", "meningioma", "pituitary")):
        return "mri"
    if any(term in normalized for term in ("xray", "fracture", "broken bone", "broken arm", "broken wrist", "broken elbow")):
        return "xray"
    if any(term in normalized for term in ("liver", "bilirubin", "fatty liver", "hepatitis")):
        return "liver"
    return None


def create_cura_agent(model: Any):
    """Create the LangChain tool-calling agent graph for a configured chat model."""
    return create_agent(model=model, tools=CURAI_TOOLS, system_prompt=SYSTEM_PROMPT, name="cura_medical_agent")


def get_agent_response(question: str) -> dict[str, Any]:
    """Route a request through one of the four LangChain tools.

    Deterministic routing keeps local prediction navigation available even when a
    cloud chat-model credential is not configured.
    """
    target = _navigation_target(question)
    if target == "mri":
        payload = navigation_payload(mri_analysis_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}
    if target == "xray":
        payload = navigation_payload(xray_fracture_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}
    if target == "liver":
        payload = navigation_payload(liver_prediction_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}

    from .tools import medical_rag_tool
    return json.loads(medical_rag_tool.invoke({"question": question}))
