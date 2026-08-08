"""A LangChain agent facade for CURA AI chat routing."""

from __future__ import annotations

import json
from typing import Any

from langchain.agents import create_agent

from .tools import CURAI_TOOLS, liver_prediction_tool, mri_analysis_tool, navigation_payload, xray_fracture_tool

SYSTEM_PROMPT = """You are CURA AI Medical Assistant. You have exactly four tools: liver prediction navigation, X-ray fracture navigation, MRI navigation, and grounded medical-document RAG. Never diagnose from chat alone. Use a prediction-navigation tool only when a user explicitly wants a prediction, assessment, test, or image analysis. For questions about precautions, prevention, symptoms, causes, treatment, diet, care, or general condition information, use grounded RAG only and do not return a navigation tool. Include that this is not medical advice when discussing medical matters."""


INFORMATIONAL_TERMS = (
    "precaution", "precautions", "prequaction", "prequactions", "prevention",
    "prevent", "symptom", "symptoms", "cause", "causes", "treatment",
    "medicine", "medication", "diet", "food", "care", "manage", "management",
    "what should", "what can", "what to do", "information", "explain",
)


def create_cura_agent(model: Any):
    """Create the LangChain tool-calling agent graph for a configured chat model."""
    return create_agent(model=model, tools=CURAI_TOOLS, system_prompt=SYSTEM_PROMPT, name="cura_medical_agent")


def get_agent_response(question: str) -> dict[str, Any]:
    """Route a request through one of the four LangChain tools.

    Deterministic routing keeps local prediction navigation available even when a
    cloud chat-model credential is not configured.
    """
    normalized = " ".join(question.lower().replace("x-ray", "xray").split())

    # Information requests must remain in the RAG experience: they should not
    # display a prediction/upload button simply because a disease is mentioned.
    if any(term in normalized for term in INFORMATIONAL_TERMS):
        from RAG.scripts import answer_from_documents
        return answer_from_documents(question)

    if any(term in normalized for term in ("mri", "brain tumor", "brain tumour", "glioma", "meningioma", "pituitary")):
        payload = navigation_payload(mri_analysis_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}
    if any(term in normalized for term in ("xray", "fracture", "broken bone", "broken arm", "broken wrist", "broken elbow")):
        payload = navigation_payload(xray_fracture_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}
    if any(term in normalized for term in ("liver", "bilirubin", "fatty liver", "hepatitis")):
        payload = navigation_payload(liver_prediction_tool.invoke({"question": question}))
        return {"answer": payload["message"], "tool": payload}

    from .tools import medical_rag_tool
    return json.loads(medical_rag_tool.invoke({"question": question}))
