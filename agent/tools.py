"""The four LangChain tools exposed by the CURA AI medical assistant."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool


def _navigation_tool(tool_name: str, message: str, target: str) -> str:
    return json.dumps({"tool_name": tool_name, "message": message, "action": "navigate", "target": target})


@tool
def liver_prediction_tool(question: str) -> str:
    """Use only when the user explicitly asks to run a liver prediction or assessment. Do not use for general disease questions."""
    return _navigation_tool("liver", "Open the liver assessment form to enter patient values and run the ML prediction.", "#liver")


@tool
def xray_fracture_tool(question: str) -> str:
    """Use only when the user explicitly asks to upload or classify an X-ray. Do not use for general fracture questions."""
    return _navigation_tool("xray", "Open the X-ray section to upload an image for fracture classification.", "#xray")


@tool
def mri_analysis_tool(question: str) -> str:
    """Use only when the user explicitly asks to upload or classify a brain MRI. Do not use for general tumor questions."""
    return _navigation_tool("mri", "Open the MRI section to upload an image for brain MRI classification.", "#mri")


@tool
def medical_rag_tool(question: str) -> str:
    """Use for general medical-document questions that do not need a prediction form. Searches curated medical documents."""
    # Lazy import keeps this module usable without loading embeddings at startup.
    from RAG.scripts import answer_from_documents
    return json.dumps(answer_from_documents(question))


CURAI_TOOLS = [liver_prediction_tool, xray_fracture_tool, mri_analysis_tool, medical_rag_tool]


def navigation_payload(tool_result: str) -> dict[str, Any]:
    """Parse tool output into the API response shape used by the browser."""
    return json.loads(tool_result)
