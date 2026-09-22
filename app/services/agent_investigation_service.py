"""LangGraph workflow coordinating local fraud-investigation application tools."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.schemas.prediction import PredictionRequest
from app.services.investigation_service import InvestigationService, InvestigationUnavailableError, investigation_service
from app.services.prediction_service import ModelUnavailableError, PredictionService, prediction_service
from app.services.similar_case_service import SimilarCaseService, SimilarCaseUnavailableError, similar_case_service


HIGH_RISK_MARKERS = ("international", "night-time", "failed attempt", "pin change", "crypto")


class CaseState(TypedDict, total=False):
    """Shared state passed between LangGraph investigation agents."""

    case_id: str
    question: str
    transaction_description: str
    transaction: PredictionRequest | None
    route_taken: Literal["direct_assessment", "multi_agent"]
    agent_findings: dict[str, str]
    tools_used: list[str]
    sources: list[dict[str, Any]]
    evidence: list[str]
    tool_failures: list[str]
    final_investigation_summary: str


class AgentInvestigationService:
    """Coordinate triage, prediction, case retrieval, and summary agents with LangGraph."""

    def __init__(
        self,
        prediction_tool: PredictionService = prediction_service,
        similar_case_tool: SimilarCaseService = similar_case_service,
        investigation_tool: InvestigationService = investigation_service,
    ) -> None:
        """Configure reusable application tools and compile the workflow once."""

        self.prediction_tool = prediction_tool
        self.similar_case_tool = similar_case_tool
        self.investigation_tool = investigation_tool
        self._case_history: dict[str, dict[str, Any]] = {}
        self._history_lock = Lock()
        self._graph = self._build_graph()

    def investigate(
        self,
        question: str,
        transaction_description: str,
        transaction: PredictionRequest | None = None,
    ) -> dict[str, Any]:
        """Run the appropriate LangGraph route and retain the completed case in memory."""

        initial_state: CaseState = {
            "case_id": str(uuid4()),
            "question": question.strip(),
            "transaction_description": transaction_description.strip(),
            "transaction": transaction,
            "agent_findings": {},
            "tools_used": [],
            "sources": [],
            "evidence": [f"Reported transaction description: {transaction_description.strip()}"],
            "tool_failures": [],
        }
        final_state = self._graph.invoke(initial_state)
        response = self._response_payload(final_state)
        with self._history_lock:
            self._case_history[response["case_id"]] = deepcopy(response)
        return response

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Return a copy of an in-memory completed case, if it is still available."""

        with self._history_lock:
            case = self._case_history.get(case_id)
            return deepcopy(case) if case is not None else None

    def _build_graph(self) -> Any:
        """Create the conditional LangGraph workflow for case investigation."""

        workflow = StateGraph(CaseState)
        workflow.add_node("route_case", self._route_case)
        workflow.add_node("fraud_analysis_agent", self._fraud_analysis_agent)
        workflow.add_node("similar_case_evidence_agent", self._similar_case_evidence_agent)
        workflow.add_node("direct_assessment", self._direct_assessment)
        workflow.add_node("investigation_summary_agent", self._investigation_summary_agent)
        workflow.add_edge(START, "route_case")
        workflow.add_conditional_edges(
            "route_case",
            self._select_route,
            {
                "simple": "direct_assessment",
                "complex": "fraud_analysis_agent",
            },
        )
        workflow.add_edge("fraud_analysis_agent", "similar_case_evidence_agent")
        workflow.add_edge("similar_case_evidence_agent", "investigation_summary_agent")
        workflow.add_edge("direct_assessment", "investigation_summary_agent")
        workflow.add_edge("investigation_summary_agent", END)
        return workflow.compile()

    def _route_case(self, state: CaseState) -> dict[str, str]:
        """Classify a case as simple or complex using reported risk indicators."""

        route = "multi_agent" if self._is_complex(state) else "direct_assessment"
        return {"route_taken": route}

    @staticmethod
    def _select_route(state: CaseState) -> Literal["simple", "complex"]:
        """Map the stored route decision to the corresponding graph edge."""

        return "complex" if state["route_taken"] == "multi_agent" else "simple"

    def _is_complex(self, state: CaseState) -> bool:
        """Identify cases that warrant all specialist agents rather than direct assessment."""

        transaction = state.get("transaction")
        if transaction is not None:
            if transaction.failed_attempts >= 2:
                return True
            if transaction.is_international and transaction.is_night_transaction:
                return True
            if transaction.pin_changed_recently and transaction.is_international:
                return True
        description = state["transaction_description"].lower()
        return sum(marker in description for marker in HIGH_RISK_MARKERS) >= 2

    def _fraud_analysis_agent(self, state: CaseState) -> dict[str, Any]:
        """Fraud Analysis Agent calls the saved prediction service when inputs are available."""

        findings = dict(state["agent_findings"])
        tools_used = list(state["tools_used"])
        failures = list(state["tool_failures"])
        transaction = state.get("transaction")
        if transaction is None:
            findings["Fraud Analysis Agent"] = (
                "No structured transaction input was supplied, so the prediction tool was not called."
            )
            return {"agent_findings": findings}
        try:
            prediction = self.prediction_tool.predict(transaction)
            tools_used.append("fraud_prediction_service")
            findings["Fraud Analysis Agent"] = (
                f"Saved model output: {prediction.prediction}; risk level {prediction.risk_level}; "
                f"fraud probability {prediction.fraud_probability:.4f}."
            )
        except (ModelUnavailableError, ValueError, RuntimeError) as error:
            failures.append(f"fraud_prediction_service: {error}")
            findings["Fraud Analysis Agent"] = "Prediction tool was unavailable; no model conclusion was used."
        return {"agent_findings": findings, "tools_used": tools_used, "tool_failures": failures}

    def _similar_case_evidence_agent(self, state: CaseState) -> dict[str, Any]:
        """Evidence Agent retrieves similar cases and RAG context with independent safeguards."""

        findings = dict(state["agent_findings"])
        tools_used = list(state["tools_used"])
        sources = list(state["sources"])
        evidence = list(state["evidence"])
        failures = list(state["tool_failures"])

        try:
            cases = self.similar_case_tool.search(state["transaction_description"], top_k=3)
            tools_used.append("similar_case_service")
            sources.extend({"source_type": "similar_case", **case} for case in cases)
            findings["Similar Case / Evidence Agent"] = f"Retrieved {len(cases)} similar historical cases."
        except (SimilarCaseUnavailableError, ValueError, RuntimeError) as error:
            failures.append(f"similar_case_service: {error}")
            findings["Similar Case / Evidence Agent"] = "Similar-case retrieval was unavailable."

        return self._add_rag_context(
            state,
            findings,
            tools_used,
            sources,
            evidence,
            failures,
        )

    def _direct_assessment(self, state: CaseState) -> dict[str, Any]:
        """For simple cases, call the RAG tool directly before a concise summary."""

        findings = dict(state["agent_findings"])
        return self._add_rag_context(
            state,
            findings,
            list(state["tools_used"]),
            list(state["sources"]),
            list(state["evidence"]),
            list(state["tool_failures"]),
            finding_name="Direct Assessment",
        )

    def _add_rag_context(
        self,
        state: CaseState,
        findings: dict[str, str],
        tools_used: list[str],
        sources: list[dict[str, Any]],
        evidence: list[str],
        failures: list[str],
        finding_name: str = "Similar Case / Evidence Agent",
    ) -> dict[str, Any]:
        """Call the investigation RAG tool and preserve its traceable context in shared state."""

        try:
            rag_response = self.investigation_tool.investigate(
                state["question"],
                state["transaction_description"],
            )
            tools_used.append("investigation_rag_service")
            sources.extend({"source_type": "knowledge_base", **source} for source in rag_response["sources"])
            evidence.extend(rag_response["observed_evidence"])
            if finding_name == "Direct Assessment":
                findings[finding_name] = rag_response["investigation_response"]
            else:
                existing = findings.get(finding_name, "")
                findings[finding_name] = f"{existing} {rag_response['investigation_response']}".strip()
        except (InvestigationUnavailableError, ValueError, RuntimeError) as error:
            failures.append(f"investigation_rag_service: {error}")
            if finding_name not in findings:
                findings[finding_name] = "Investigation-context retrieval was unavailable."
        return {
            "agent_findings": findings,
            "tools_used": tools_used,
            "sources": sources,
            "evidence": list(dict.fromkeys(evidence)),
            "tool_failures": failures,
        }

    @staticmethod
    def _investigation_summary_agent(state: CaseState) -> dict[str, Any]:
        """Investigation Summary Agent combines only findings produced by prior agents."""

        finding_text = " ".join(state["agent_findings"].values())
        if state["tool_failures"]:
            summary = (
                f"{finding_text} One or more tools were unavailable, so the assessment is incomplete. "
                "Verify the reported transaction against authoritative records before action."
            )
        else:
            summary = (
                f"{finding_text} Historical similarity and retrieved policy context are not proof of a fraud outcome."
            )
        findings = dict(state["agent_findings"])
        findings["Investigation Summary Agent"] = summary
        return {"agent_findings": findings, "final_investigation_summary": summary}

    @staticmethod
    def _response_payload(state: CaseState) -> dict[str, Any]:
        """Return only the public workflow fields from the final shared state."""

        return {
            "case_id": state["case_id"],
            "route_taken": state["route_taken"],
            "agent_findings": state["agent_findings"],
            "tools_used": state["tools_used"],
            "final_investigation_summary": state["final_investigation_summary"],
            "sources": state["sources"],
            "evidence": state["evidence"],
            "tool_failures": state["tool_failures"],
        }


agent_investigation_service = AgentInvestigationService()
