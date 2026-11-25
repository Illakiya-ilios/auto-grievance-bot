from __future__ import annotations

import re
from typing import Literal, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt


REQUIRED_FIELDS = ("description", "name", "contact", "category", "subject", "recipient")


class ChatState(TypedDict, total=False):
	message: str
	intent: Literal["registration", "tracking", "escalation", "general"]
	details: dict[str, str]
	required_follow_up_fields: list[str]
	awaiting: str
	missing: list[str]
	response: str
	action: str
	follow_up_question: str
	previous_follow_up_questions: list[str]
	follow_up_declined: bool
	approved: bool
	cancelled: bool


class GrievanceLLM(Protocol):
	def extract_details(self, message: str) -> dict[str, str]: ...

	def generate_reply(
		self,
		message: str,
		intent: str,
		instruction: str,
		context: dict[str, object] | None = None,
		documents: list[dict[str, str]] | None = None,
	) -> str: ...

	def follow_up_question(
		self,
		description: str,
		details: dict[str, str],
		documents: list[dict[str, str]] | None = None,
	) -> str | None: ...


FIELD_LABELS = {
	"name": "your name",
	"contact": "a phone number or email address",
	"category": "the grievance category",
	"subject": "a short subject",
	"description": "a description of the issue",
	"recipient": "the person or department that should receive the grievance",
	"impact": "how the issue has affected you",
	"expected_resolution": "what outcome you want",
}


def extract_local_details(message: str, details: dict[str, str]) -> dict[str, str]:
	updated = dict(details)
	description = re.sub(
		r"^\s*(?:i(?:'d like| want) to (?:register|file|submit) (?:a )?(?:grievance|complaint)|(?:register|file) (?:a )?(?:grievance|complaint))[\s.!,:;-]*",
		"",
		message,
		flags=re.I,
	).strip()
	name = re.search(
		r"\bmy name is\s+([A-Za-z][A-Za-z '-]{1,60}?)(?=\s+(?:and|with|my|phone|email)\b|[,.;!?]|$)",
		message,
		re.I,
	)
	email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", message)
	mobile_number = re.search(r"\+?\d[\d ()-]{6,}\d", message)
	if name:
		updated["name"] = name.group(1).strip(" .,!?")
	if email:
		updated["email"] = email.group(0).strip()
		updated["contact"] = updated["email"]
	if mobile_number:
		updated["mobile_number"] = mobile_number.group(0).strip()
		updated.setdefault("contact", updated["mobile_number"])
	if not updated.get("description") and len(description) > 8 and not name and not email and not mobile_number:
		updated["description"] = description
	if not updated.get("subject") and updated.get("description"):
		updated["subject"] = updated["description"][:72].strip()
	return updated


def _is_duplicate_follow_up(question: str, previous_questions: list[str]) -> bool:
	stop_words = {
		"about", "are", "can", "could", "did", "do", "for", "from", "how", "is",
		"me", "of", "please", "specify", "specific", "the", "this", "to", "was",
		"what", "when", "where", "which", "who", "you",
	}

	def keywords(value: str) -> set[str]:
		return {
			word for word in re.findall(r"[a-z]+", value.lower())
			if len(word) > 2 and word not in stop_words
		}

	current = keywords(question)
	if not current:
		return False
	for previous in previous_questions:
		prior = keywords(previous)
		if not prior:
			continue
		shared = len(current & prior)
		if shared >= 2 and shared / min(len(current), len(prior)) >= 0.6:
			return True
	return False


def identify_intent(state: ChatState) -> ChatState:
	if state.get("intent") not in {"registration", "tracking", "escalation", "general"}:
		raise RuntimeError("A valid LLM-classified intent is required.")
	return state


def extract_grievance_details(state: ChatState, config: RunnableConfig) -> ChatState:
	message = state.get("message", "")
	details = state.get("details", {})
	local_details = extract_local_details(message, details)
	extracted: dict[str, str] = {}
	configurable = config.get("configurable", {})
	llm: GrievanceLLM | None = configurable.get("llm_service")
	message_details = re.sub(
		r"^\s*(?:i(?:'d like| want) to (?:register|file|submit) (?:a )?(?:grievance|complaint)|(?:register|file) (?:a )?(?:grievance|complaint))[\s.!,:;-]*",
		"",
		message,
		flags=re.I,
	).strip(" \t\r\n.,!?;:-")
	if llm and message_details.lower() not in {"", "please"} and state.get("awaiting") != "contact":
		extracted = {
			key: value.strip()
			for key, value in llm.extract_details(message).items()
			if isinstance(value, str) and value.strip()
		}
	state["details"] = {**local_details, **extracted}
	if not details.get("description") and local_details.get("description"):
		state["details"]["description"] = local_details["description"]
	return state


def collect_registration_details(state: ChatState) -> ChatState:
	details = extract_local_details(state.get("message", ""), state.get("details", {}))
	awaiting = state.get("awaiting")
	answer = state.get("message", "").strip()
	collectible_fields = (*REQUIRED_FIELDS, *state.get("required_follow_up_fields", []), "supporting_information")
	if awaiting in collectible_fields and not details.get(awaiting) and answer:
		details[awaiting] = answer
	if not details.get("subject") and details.get("description"):
		details["subject"] = details["description"][:72].strip()
	state["details"] = details
	return state


def check_missing_information(state: ChatState) -> ChatState:
	required_fields = (*REQUIRED_FIELDS, *state.get("required_follow_up_fields", []))
	state["missing"] = [field for field in required_fields if not state["details"].get(field)]
	return state


def generate_follow_up_question(state: ChatState, config: RunnableConfig) -> ChatState:
	if state.get("follow_up_question"):
		return state
	if (
		not state.get("details", {}).get("description")
		or state.get("awaiting") not in (
			None,
			"description",
			"supporting_information",
			*state.get("required_follow_up_fields", []),
		)
	):
		return state

	configurable = config.get("configurable", {})
	llm: GrievanceLLM | None = configurable.get("llm_service")
	if not llm:
		raise RuntimeError("An LLM service is required to generate grievance follow-up questions.")

	previous_questions = state.get("previous_follow_up_questions", [])
	llm_details = dict(state.get("details", {}))
	if previous_questions:
		llm_details["_previous_follow_up_questions"] = "\n".join(previous_questions)
	if state.get("follow_up_declined"):
		llm_details["_follow_up_declined"] = "true"
	question = llm.follow_up_question(
		state["details"]["description"],
		llm_details,
		configurable.get("documents", []),
	)
	if question and not _is_duplicate_follow_up(question, previous_questions):
		state["follow_up_question"] = question
		state.setdefault("previous_follow_up_questions", []).append(question)
	return state


def route_intent(state: ChatState) -> Literal["registration", "tracking", "escalation", "general"]:
	return state["intent"]


def ask_for_missing_details(state: ChatState, config: RunnableConfig) -> ChatState:
	configurable = config.get("configurable", {})
	llm: GrievanceLLM | None = configurable.get("llm_service")
	if not llm:
		raise RuntimeError("An LLM service is required to generate chatbot responses.")
	follow_up_question = state.get("follow_up_question")
	if follow_up_question:
		state["action"] = "clarify"
		state["awaiting"] = "supporting_information"
		state["response"] = follow_up_question
		return state
	missing = state.get("missing", list(REQUIRED_FIELDS))
	if missing:
		field = missing[0]
		field_label = FIELD_LABELS.get(field, field.replace("_", " "))
		state["awaiting"] = field
		state["response"] = llm.generate_reply(
			state.get("message", ""),
			"registration",
			f"Ask the user for the missing {field} field ({field_label}). Ask only for this field.",
			{"missing_field": field, "known_details": state.get("details", {})},
			configurable.get("documents", []),
		)
	else:
		state["awaiting"] = ""
		state["action"] = "review"
		state["response"] = llm.generate_reply(
			state.get("message", ""),
			"registration",
			"Tell the user the grievance draft is ready for review. Clearly state it is not submitted until they approve it.",
			{"known_details": state.get("details", {})},
			configurable.get("documents", []),
		)
	return state


def explain_tracking(state: ChatState, config: RunnableConfig) -> ChatState:
	state["action"] = "tracking"
	configurable = config.get("configurable", {})
	llm: GrievanceLLM | None = configurable.get("llm_service")
	if not llm:
		raise RuntimeError("An LLM service is required to generate chatbot responses.")
	state["response"] = llm.generate_reply(
		state.get("message", ""),
		"tracking",
		"Ask the user for the grievance ID so its status can be checked.",
	)
	return state


def explain_escalation(state: ChatState, config: RunnableConfig) -> ChatState:
	state["action"] = "escalation"
	configurable = config.get("configurable", {})
	llm: GrievanceLLM | None = configurable.get("llm_service")
	if not llm:
		raise RuntimeError("An LLM service is required to generate chatbot responses.")
	state["response"] = llm.generate_reply(
		state.get("message", ""),
		"escalation",
		"Ask for the grievance ID and a concise reason for escalation.",
	)
	return state


def wait_for_approval(state: ChatState, config: RunnableConfig) -> ChatState:
	while True:
		decision = interrupt({"details": state.get("details", {})})
		if decision == "approve":
			state["approved"] = True
			return state
		if decision == "cancel":
			state["cancelled"] = True
			return state
		if isinstance(decision, dict) and decision.get("action") == "edit":
			state["details"] = {**state.get("details", {}), **decision.get("details", {})}
			continue
		configurable = config.get("configurable", {})
		llm: GrievanceLLM | None = configurable.get("llm_service")
		if not llm:
			raise RuntimeError("An LLM service is required to generate chatbot responses.")
		state["response"] = llm.generate_reply(
			"",
			"registration",
			"Ask the user to choose approve, edit, or cancel for the pending grievance draft.",
			{"known_details": state.get("details", {})},
		)


def route_approval(state: ChatState) -> Literal["approval", "done"]:
	return "approval" if state.get("action") == "review" else "done"


def build_graph():
	graph = StateGraph(ChatState)
	graph.add_node("identify_intent", identify_intent)
	graph.add_node("extract_grievance_details", extract_grievance_details)
	graph.add_node("collect_registration_details", collect_registration_details)
	graph.add_node("check_missing_information", check_missing_information)
	graph.add_node("generate_follow_up_question", generate_follow_up_question)
	graph.add_node("ask_for_missing_details", ask_for_missing_details)
	graph.add_node("tracking", explain_tracking)
	graph.add_node("escalation", explain_escalation)
	graph.add_node("approval", wait_for_approval)
	graph.set_entry_point("identify_intent")
	graph.add_conditional_edges(
		"identify_intent",
		route_intent,
		{
			"registration": "extract_grievance_details",
			"tracking": "tracking",
			"escalation": "escalation",
			"general": END,
		},
	)
	graph.add_edge("extract_grievance_details", "collect_registration_details")
	graph.add_edge("collect_registration_details", "check_missing_information")
	graph.add_edge("check_missing_information", "generate_follow_up_question")
	graph.add_edge("generate_follow_up_question", "ask_for_missing_details")
	graph.add_conditional_edges("ask_for_missing_details", route_approval, {"approval": "approval", "done": END})
	graph.add_edge("approval", END)
	graph.add_edge("tracking", END)
	graph.add_edge("escalation", END)
	return graph.compile(checkpointer=MemorySaver())


workflow = build_graph()
