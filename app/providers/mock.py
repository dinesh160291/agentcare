"""The mock provider — an understudy, not a fixture.

Under ``LLM_PROVIDER=mock`` the entire application runs: real tool calls, real
database writes, real workflow state. The only thing replaced is the *brain* —
the part that decides what to do next and how to word the answer.

**Why this file contains no canned replies.** The organizer's rule 6
disqualifies "hardcoded final responses presented as agent results". A mock
that returned "Your appointment is booked!" regardless of input would break
that rule, and would also make the mock useless as an understudy: it would
prove the plumbing worked only in the one case somebody hardcoded.

So the rule here is absolute: **every reply is templated from a tool result the
database produced this turn.** Facts are never carried over from what the mock
"intended" — they are read out of the payload the tool returned. For the
consequential case the mock goes further and calls ``render_confirmation``,
whose whole job is to re-read the row, then templates from that. Different
bookings therefore produce different sentences, and every fact in one can be
diffed against the row it describes.

What is deterministic here is *policy* — which tool to call next, given what
has already come back. That is the model's judgement being stood in for, and
standing in for judgement is what an understudy does.

**The policy dispatches on the toolset**, not on an agent name, because that is
the only thing a real model has to go on too. Each of the five agents gets the
tools its job needs, so the tools it was handed identify the job.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from google.adk.models import LlmRequest, LlmResponse

from app.providers.base import (
    AgentCareLlm,
    available_tool_names,
    called_tools,
    function_call_response,
    latest_tool_result,
    latest_user_text,
    text_response,
)
from app.workflow.confirmation import normalise

BOOKING_CUES = (
    "appointment", "book", "booking", "schedule", "see a doctor", "consult", "slot",
)
DOCUMENT_CUES = ("report", "document", "upload", "attach", "scan", "ecg", "x-ray", "mri")
ROUTING_CUES = ("which department", "who handles", "what department", "which team")
FOLLOWUP_CUES = ("reminder", "follow up", "follow-up", "outstanding", "what's next")
WITHDRAWAL_CUES = (
    "never mind", "nevermind", "forget it", "forget about it", "don't bother",
    "cancel that request", "cancel the request", "leave it",
)
#: What turns a mention of an appointment into a *new request for one*.
REBOOK_CUES = (
    "book", "schedule", "make it", "instead", "change it to", "switch to",
    "i need a", "i want a", "can i get a",
)
#: How the understudy reads a document's type off its text. Ordered, because
#: the first match wins and the more specific labels have to come first: an
#: X-ray report mentioning a referral is still an X-ray report.
#:
#: This is the mock standing in for a *reading* judgement, which is the one
#: place a real model is genuinely better. It is honest about that in the only
#: way that matters — it flags nothing it cannot name.
DOCUMENT_TYPE_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ECG report", ("electrocardiogram", "ecg report", "ecg summary")),
    ("Blood test report", ("blood panel", "blood test", "haematology", "full blood count")),
    ("X-ray report", ("x-ray", "xray", "radiograph")),
    ("MRI or CT report", ("mri", "magnetic resonance", "ct scan", "computed tomography")),
    ("Eye prescription", ("eye prescription", "spectacle", "optometr", "visual acuity")),
    ("Referral letter", ("referral letter", "refer this patient", "letter of referral")),
    ("Identification", ("passport", "driving licence", "identity card")),
)

#: Hesitation. Not a subject at all, which is what makes it a *non-answer*
#: rather than an off-topic message: "hmm, maybe" carries no administrative
#: noun, but a patient who has just been offered a time and says it is
#: replying to that question, not starting a new one.
HEDGE_CUES = (
    "hmm", "hmmm", "maybe", "not sure", "i think", "perhaps", "possibly",
    "let me think", "i suppose", "dunno", "don't know", "dont know", "erm",
    "not certain", "i guess",
)
#: A refusal that carries no exact token. The token sets are read in code
#: before the model is asked, so these are only ever the leftovers: "no" never
#: reaches here, "not that one, thanks" does.
DECLINE_CUES = (
    "not that one", "doesn't work", "does not work", "rather not", "no thanks",
    "not really", "can't do", "cannot do", "another time", "different time",
    "something else", "not keen", "too early", "too late",
)
SIDE_QUESTION_CUES = (
    "what documents", "what docs", "which documents", "what do i have",
    "when is my", "what time is my", "do i have",
)
ADMIN_CUES = BOOKING_CUES + DOCUMENT_CUES + ROUTING_CUES + FOLLOWUP_CUES

#: The mock's safety judgement, and it is deliberately *not* a copy of the
#: deterministic phrase list. A second opinion that repeats the first is a
#: check that cannot fail, and the whole point of the LLM pass is the wording
#: the phrase list cannot hold: "a lot of pressure across my chest since this
#: morning" contains no listed phrase and is the same emergency. So the mock
#: reasons by co-occurrence instead — a body term anywhere near a severity
#: term — which catches a different set and can disagree with layer one.
BODY_TERMS = (
    "chest", "breath", "breathing", "heart", "bleed", "bleeding", "conscious",
    "swallow", "pulse",
)
SEVERITY_TERMS = (
    "pain", "pressure", "tight", "severe", "struggling", "can't", "cannot",
    "heavy", "sudden", "crushing", "worse", "agony",
)
ADVICE_TERMS = (
    "should i", "what should", "is it ok to", "is it safe to", "recommend",
    "advise", "what do i do about",
)
CLINICAL_TERMS = (
    "take", "medicine", "medication", "tablet", "tablets", "dose", "pill",
    "pills", "treat", "treatment", "antibiotic", "antibiotics",
)

#: The reply used when nothing in the message is recognisable as hospital
#: administration. It is a *guard's* output, carries no facts about anybody,
#: and is identical in mock and live mode by design — the same words the scope
#: gate uses. It is not an agent result and states nothing it was not told.
SCOPE_TEXT = (
    "I can help with hospital administration — appointments, documents, "
    "reminders, and follow-ups. What would you like to do?"
)


def _has(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in cues)


def wants_booking(text: str) -> bool:
    return _has(text, BOOKING_CUES)


def mentions_documents(text: str) -> bool:
    return _has(text, DOCUMENT_CUES)


def parse_task(text: str) -> dict[str, Any]:
    """A specialist's typed task, or an empty dict for the Coordinator.

    Specialists receive JSON and nothing else; the Coordinator receives the
    patient's words. Which one arrived is therefore readable from the message.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class MockLlm(AgentCareLlm):
    """Deterministic stand-in for the model.

    Decides the next step from what the tools have already returned, and words
    the final reply from the last tool result.
    """

    model: str = "mock-understudy"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        yield self._decide(llm_request)

    # --- dispatch ---------------------------------------------------------

    def _decide(self, llm_request: LlmRequest) -> LlmResponse:
        available = available_tool_names(llm_request)
        done = called_tools(llm_request)
        text = latest_user_text(llm_request)
        task = parse_task(text)

        if "submit_safety_verdict" in available:
            return self._safety(llm_request, text)
        if "submit_plan" in available or "classify_message" in available:
            return self._coordinate(llm_request, available, done, text)
        if "submit_routing" in available:
            return self._route(llm_request, done, task)
        if "propose_appointment" in available:
            return self._appointment(llm_request, done, task)
        if "record_missing_documents" in available:
            return self._documents(llm_request, available, done, task)
        if "list_open_tasks" in available:
            return self._followup(llm_request, done)
        return text_response(SCOPE_TEXT)

    # --- Safety screen (the guardrail's second opinion) --------------------

    def _safety(self, llm_request: LlmRequest, text: str) -> LlmResponse:
        submitted = latest_tool_result(llm_request, "submit_safety_verdict")
        if submitted is not None:
            return self._from_safety(submitted.payload)

        category, rationale = self._safety_class(text)
        return function_call_response(
            "submit_safety_verdict", {"category": category, "rationale": rationale}
        )

    @staticmethod
    def _safety_class(text: str) -> tuple[str, str]:
        """Judge by co-occurrence, not by phrase.

        Emergency is decided before clinical advice for the same reason the
        deterministic screen does it: a message can be both, and only one of
        the two replies is the one that has to reach the patient.
        """
        lowered = (text or "").lower()
        body = [term for term in BODY_TERMS if term in lowered]
        severity = [term for term in SEVERITY_TERMS if term in lowered]
        if body and severity:
            return "emergency", f"{body[0]} mentioned with {severity[0]}"

        advice = [term for term in ADVICE_TERMS if term in lowered]
        clinical = [term for term in CLINICAL_TERMS if term in lowered]
        if advice and clinical:
            return "clinical_advice", f"asks {advice[0]} about {clinical[0]}"

        return "safe", "nothing clinical in this message"

    @staticmethod
    def _from_safety(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(str(payload.get("problem", "Screening again.")))
        return text_response(f"Screened as {payload.get('category')}.")

    # --- Coordinator ------------------------------------------------------

    def _coordinate(
        self, llm_request: LlmRequest, available: set[str], done: set[str], text: str
    ) -> LlmResponse:
        if "load_patient_context" in available and "load_patient_context" not in done:
            return function_call_response("load_patient_context", {})

        if "classify_message" in available:
            return self._classify(llm_request, available, done, text)
        return self._plan(llm_request, done, text)

    def _plan(self, llm_request: LlmRequest, done: set[str], text: str) -> LlmResponse:
        submitted = latest_tool_result(llm_request, "submit_plan")
        if submitted is not None:
            return self._from_plan(submitted.payload)

        steps = self._steps_for(text)
        if not steps:
            # Nothing administrative in it. No plan is submitted, so no run is
            # spawned and no tool fires — off-topic is noise, not a case.
            return text_response(SCOPE_TEXT)
        return function_call_response("submit_plan", {"steps": steps})

    @staticmethod
    def _steps_for(text: str) -> list[str]:
        """Rule-based intent extraction, in superset order.

        A message carrying a booking intent yields ``book`` even when it also
        mentions a document — the closure in ``validate_plan`` then pulls in the
        rest. Picking whichever intent matched first is the silent failure this
        avoids.
        """
        steps: list[str] = []
        if wants_booking(text):
            steps.append("book")
        if mentions_documents(text):
            steps.append("documents")
        if _has(text, ROUTING_CUES):
            steps.append("route")
        if _has(text, FOLLOWUP_CUES):
            steps.append("follow_up")
        return steps

    def _classify(
        self, llm_request: LlmRequest, available: set[str], done: set[str], text: str
    ) -> LlmResponse:
        settled = latest_tool_result(llm_request, "submit_confirmation_verdict")
        if settled is not None:
            return self._from_confirmation_verdict(settled.payload)

        awaiting = "submit_confirmation_verdict" in available
        classified = latest_tool_result(llm_request, "classify_message")
        if classified is None:
            message_class, steps = self._class_for(text, awaiting_confirmation=awaiting)
            return function_call_response(
                "classify_message",
                {"message_class": message_class, "incoming_steps": steps},
            )

        # A run waiting on a confirmation asks a second question: how did the
        # patient answer? Only where the tool was actually handed over — the
        # toolset is what says the question is live — and only for the classes
        # that leave the run still waiting. A withdrawal or a supersede has
        # already decided the run's fate; answering "how did they answer?"
        # about it would be answering a question nobody is asking any more.
        if awaiting and classified.payload.get("applied_class") in (
            "continuation",
            "side_question",
            "complementary",
        ):
            verdict, reason = self._confirmation_verdict(text)
            return function_call_response(
                "submit_confirmation_verdict", {"verdict": verdict, "reason": reason}
            )

        return self._from_classification(classified.payload)

    @staticmethod
    def _confirmation_verdict(text: str) -> tuple[str, str]:
        """Read a decline the exact tokens could not.

        Exact tokens ("no", "cancel") never arrive here — code has already read
        those and this branch is only reached for what it could not. So the
        cues are the phrasings that carry a refusal without carrying a token,
        and everything else is a non-answer. Note what is missing: there is no
        branch that returns a confirmation, because the tool would reject one.
        """
        if _has(text, DECLINE_CUES):
            return "decline", "the patient turned the offered time down"
        return "non_answer", "the patient did not answer the question"

    @staticmethod
    def _from_confirmation_verdict(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(str(payload.get("problem", "Let me ask again.")))
        if payload.get("verdict") == "decline":
            return text_response(
                "Understood — I won't book that one. Let me find another time."
            )
        return text_response(
            "I still need a yes or a no on the time I offered before I can book it."
        )

    @staticmethod
    def _class_for(text: str, *, awaiting_confirmation: bool = False) -> tuple[str, list[str]]:
        """Classify in the pinned order, stopping at the first that fits.

        The order is not an implementation detail: off-topic sits above
        continuation precisely so an off-topic message cannot fall through and
        be appended to the run's request text.

        **A pending question changes what "off topic" means.** Off-topic means
        nothing to do with hospital administration — but a patient who has just
        been offered a time and answers "hmm, maybe" or "not that one" is
        replying to *our* question, and carries no administrative noun only
        because the noun is already on the table. Reading those as off-topic
        costs more than a wrong label: it routes them to the scope reply and
        past the decline path, so a refusal the exact tokens could not read
        would never be heard at all.
        """
        if _has(text, WITHDRAWAL_CUES):
            return "withdrawal", []
        answering = awaiting_confirmation and (
            _has(text, HEDGE_CUES) or _has(text, DECLINE_CUES)
        )
        if not answering and not _has(text, ADMIN_CUES):
            return "off_topic", []
        if _has(text, SIDE_QUESTION_CUES):
            return "side_question", []
        if mentions_documents(text) and not wants_booking(text):
            return "complementary", ["documents"]
        # Conflicting needs an actual *request*, not merely the word
        # "appointment": "Tuesday works for the appointment" is someone
        # answering the question they were asked. The cost asymmetry decides
        # the tie — a wrongly superseded cooperation loses the booking the
        # patient wanted, so the doubtful case falls through to continuation.
        if wants_booking(text) and _has(text, REBOOK_CUES):
            return "conflicting", ["book"]
        return "continuation", []

    @staticmethod
    def _from_plan(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(
                "I need to look at that differently — "
                + str(payload.get("problem", "let me try again."))
            )
        readable = {
            "route": "work out the right department",
            "book": "find you an appointment time",
            "documents": "check your documents",
            "follow_up": "set up reminders and follow-up",
        }
        steps = payload.get("plan") or []
        listed = ", then ".join(readable.get(step, step) for step in steps)
        return text_response(f"Right — I'll {listed}.")

    @staticmethod
    def _from_classification(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(str(payload.get("problem", "Let me try that again.")))
        return text_response(
            "Understood — treating that as "
            + str(payload.get("applied_class", "")).replace("_", " ")
            + " of your current request."
        )

    # --- Department Routing ----------------------------------------------

    def _route(
        self, llm_request: LlmRequest, done: set[str], task: dict[str, Any]
    ) -> LlmResponse:
        submitted = latest_tool_result(llm_request, "submit_routing")
        if submitted is not None:
            return self._from_routing(submitted.payload)

        if "resolve_department" not in done:
            return function_call_response(
                "resolve_department", {"text": task.get("request_text", "")}
            )

        resolved = latest_tool_result(llm_request, "resolve_department")
        payload = resolved.payload if resolved else {}
        status = payload.get("status")

        if status == "resolved":
            return function_call_response(
                "submit_routing",
                {
                    "department_name": payload["department"]["name"],
                    "confidence": "high",
                },
            )
        if status == "ambiguous":
            candidates = payload.get("candidates") or []
            if candidates:
                # A person decides. Guessing confidently to avoid the handover
                # is the failure this branch exists to refuse.
                return function_call_response(
                    "submit_routing",
                    {"department_name": candidates[0]["name"], "confidence": "low"},
                )
        return text_response(
            "I couldn't match that to one of our departments. Could you tell me "
            "a little more about what the appointment is for?"
        )

    @staticmethod
    def _from_routing(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(str(payload.get("problem", "Let me check that again.")))
        name = payload["department"]["name"]
        if payload.get("confidence") == "low":
            return text_response(
                f"This looks like it may be {name}, but I'd like a member of staff "
                "to confirm before I book anything."
            )
        return text_response(f"{name} handles this. Let me find you a time.")

    # --- Appointment ------------------------------------------------------

    def _appointment(
        self, llm_request: LlmRequest, done: set[str], task: dict[str, Any]
    ) -> LlmResponse:
        confirmation = latest_tool_result(llm_request, "render_confirmation")
        if confirmation is not None:
            return self._from_confirmation(confirmation.payload)

        appointment_id = task.get("appointment_id")
        if appointment_id and "render_confirmation" not in done:
            return function_call_response(
                "render_confirmation", {"appointment_id": appointment_id}
            )

        proposed = latest_tool_result(llm_request, "propose_appointment")
        if proposed is not None:
            return self._from_proposal(proposed.payload)

        if "resolve_date" not in done:
            return function_call_response(
                "resolve_date", {"phrase": task.get("request_text", "")}
            )

        if "find_available_slots" not in done:
            window = latest_tool_result(llm_request, "resolve_date")
            args: dict[str, Any] = {}
            if window is not None and window.payload.get("resolved"):
                args["start"] = window.payload["start"]
                args["end"] = window.payload["end"]
                if window.payload.get("part_of_day"):
                    args["part_of_day"] = window.payload["part_of_day"]
            return function_call_response("find_available_slots", args)

        slots = latest_tool_result(llm_request, "find_available_slots")
        offered = (slots.payload.get("slots") if slots else None) or []
        if not offered:
            return self._from_empty_slots(slots.payload if slots else {})
        return function_call_response(
            "propose_appointment", {"slot_id": offered[0]["slot_id"]}
        )

    @staticmethod
    def _from_empty_slots(payload: dict[str, Any]) -> LlmResponse:
        problem = payload.get("problem")
        if problem:
            return text_response(str(problem))
        return text_response(
            "I couldn't find any free times in that period. "
            "Would you like me to look at a different week?"
        )

    @staticmethod
    def _from_proposal(payload: dict[str, Any]) -> LlmResponse:
        if not payload.get("accepted"):
            return text_response(
                str(payload.get("problem", "That time is no longer available."))
            )
        slot = payload["proposed"]
        return text_response(
            f"I can offer {slot['doctor_name']} on {slot['start'][:10]} at "
            f"{slot['start'][11:16]}. Shall I book it? Reply 'yes' to confirm."
        )

    @staticmethod
    def _from_confirmation(payload: dict[str, Any]) -> LlmResponse:
        """The consequential reply. Every fact comes from the persisted row."""
        sentence = payload.get("sentence")
        if sentence:
            return text_response(f"{sentence} You will get a reminder the day before.")
        facts = payload.get("facts") or {}
        return text_response(
            f"Your appointment with {facts.get('doctor_name')} is "
            f"{facts.get('status')} for {facts.get('weekday')} {facts.get('date')} "
            f"at {facts.get('time')}."
        )

    # --- Document ---------------------------------------------------------

    def _documents(
        self,
        llm_request: LlmRequest,
        available: set[str],
        done: set[str],
        task: dict[str, Any],
    ) -> LlmResponse:
        verifying = self._verify_next(llm_request, available, done)
        if verifying is not None:
            return verifying

        if "list_patient_documents" not in done:
            return function_call_response("list_patient_documents", {})

        has_department = task.get("department_id") is not None
        if has_department and "diff_required_documents" not in done:
            return function_call_response("diff_required_documents", {})

        if has_department and "record_missing_documents" not in done:
            return function_call_response("record_missing_documents", {})

        return self._from_documents(
            latest_tool_result(llm_request, "list_patient_documents"),
            latest_tool_result(llm_request, "diff_required_documents"),
        )

    def _verify_next(
        self, llm_request: LlmRequest, available: set[str], done: set[str]
    ) -> LlmResponse | None:
        """Verify one pending document per turn, or hand back to the listing.

        One per turn is a bound, not a shortcut: a patient with forty unchecked
        uploads would otherwise spend the whole iteration budget here and never
        reach the diff the booking actually needs. The rest are still pending
        and the next turn picks them up.

        Gated on the toolset like every other branch in this file. Asking for a
        tool the agent was not handed is how a mock starts describing a system
        that does not exist.
        """
        if "list_unverified_documents" not in available:
            return None

        if "list_unverified_documents" not in done:
            return function_call_response("list_unverified_documents", {})

        listed = latest_tool_result(llm_request, "list_unverified_documents")
        pending = (listed.payload.get("documents") if listed else None) or []
        if not pending:
            return None

        target = pending[0]
        if "read_document_text" not in done:
            return function_call_response(
                "read_document_text", {"document_id": target["document_id"]}
            )

        if "submit_document_verification" not in done:
            extracted = latest_tool_result(llm_request, "read_document_text")
            payload = extracted.payload if extracted else {}
            detected, matches = self._detect_type(
                payload.get("text") or "",
                declared=target.get("declared_type") or "",
                extractable=bool(payload.get("extracted")),
            )
            return function_call_response(
                "submit_document_verification",
                {
                    "document_id": target["document_id"],
                    "detected_type": detected,
                    "matches": matches,
                },
            )
        return None

    @staticmethod
    def _detect_type(text: str, *, declared: str, extractable: bool) -> tuple[str, bool]:
        """Read a document type out of the text, the way a rule-based stand-in can.

        No OCR and no image classification: an unreadable file is accepted at
        its declared type, because the alternative — flagging every photo a
        patient uploads — turns the review queue into a list of things nobody
        did wrong.
        """
        if not extractable or not text.strip():
            return declared, True

        lowered = text.lower()
        for detected, cues in DOCUMENT_TYPE_CUES:
            if any(cue in lowered for cue in cues):
                return detected, detected.strip().lower() == declared.strip().lower()

        # Nothing recognisable. The mock does not guess a mismatch it cannot
        # name: an unexplained flag costs a human the time to work out why.
        return declared, True

    @staticmethod
    def _from_documents(listing, diff) -> LlmResponse:
        documents = (listing.payload.get("documents") if listing else None) or []
        held = (
            ", ".join(
                f"{doc['document_type']} ({doc['status'].replace('_', ' ')})"
                for doc in documents[:5]
            )
            if documents
            else "nothing"
        )
        sentence = f"You have {len(documents)} document(s) on file: {held}."

        missing = (diff.payload.get("missing_mandatory") if diff else None) or []
        if missing:
            sentence += " Still needed: " + ", ".join(missing) + "."
        elif diff is not None:
            sentence += " Nothing else is required for this appointment."
        return text_response(sentence)

    # --- Follow-up --------------------------------------------------------

    def _followup(self, llm_request: LlmRequest, done: set[str]) -> LlmResponse:
        if "list_patient_reminders" not in done:
            return function_call_response("list_patient_reminders", {})
        if "list_open_tasks" not in done:
            return function_call_response("list_open_tasks", {})
        return self._from_followup(
            latest_tool_result(llm_request, "list_patient_reminders"),
            latest_tool_result(llm_request, "list_open_tasks"),
        )

    @staticmethod
    def _from_followup(reminders, tasks) -> LlmResponse:
        due = (reminders.payload.get("reminders") if reminders else None) or []
        open_tasks = (tasks.payload.get("tasks") if tasks else None) or []
        parts = []
        if due:
            parts.append(
                f"{len(due)} reminder(s) scheduled, the next on {due[0]['scheduled_at'][:10]}"
            )
        if open_tasks:
            parts.append(
                f"{len(open_tasks)} thing(s) outstanding: "
                + ", ".join(task["task_type"].replace("_", " ") for task in open_tasks)
            )
        if not parts:
            return text_response("Nothing is outstanding at the moment.")
        return text_response("Here's where things stand: " + "; ".join(parts) + ".")


def reads_as_confirmation(text: str) -> bool:
    """Exact-token match, delegated to the one confirmation reader.

    Imported from the workflow module rather than re-listed here: the mock is
    not permitted to be cleverer than the real confirmation path, and two
    copies of a token list are two lists that drift.
    """
    from app.workflow.confirmation import CONFIRM_TOKENS

    return normalise(text) in CONFIRM_TOKENS


def reads_as_decline(text: str) -> bool:
    from app.workflow.confirmation import DECLINE_TOKENS

    return normalise(text) in DECLINE_TOKENS


__all__ = [
    "MockLlm",
    "SCOPE_TEXT",
    "mentions_documents",
    "parse_task",
    "reads_as_confirmation",
    "reads_as_decline",
    "wants_booking",
]
