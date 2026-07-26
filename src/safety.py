"""
safety.py — a small, transparent guardrail layer for MediCare AI.

This is intentionally simple: a keyword check, not a clinical triage model.
The idea is that RAG + an LLM is useful for general health information, but
a possible medical or mental-health emergency should never be routed through
an LLM at all — it should immediately point the person to real help. This
"circuit breaker" pattern (checked BEFORE the RAG chain runs, see app.py)
is a common and easy-to-explain safety design, which makes it a good talking
point for a project viva.

NOTE for Pritam: the phone numbers below are for India (since that's the
most likely audience for a college project) plus a couple of international
defaults. If you're demoing this outside India, swap in your own country's
numbers before you present.
"""

from typing import List

EMERGENCY_KEYWORDS: List[str] = [
    # Physical emergencies
    "chest pain", "can't breathe", "cant breathe", "cannot breathe",
    "difficulty breathing", "trouble breathing", "not breathing",
    "severe bleeding", "heavy bleeding", "bleeding a lot",
    "unconscious", "unresponsive", "fainted", "passed out",
    "heart attack", "stroke", "seizure", "convulsion",
    "overdose", "poisoning", "severe allergic reaction", "anaphylaxis",
    "can't move my", "cant move my", "slurred speech",
    # Mental health emergencies
    "suicide", "suicidal", "kill myself", "want to die", "end my life",
    "self harm", "self-harm", "hurting myself",
]

EMERGENCY_MESSAGE = (
    "**This sounds like it could be a medical emergency.** Please contact "
    "emergency services or go to the nearest emergency room right now — "
    "don't wait on a chatbot for this.\n\n"
    "**India — Emergency services:** 112 · **Ambulance:** 108\n\n"
    "**India — Mental health crisis:** Tele-MANAS 14416 / 1-800-891-4416 "
    "(24x7, Govt. of India) · KIRAN 1800-599-0019 (24x7)\n\n"
    "**Outside India:** please use your local emergency number "
    "(for example 911 in the US, 999 in the UK, 112 across the EU).\n\n"
    "Once you're safe and have reached out for help, I'm happy to share "
    "general reference information."
)


def detect_emergency(text: str) -> bool:
    """Return True if the message contains a phrase suggesting a possible
    medical or mental-health emergency."""
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in EMERGENCY_KEYWORDS)
