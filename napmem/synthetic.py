from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .pyramid import MemoryPyramid
from .schema import MemoryRecord, Message

AnswerMode = Literal["exact_string", "semantic"]


@dataclass(frozen=True)
class QAExample:
    qid: str
    user_id: str
    question: str
    answer: str
    support: tuple[str, ...]
    tag: str
    requires_memory: bool = True
    answer_mode: AnswerMode = "exact_string"
    evidence_quote: str = ""


@dataclass
class SyntheticBenchmark:
    pyramid: MemoryPyramid
    examples: list[QAExample] = field(default_factory=list)


@dataclass(frozen=True)
class UserMemorySpec:
    user_id: str
    raw_message: str
    record_content: str
    topic_file: str
    topic_markdown: str
    profile_line: str
    raw_question: str
    raw_answer: str
    raw_quote: str
    record_question: str
    record_answer: str
    record_quote: str
    topic_question: str
    topic_answer: str
    topic_quote: str
    profile_question: str
    profile_answer: str
    profile_quote: str
    raw_qid: str = ""
    record_qid: str = ""
    topic_qid: str = ""
    profile_qid: str = ""
    topic_mode: AnswerMode = "semantic"


USER_SPECS = [
    UserMemorySpec(
        "nate",
        "I visited Tampa beach on the turtle trip because I needed peace.",
        "Nate visited Tampa beach on a turtle trip for peace and relaxation.",
        "nate-travel.md",
        "# Nate Travel\n\nSummary: Nate traveled to Tampa beach during a turtle trip.\n\n"
        "Evidence: r_nate.\n\nInference anchor: Tampa is in Florida.\n",
        "Nate prefers peaceful beach trips.",
        "Which exact beach did Nate visit on the turtle trip?",
        "Tampa beach",
        "visited Tampa beach",
        "Why did Nate visit Tampa beach?",
        "peace and relaxation",
        "peace and relaxation",
        "What state did Nate visit on the turtle trip?",
        "Florida",
        "Tampa is in Florida",
        "What kind of trips does Nate prefer?",
        "peaceful beach trips",
        "peaceful beach trips",
        topic_qid="q_topic_state",
    ),
    UserMemorySpec(
        "mira",
        "Please avoid almonds in snack suggestions; they make my throat itch.",
        "Mira wants snack suggestions to avoid a nut ingredient that causes throat irritation.",
        "mira-food.md",
        "# Mira Food\n\nSummary: Mira avoids almonds in snack planning.\n\nEvidence: r_mira.\n",
        "Mira needs snack suggestions that avoid almonds.",
        "Which exact ingredient should Mira avoid in snack suggestions?",
        "almonds",
        "avoid almonds",
        "What reaction does Mira get from the snack ingredient?",
        "throat irritation",
        "throat irritation",
        "What food planning constraint does Mira have?",
        "avoid almonds",
        "avoids almonds",
        "What should Mira's snack suggestions avoid?",
        "almonds",
        "avoid almonds",
        raw_qid="q_raw_allergy",
    ),
    UserMemorySpec(
        "linh",
        "When we work on research plans, answer me in Vietnamese first.",
        "For research plans, answer Linh in Vietnamese first.",
        "linh-research.md",
        "# Linh Research\n\nSummary: Linh wants research-plan answers in Vietnamese first.\n\nEvidence: r_linh.\n",
        "Linh wants Vietnamese first for research-plan work.",
        "For what work context did Linh request Vietnamese first?",
        "research plans",
        "research plans",
        "For research plans, which language should come first?",
        "Vietnamese",
        "Vietnamese first",
        "How should research-plan answers for Linh begin?",
        "Vietnamese first",
        "Vietnamese first",
        "What is Linh's language preference for research-plan work?",
        "Vietnamese first",
        "Vietnamese first",
        record_qid="q_record_instruction",
    ),
    UserMemorySpec(
        "jo",
        "I like short weekly summaries with crisp bullets and no fluff.",
        "Jo likes concise weekly summaries.",
        "jo-workflow.md",
        "# Jo Workflow\n\nSummary: Jo prefers crisp bullets in weekly summaries.\n\nEvidence: r_jo.\n",
        "Jo prefers crisp bullets in weekly summaries.",
        "What should Jo's weekly summaries avoid?",
        "fluff",
        "no fluff",
        "What kind of weekly summaries does Jo like?",
        "concise weekly summaries",
        "concise weekly summaries",
        "How should Jo's weekly summaries be formatted?",
        "crisp bullets",
        "crisp bullets",
        "What style does Jo prefer for weekly summaries?",
        "crisp bullets",
        "crisp bullets",
        profile_qid="q_profile_style",
        topic_mode="exact_string",
    ),
    UserMemorySpec(
        "ava",
        "After 11 pm, please keep notifications quiet because my baby sleeps.",
        "Ava prefers quiet notifications after 11 pm.",
        "ava-home.md",
        "# Ava Home\n\nSummary: Ava keeps notifications quiet after 11 pm due to baby sleep.\n\nEvidence: r_ava.\n",
        "Ava has quiet-notification hours after 11 pm.",
        "Who is sleeping when Ava asks for quiet notifications?",
        "baby",
        "my baby sleeps",
        "After what time does Ava prefer quiet notifications?",
        "11 pm",
        "after 11 pm",
        "Why does Ava keep notifications quiet late at night?",
        "baby sleep",
        "baby sleep",
        "What notification boundary does Ava have?",
        "after 11 pm",
        "after 11 pm",
    ),
    UserMemorySpec(
        "bao",
        "Use invoice prefix BWP- for Build with Paper consulting bills.",
        "Bao uses invoice prefix BWP- for consulting bills.",
        "bao-billing.md",
        "# Bao Billing\n\nSummary: Bao's Build with Paper consulting invoices use BWP-.\n\nEvidence: r_bao.\n",
        "Bao's consulting invoice prefix is BWP-.",
        "What exact invoice prefix did Bao ask to use?",
        "BWP-",
        "prefix BWP-",
        "What is Bao's consulting invoice prefix?",
        "BWP-",
        "prefix BWP-",
        "Which project are Bao's BWP- invoices for?",
        "Build with Paper",
        "Build with Paper",
        "What prefix should Bao's consulting invoices use?",
        "BWP-",
        "BWP-",
        topic_mode="exact_string",
    ),
    UserMemorySpec(
        "cy",
        "I dislike cilantro; it makes soup taste soapy to me.",
        "Cy dislikes an herb because it makes soup taste soapy.",
        "cy-food.md",
        "# Cy Food\n\nSummary: Cy dislikes cilantro because it tastes soapy in soup.\n\nEvidence: r_cy.\n",
        "Cy dislikes cilantro in soup.",
        "Which exact herb does Cy dislike?",
        "cilantro",
        "dislike cilantro",
        "Why does Cy dislike the herb?",
        "soapy",
        "soapy",
        "In what dish does cilantro taste soapy to Cy?",
        "soup",
        "soup",
        "What herb should be avoided for Cy?",
        "cilantro",
        "cilantro",
        topic_mode="exact_string",
    ),
    UserMemorySpec(
        "diego",
        "I use Obsidian daily for lab notes, but keep final drafts in Google Docs.",
        "Diego uses Obsidian for lab notes and Google Docs for final drafts.",
        "diego-writing.md",
        "# Diego Writing\n\nSummary: Diego drafts final documents in Google Docs after taking lab notes in Obsidian.\n\n"
        "Evidence: r_diego.\n",
        "Diego uses Obsidian for lab notes and Google Docs for final drafts.",
        "How often does Diego use Obsidian for lab notes?",
        "daily",
        "daily",
        "Where does Diego keep final drafts?",
        "Google Docs",
        "Google Docs",
        "What is Diego's workflow from lab notes to final documents?",
        "Obsidian to Google Docs",
        "lab notes in Obsidian",
        "What tool does Diego use for lab notes?",
        "Obsidian",
        "Obsidian",
    ),
]

NON_MEMORY_EXAMPLES = [
    ("q_non_memory_math", "nate", "What is 2 + 2?", "4"),
    ("q_non_memory_capital", "mira", "What is the capital of France?", "Paris"),
    ("q_non_memory_water", "linh", "What chemical formula represents water?", "H2O"),
    ("q_non_memory_week", "jo", "How many days are in a week?", "7"),
    ("q_non_memory_color", "ava", "What color do you get by mixing red and white?", "pink"),
    ("q_non_memory_planet", "bao", "Which planet is known as the Red Planet?", "Mars"),
    ("q_non_memory_square", "cy", "What is 9 squared?", "81"),
    ("q_non_memory_author", "diego", "Who wrote Pride and Prejudice?", "Jane Austen"),
]


def build_synthetic_benchmark(root: str | Path) -> SyntheticBenchmark:
    """Build a balanced benchmark where evidence sits at different layers."""

    pyramid = MemoryPyramid(root)
    examples: list[QAExample] = []
    profile_lines = ["# Profile", ""]

    for idx, spec in enumerate(USER_SPECS, start=1):
        message_id = f"m_{spec.user_id}"
        record_id = f"r_{spec.user_id}"
        timestamp = f"2026-07-{idx:02d}T09:00:00"
        pyramid.append_message(
            Message(message_id, spec.user_id, f"s_{spec.user_id}", "user", spec.raw_message, timestamp)
        )
        pyramid.add_record(
            MemoryRecord(
                record_id,
                spec.user_id,
                "preference",
                spec.record_content,
                timestamp,
                timestamp,
                [message_id],
            )
        )
        pyramid.upsert_topic_track(spec.topic_file, spec.topic_markdown.replace(f"r_{spec.user_id}", record_id))
        profile_lines.append(f"- {spec.profile_line} Evidence: {record_id}.")
        examples.extend([
            QAExample(
                spec.raw_qid or f"q_raw_{spec.user_id}",
                spec.user_id,
                spec.raw_question,
                spec.raw_answer,
                (f"message:{message_id}",),
                "raw",
                True,
                "exact_string",
                spec.raw_quote,
            ),
            QAExample(
                spec.record_qid or f"q_record_{spec.user_id}",
                spec.user_id,
                spec.record_question,
                spec.record_answer,
                (f"record:{record_id}",),
                "record",
                True,
                "exact_string",
                spec.record_quote,
            ),
            QAExample(
                spec.topic_qid or f"q_topic_{spec.user_id}",
                spec.user_id,
                spec.topic_question,
                spec.topic_answer,
                (f"file:{spec.topic_file}",),
                "topic",
                True,
                spec.topic_mode,
                spec.topic_quote,
            ),
            QAExample(
                spec.profile_qid or f"q_profile_{spec.user_id}",
                spec.user_id,
                spec.profile_question,
                spec.profile_answer,
                ("file:profile.md",),
                "profile",
                True,
                "exact_string",
                spec.profile_quote,
            ),
        ])

    pyramid.update_profile("\n".join(profile_lines) + "\n")
    examples.extend([
        QAExample(qid, user_id, question, answer, (), "non_memory", False, "semantic", "")
        for qid, user_id, question, answer in NON_MEMORY_EXAMPLES
    ])
    return SyntheticBenchmark(pyramid, examples)
