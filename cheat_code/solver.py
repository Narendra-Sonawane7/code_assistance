"""
Question Solver — Detects questions and answers them using Groq API (Free).
Model: llama-3.3-70b-versatile — fast, high quality, free tier.
Multi-key rotation: automatically cycles to next key on rate-limit (429).
Add up to 3 free Groq keys in config → ~43,000 req/day total.
Supports Interview, Exam, and Meeting modes with tailored AI prompts.
"""

import re
import openai
from config import load_config
from typing import Callable, Optional

# ── Groq config ───────────────────────────────────────────────────────────────
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL    =  "llama-3.3-70b-versatile"    #"openai/gpt-oss-120b" 
GROQ_HEADERS  = {}   # Groq doesn't need extra headers

# ── Multi-key rotation state ──────────────────────────────────────────────────
# Cycles through all configured keys; skips exhausted ones automatically.
# Resets at midnight (app restart) or when keys are updated in settings.

_key_index: int = 0
_exhausted_keys: set = set()          # keys that hit 429 today
_client_cache: dict[str, openai.OpenAI] = {}  # key → cached client


def _get_keys() -> list[str]:
    """Return all non-empty API keys from config."""
    config = load_config()
    keys = [
        config.get("groq_api_key",  "").strip(),
        config.get("groq_api_key2", "").strip(),
        config.get("groq_api_key3", "").strip(),
    ]
    return [k for k in keys if k]


def _get_client(api_key: str) -> openai.OpenAI:
    """Return a cached OpenAI client for the given key."""
    if api_key not in _client_cache:
        _client_cache[api_key] = openai.OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
            default_headers=GROQ_HEADERS,
        )
    return _client_cache[api_key]


def _next_available_key() -> Optional[str]:
    """
    Round-robin through keys, skipping exhausted ones.
    Returns None if all keys are exhausted.
    """
    global _key_index
    keys = _get_keys()
    if not keys:
        return None
    for _ in range(len(keys)):
        key = keys[_key_index % len(keys)]
        _key_index = (_key_index + 1) % len(keys)
        if key not in _exhausted_keys:
            return key
    return None  # all exhausted


def _mark_exhausted(key: str):
    _exhausted_keys.add(key)


def reset_exhausted_keys():
    """Call this at midnight or when user adds new keys."""
    _exhausted_keys.clear()


# ── Mode Prompts ──────────────────────────────────────────────────────────────

MODE_PROMPTS = {
    "interview": """You are an expert interview coach and co-pilot assisting someone in a live job interview.
When you receive a question:

- BEHAVIORAL questions (Tell me about yourself / Describe a time / Give an example):
  → Give a polished STAR-format answer (Situation, Task, Action, Result). Keep each part 1-2 sentences.

- CODING questions (write a function / implement / solve / reverse / find / two sum etc.):
  → Write clean, working code with language specified or Python by default.
  → Add 2-3 line explanation below the code.

- SYSTEM DESIGN questions (design a URL shortener / architect a system):
  → Give structured answer: Components → Data Flow → Database → Scaling. Use bullet points.

- TECHNICAL CONCEPT questions (what is X / explain X / difference between X and Y):
  → Give a clear, accurate, concise explanation with a real-world example.

- GENERAL / OTHER:
  → Answer professionally and confidently in 3-5 sentences.

IMPORTANT: Be concise. Interviewer is listening. Keep total answer under 150 words unless it is a coding question.""",

    "exam": """You are a precise academic tutor helping someone during an exam or test.

- MCQ / True-False: State the correct answer immediately, then explain why in 1-2 sentences.
- Calculation / Math: Show step-by-step working clearly. Label each step.
- Theory / Definition: Give a textbook-accurate explanation in 3-5 sentences.
- Fill in the blank: Give the exact answer word/phrase first, then context.

Be accurate, direct, and brief. No filler words.""",

    "meeting": """You are a smart professional meeting assistant.

- If a question is asked: Answer it clearly and briefly.
- If a topic is discussed: Summarize the key point in 2-3 sentences.
- If a decision or action is mentioned: Extract and list action items.
- If technical jargon is used: Explain it simply.

Be professional, concise, and helpful.""",

    "hr": """You are an expert HR interview coach helping a candidate ace a human resources / behavioural round.

When you receive a question:

- INTRODUCTION / "Tell me about yourself":
  → Deliver a polished 3-part answer: (1) Who you are professionally, (2) Key achievements/skills, (3) Why this role/company. Keep it under 90 seconds when spoken (~120 words).

- BEHAVIOURAL questions (Tell me about a time / Describe a situation / Give an example):
  → Use STAR format (Situation → Task → Action → Result). Each part 1-2 sentences. End with a positive, quantified result where possible.

- STRENGTHS & WEAKNESSES:
  → Strengths: name it, give a concrete example, link it to the role.
  → Weakness: choose a real but non-critical one, show active improvement steps, frame it positively.

- MOTIVATION / FIT questions (Why this company? Why this role? Where do you see yourself in 5 years?):
  → Research-based, enthusiastic but measured. 3-4 sentences. Link personal goals to company mission.

- SALARY / COMPENSATION questions:
  → Guide the candidate to deflect or give a confident range: acknowledge the question, give a researched range, express flexibility.

- CONFLICT / TEAMWORK questions:
  → Emphasise empathy, communication, and resolution. Never blame others. Show maturity and professionalism.

- CULTURE FIT / VALUES questions:
  → Align answer with universal professional values (ownership, collaboration, growth mindset, integrity). Give a brief real example.

- GENERAL HR questions:
  → Answer confidently, professionally, in 3-5 sentences. Use positive framing.

IMPORTANT: Keep answers concise and conversational — the interviewer is listening. Use first-person. Total answer under 120 words unless STAR format needs more. Never use bullet points in the spoken answer — write as natural sentences.""",
}


# ── Question Detection ────────────────────────────────────────────────────────

QUESTION_PATTERNS = [
    r'\?',
    r'\b(?:what|who|where|when|why|how|which|whom|whose)\b',
    r'\b(?:is|are|was|were|do|does|did|can|could|will|would|shall|should|may|might)\s+\w+',
    r'\b(?:define|explain|describe|calculate|find|solve|evaluate|compute|determine|list|state|prove|derive)\b',
    r'\b(?:true or false|choose the correct|select the|pick the|fill in|which of the following)\b',
    r'\b(?:tell me|walk me|talk me)\b',
    r'\b(?:give me|give an|give your|give a)\b',
    r'\b(?:describe a|describe your|describe the|describe how)\b',
    r'\b(?:tell us|tell me about|about yourself|your background|your experience|your strength|your weakness)\b',
    r'\b(?:write|implement|code|build|create|develop|program|construct)\b',
    r'\b(?:design|architect|architecture|system for|how would you build|how would you design)\b',
    r'\b(?:difference between|compare|versus|vs\.|pros and cons|advantages of|disadvantages of)\b',
    r'\b(?:have you|have you ever|have you worked|have you used)\b',
    r'\b(?:how do you|how would you|how did you|how have you)\b',
    r'\b(?:what would you|what do you think|what is your)\b',
    r'\b(?:in your opinion|your approach|your strategy|your plan)\b',
    r'\b(?:tell me about yourself|about yourself|introduce yourself)\b',
    r'\b(?:strength|weakness|challenge|achievement|accomplishment)\b',
    r'\b(?:why this company|why this role|why should we hire|where do you see yourself)\b',
    r'\b(?:salary|compensation|expectation|package|ctc|notice period)\b',
    r'\b(?:conflict|disagreement|difficult colleague|team issue|feedback received)\b',
    r'\b(?:culture|values|work style|management style|work-life)\b',
]


def is_question(text: str) -> bool:
    if not text or len(text.strip()) < 8:
        return False
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower) for p in QUESTION_PATTERNS)


def extract_questions(text: str) -> list:
    if not text:
        return []
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    questions = [s.strip() for s in sentences if s.strip() and is_question(s)]
    if not questions and is_question(text):
        questions = [text.strip()]
    seen, unique = set(), []
    for q in questions:
        if q not in seen:
            seen.add(q); unique.append(q)
    return unique


# ── Build dynamic system prompt ───────────────────────────────────────────────

def _build_system_prompt(mode: str, language: str = None, resume: str = None) -> str:
    prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["interview"])

    ROLE_CONTEXTS = {
        "AI Engineer": (
            "\n\nROLE CONTEXT: The candidate is interviewing for an AI Engineer role. "
            "Default coding language is Python. Prioritise frameworks: PyTorch, TensorFlow, "
            "Hugging Face Transformers, LangChain, FastAPI. "
            "For coding questions use Python with type hints and docstrings. "
            "For concept questions cover: LLMs, RAG, fine-tuning, embeddings, vector databases, "
            "prompt engineering, model deployment, inference optimisation, MLOps, and AI system design. "
            "For system design questions focus on: LLM APIs, retrieval pipelines, latency/cost tradeoffs, "
            "evaluation frameworks, and serving infrastructure."
        ),
        "ML Engineer": (
            "\n\nROLE CONTEXT: The candidate is interviewing for an ML Engineer role. "
            "Default coding language is Python. Prioritise frameworks: PyTorch, scikit-learn, "
            "TensorFlow/Keras, XGBoost, pandas, numpy, MLflow, Airflow. "
            "For coding questions use Python with numpy-style operations and vectorised code. "
            "For concept questions cover: supervised/unsupervised learning, feature engineering, "
            "model evaluation metrics, bias-variance tradeoff, regularisation, ensemble methods, "
            "neural network architectures, backpropagation, and hyperparameter tuning. "
            "For system design questions focus on: ML pipelines, training infrastructure, "
            "data preprocessing, model versioning, A/B testing, monitoring, and production deployment."
        ),
        "HR Interview": (
            "\n\nROLE CONTEXT: This is an HR / behavioural round. "
            "Override all coding/technical instructions. There will be NO coding questions. "
            "Focus exclusively on: self-introduction, STAR-format behavioural answers, "
            "strengths & weaknesses, motivation & culture fit, conflict resolution, teamwork, "
            "salary/notice period, and career goals. "
            "Answer in FIRST PERSON, using natural spoken sentences — NO bullet points, NO code. "
            "Keep every answer under 120 words (≈60 seconds spoken). "
            "Always end behavioural answers with a positive outcome or lesson learned. "
            "For 'Tell me about yourself': structure as (1) current role/background, "
            "(2) key achievement, (3) why this opportunity. "
            "For strengths: name → example → link to role. "
            "For weaknesses: real but non-critical → active improvement → positive reframe. "
            "For salary questions: acknowledge → give a researched range → express flexibility. "
            "Never speak negatively about previous employers or colleagues."
        ),
    }

    if language and language not in ("Auto-Detect", "", None):
        if language in ROLE_CONTEXTS:
            prompt += ROLE_CONTEXTS[language]
        else:
            prompt += (
                f"\n\nLANGUAGE CONTEXT: The candidate is interviewing for a {language} role. "
                f"For ALL coding questions, write code in {language} by default unless the question "
                f"explicitly specifies a different language. Tailor examples and idioms to {language} "
                f"best practices."
            )

    if resume and resume.strip():
        prompt += (
            f"\n\nCANDIDATE RESUME:\n{resume.strip()}\n\n"
            "RESUME INSTRUCTIONS: You have the candidate's resume above. "
            "When asked about their experience, projects, skills, achievements, or "
            "'tell me about yourself', answer in FIRST PERSON as if you ARE the candidate. "
            "Reference specific project names, technologies, companies, and accomplishments "
            "from the resume. Be concrete and specific — don't give generic answers."
        )

    return prompt


# ── Streaming solver (PRIMARY PATH — audio + manual input) ───────────────────

def solve_streaming(
    question: str,
    mode: str = "interview",
    language: str = None,
    resume: str = None,
    conversation_history: list = None,
    on_token: Callable[[str], None] = None,
    on_done: Callable[[str], None] = None,
    on_error: Callable[[str], None] = None,
) -> None:
    """
    Stream the answer token-by-token with automatic key rotation.
    If the active key hits rate limit (429), silently switches to the next key.
    Calls on_token(token) per chunk, on_done(full_text) when complete.
    Runs synchronously; caller is responsible for running this in a thread.
    """
    api_key = _next_available_key()

    if not api_key:
        if on_error:
            on_error("⚠️ All API keys exhausted for today (200 req/key/day).\nAdd more keys via Right-click tray → Change API Key.")
        return

    system_prompt = _build_system_prompt(mode, language=language, resume=resume)

    history  = conversation_history or []
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        client = _get_client(api_key)
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=700,
            stream=True,
            messages=messages,
        )

        collected: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                collected.append(delta)
                if on_token:
                    on_token(delta)

        if on_done:
            on_done("".join(collected))

    except openai.AuthenticationError:
        if on_error: on_error("❌ Invalid Groq API key. Right-click tray → Change API Key.")
    except openai.RateLimitError:
        # Mark this key as exhausted and retry with the next one
        _mark_exhausted(api_key)
        next_key = _next_available_key()
        if next_key:
            if on_token: on_token("")   # clear any partial output
            solve_streaming(question, mode=mode, language=language, resume=resume,
                            conversation_history=conversation_history,
                            on_token=on_token, on_done=on_done, on_error=on_error)
        else:
            if on_error: on_error("⚠️ All API keys exhausted for today.\nAdd more keys via tray → Change API Key.")
    except openai.APIConnectionError:
        if on_error: on_error("❌ No internet connection. Check your network.")
    except Exception as e:
        if on_error: on_error(f"❌ Error: {str(e)}")


# ── Non-streaming solver (kept for screen-scan path) ─────────────────────────

def solve_with_claude(
    question: str,
    mode: str = "interview",
    language: str = None,
    resume: str = None,
    conversation_history: list = None,
) -> str:
    """
    Blocking version — used by the screen-scan path (process_text).
    Uses multi-key rotation; auto-switches key on rate limit.
    """
    api_key = _next_available_key()

    if not api_key:
        return "⚠️ All API keys exhausted for today. Add more keys via tray → Change API Key."

    system_prompt = _build_system_prompt(mode, language=language, resume=resume)

    history  = conversation_history or []
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    try:
        client = _get_client(api_key)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=700,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    except openai.AuthenticationError:
        return "❌ Invalid Groq API key. Right-click tray → Change API Key."
    except openai.RateLimitError:
        _mark_exhausted(api_key)
        # Retry once with next available key
        return solve_with_claude(question, mode=mode, language=language,
                                 resume=resume, conversation_history=conversation_history)
    except openai.APIConnectionError:
        return "❌ No internet connection. Check your network."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ── Main Entry Point (screen scan) ───────────────────────────────────────────

def process_text(
    text: str,
    mode: str = "interview",
    language: str = None,
    resume: str = None,
    conversation_history: list = None,
) -> list:
    questions = extract_questions(text)
    return [
        (q, solve_with_claude(q, mode=mode, language=language, resume=resume,
                               conversation_history=conversation_history))
        for q in questions
    ]