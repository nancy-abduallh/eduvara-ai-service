"""
pipeline/quiz_generator.py
===========================
Generates MCQ questions from a video script.
Ported from video-script-mcq-quiz-generator-updated_2.ipynb (Cells 4B, 13, 13.5).

Key design:
  - chunk_script()            → split script into ~400-word overlapping passages
  - generate_mcqs_from_script() → hybrid pipeline:
        concept definitions + code blocks + comparisons + statistics + model prose
  - Returns a list of dicts, each:
        { question, options:{A,B,C,D}, correct, correct_answer,
          difficulty, passage, section, qtype }
"""

from __future__ import annotations

import gc
import hashlib
import logging
import re
import random
import string
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict, Optional, Any

import numpy as np
import torch

logger = logging.getLogger("edugenie.quiz")

# ── Constants ────────────────────────────────────────────────────────────────

LETTERS = ["A", "B", "C", "D"]
MAX_INPUT  = 512
MAX_TARGET = 200

class DIF:
    EASY   = "🟢 Easy"
    MEDIUM = "🟡 Medium"
    HARD   = "🔴 Hard"

class QTYPES:
    recall     = "📖 Recall"
    concept    = "💡 Concept"
    code       = "💻 Code"
    comparison = "⚖️ Comparison"
    statistic  = "📊 Statistic"


# ── Sentence tokeniser (no NLTK needed for basic chunking) ───────────────────

def _sent_tokenize(text: str) -> List[str]:
    """Simple sentence splitter that avoids NLTK dependency issues."""
    try:
        from nltk.tokenize import sent_tokenize as nltk_sent
        return nltk_sent(text)
    except Exception:
        # Fallback: split on period/!/?
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p.strip()]


# ── Script chunker ────────────────────────────────────────────────────────────

def chunk_script(script_text: str, target_words: int = 400, overlap: int = 2) -> List[str]:
    """
    Split a video script into overlapping passages for MCQ generation.
    Ported from Cell 12 of the quiz notebook.
    """
    # Remove timestamps, speaker labels, extra whitespace
    text = re.sub(r"\[.*?\]", "", script_text)
    text = re.sub(r"^\s*[A-Z][A-Z\s]+:\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+:\d+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = _sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.split()) >= 5]

    if not sentences:
        return []

    chunks, i = [], 0
    while i < len(sentences):
        chunk_sents, word_count = [], 0
        j = i
        while j < len(sentences) and word_count < target_words:
            chunk_sents.append(sentences[j])
            word_count += len(sentences[j].split())
            j += 1
        if chunk_sents:
            chunks.append(" ".join(chunk_sents))
        # overlap: step back by `overlap` sentences
        i = max(i + 1, j - overlap)

    chunks = [c for c in chunks if len(c.split()) >= 50]
    return chunks


# ── Prompt builder ────────────────────────────────────────────────────────────

def make_prompt(passage: str) -> str:
    return (
        f"Passage: {passage.strip()}\n\n"
        "Task: Generate a professional multiple-choice question from the passage above.\n\n"
        "Requirements for a HIGH-QUALITY question:\n"
        "  - Ask about a concept, mechanism, cause/effect, or application — NOT a trivial definition\n"
        "  - The question must be a complete, grammatical sentence ending with '?'\n"
        "  - All 4 options must be plausible and domain-relevant (not generic filler)\n"
        "  - The 3 wrong options (distractors) must be clearly wrong for an expert "
        "but tempting for a novice\n"
        "  - Options must be roughly equal in length and grammatical form\n"
        "  - Options must NOT contain the words "
        "'Question', 'Correct', 'Option', 'Answer', 'Passage'\n\n"
        "Format EXACTLY like this (no extra lines):\n"
        "Question: <question ending with ?>\n"
        "A) <option>\n"
        "B) <option>\n"
        "C) <option>\n"
        "D) <option> (Correct)\n\n"
        "Generate the MCQ:"
    )


# ── MCQ parser ────────────────────────────────────────────────────────────────

def parse_mcq_output(text: str) -> dict:
    """Parse FLAN-T5 output — handles inline AND multiline formats."""
    empty = {"question": "", "options": {k: "" for k in LETTERS}, "correct": None}
    if not text or len(text.strip()) < 15:
        return empty

    text = text.strip()
    text = re.sub(r"^[Qq]uestion\s*:\s*", "", text).strip()
    result = {"question": "", "options": {k: "" for k in LETTERS}, "correct": None}

    # Extract question
    q_match = re.match(r"^(.+?\?)\s*(?:[A-D]\))", text, re.DOTALL)
    if q_match:
        result["question"] = q_match.group(1).strip()
    else:
        lines = text.split("\n")
        result["question"] = lines[0].strip().rstrip("?") + "?" if lines else ""

    # Extract options
    for letter in LETTERS:
        pat = rf"{letter}\)\s*(.+?)(?=\s*[A-D]\)|$)"
        m = re.search(pat, text, re.DOTALL)
        if m:
            opt = m.group(1).strip()
            if "(Correct)" in opt:
                opt = opt.replace("(Correct)", "").strip()
                result["correct"] = letter
            result["options"][letter] = opt

    return result


# ── Quality helpers ───────────────────────────────────────────────────────────

def _fingerprint(q: str) -> str:
    normalized = re.sub(r"\W+", " ", q.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def _jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _is_valid_option(opt: str) -> bool:
    if not opt or len(opt.strip()) < 2:
        return False
    bad_words = {"question", "correct", "option", "answer", "passage"}
    return not any(w in opt.lower() for w in bad_words)

def _is_surface_question(q: str) -> bool:
    surface = [
        r"^what is (the |a |an )?\w+\?$",
        r"^define ",
        r"^what does .* stand for",
    ]
    ql = q.strip().lower()
    return any(re.search(p, ql) for p in surface)

def _question_too_long(q: str) -> bool:
    return len(q.split()) > 60

def _classify_difficulty(question: str, correct: str) -> str:
    q_lower = question.lower()
    hard_signals = ["derive", "prove", "calculate", "complexity", "O(", "gradient", "algorithm"]
    easy_signals = ["what is", "define", "which of the following is", "what does"]
    if any(s in q_lower for s in hard_signals):
        return DIF.HARD
    if any(s in q_lower for s in easy_signals):
        return DIF.EASY
    return DIF.MEDIUM


# ── Domain detection ──────────────────────────────────────────────────────────

_DOMAIN_SIGNALS = {
    "cs_algo": ["algorithm", "heuristic", "optimisation", "optimization", "convergence",
                "search space", "fitness", "population", "swarm"],
    "ml_dl":   ["neural network", "backpropagation", "gradient", "epoch", "training",
                "loss function", "activation", "weight", "bias", "convolutional"],
    "nlp":     ["token", "tokenization", "embedding", "vocabulary", "corpus",
                "language model", "sentiment", "syntax", "semantics"],
    "biology": ["cell", "organism", "gene", "protein", "DNA", "RNA", "evolution",
                "species", "enzyme", "membrane"],
    "physics": ["force", "energy", "momentum", "velocity", "quantum", "wave",
                "particle", "entropy", "thermodynamics"],
    "math":    ["theorem", "proof", "integral", "derivative", "matrix", "vector",
                "eigenvalue", "limit", "continuity", "series"],
    "chemistry": ["reaction", "molecule", "atom", "bond", "compound", "oxidation",
                  "catalyst", "equilibrium"],
    "economics": ["supply", "demand", "elasticity", "GDP", "inflation", "market",
                  "fiscal", "monetary"],
}

def detect_script_domain(script: str) -> str:
    sl = script.lower()
    scores: dict = defaultdict(int)
    for domain, signals in _DOMAIN_SIGNALS.items():
        for sig in signals:
            if sig.lower() in sl:
                scores[domain] += 1
    if not scores:
        return "general"
    top = max(scores, key=scores.get)
    return top if scores[top] >= 2 else "general"


# ── Concept / key-definition extraction ──────────────────────────────────────

def extract_key_definitions(script: str) -> list:
    """Extract (term, definition, sentence) triples from the script."""
    patterns = [
        r"([A-Z][a-zA-Z\s\-]{3,40})\s+(?:is|are|refers to|means|denotes)\s+([^.!?]{20,150}[.!?])",
        r"(?:called|known as|referred to as|defined as)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s\-]{3,50}?)(?:\.|,|\s+and\s|\s+or\s)",
    ]
    results = []
    for pat in patterns:
        for m in re.finditer(pat, script, re.IGNORECASE):
            term = m.group(1).strip()
            defn = m.group(0).strip()
            if 3 < len(term.split()) <= 8 and len(defn) > 20:
                results.append({"term": term, "definition": defn, "sentence": defn})
    return results[:20]


def generate_concept_questions(defs: list, all_defs: list, concepts: list, domain: str) -> list:
    questions = []
    for d in defs[:10]:
        term = d["term"]
        defn = d["definition"]
        # Simple template-based question for definitions
        q = {
            "question": f"What best describes '{term}'?",
            "options": {},
            "correct": "A",
            "correct_answer": defn[:120],
            "difficulty": DIF.MEDIUM,
            "passage": defn,
            "section": "Key Definition",
            "qtype": QTYPES.concept,
        }
        # Build 3 plausible-sounding distractors from other terms
        others = [x["term"] for x in all_defs if x["term"] != term][:3]
        while len(others) < 3:
            others.append("None of the above")
        opts = [defn[:120]] + [f"Refers to {o}" for o in others[:3]]
        random.shuffle(opts)
        correct_idx = opts.index(defn[:120])
        for i, letter in enumerate(LETTERS):
            q["options"][letter] = opts[i]
        q["correct"] = LETTERS[correct_idx]
        q["correct_answer"] = opts[correct_idx]
        questions.append(q)
    return questions


# ── Code block detection ──────────────────────────────────────────────────────

def detect_code_blocks(script: str) -> list:
    blocks = []
    for m in re.finditer(r"```(?:python|pseudocode|code)?\n(.*?)```", script, re.DOTALL | re.IGNORECASE):
        start = m.start()
        blocks.append({
            "code": m.group(1).strip(),
            "context_before": script[max(0, start - 200):start].strip()
        })
    # Indented blocks
    for block in re.findall(r"(?:^|\n)((?:[ \t]{4,}[^\n]+\n?){3,})", script):
        blocks.append({"code": block.strip(), "context_before": ""})
    return blocks[:5]


def generate_code_questions(code: str, context: str) -> list:
    """Generate simple template questions about code blocks."""
    questions = []
    # Variable / operation questions
    variables = re.findall(r"\b([a-zA-Z_]\w*)\s*=", code)
    for var in variables[:2]:
        q = {
            "question": f"In the code, what does the variable `{var}` represent?",
            "options": {
                "A": f"The value of {var} computed during the operation",
                "B": "A constant that never changes",
                "C": "A random placeholder",
                "D": "An error handler",
            },
            "correct": "A",
            "correct_answer": f"The value of {var} computed during the operation",
            "difficulty": DIF.MEDIUM,
            "passage": code[:300],
            "section": "Code Block",
            "qtype": QTYPES.code,
        }
        questions.append(q)
    return questions[:3]


# ── Comparison detection ──────────────────────────────────────────────────────

def detect_comparisons(script: str) -> list:
    sents = _sent_tokenize(script)
    comp_kws = ["compared to", "unlike", "whereas", "while", "in contrast",
                "better than", "worse than", "faster than", "slower than",
                "vs ", "versus"]
    return [s for s in sents if any(kw in s.lower() for kw in comp_kws)][:8]


def make_comparison_question(sent: str, script: str, concepts: list, domain: str) -> Optional[dict]:
    if len(sent.split()) < 10:
        return None
    return {
        "question": f"Based on the following, which statement is correct? '{sent[:120]}'?",
        "options": {
            "A": sent[:120],
            "B": "The opposite of what is stated above",
            "C": "Both approaches are identical in performance",
            "D": "Neither approach has been studied",
        },
        "correct": "A",
        "correct_answer": sent[:120],
        "difficulty": DIF.MEDIUM,
        "passage": sent,
        "section": "Comparison",
        "qtype": QTYPES.comparison,
    }


# ── Statistics detection ──────────────────────────────────────────────────────

def detect_statistics(script: str) -> list:
    sents = _sent_tokenize(script)
    stat_pats = [r"\d+%", r"\d+\.\d+", r"\d{4}", r"billion|million|thousand"]
    return [s for s in sents if any(re.search(p, s) for p in stat_pats)][:8]


def generate_stat_questions(sents: list, script: str, concepts: list, domain: str) -> list:
    questions = []
    for sent in sents[:5]:
        nums = re.findall(r"\b\d[\d.,]+\b", sent)
        if not nums:
            continue
        target_num = nums[0]
        distractors = []
        for n in nums[1:]:
            distractors.append(n)
        while len(distractors) < 3:
            try:
                v = float(target_num.replace(",", ""))
                distractors.append(str(round(v * random.choice([0.5, 2.0, 0.1]), 2)))
            except Exception:
                distractors.append("N/A")
        opts = [target_num] + distractors[:3]
        random.shuffle(opts)
        ci = opts.index(target_num)
        q = {
            "question": f"What numerical value is mentioned in: \"{sent[:100]}\"?",
            "options": {LETTERS[i]: opts[i] for i in range(4)},
            "correct": LETTERS[ci],
            "correct_answer": target_num,
            "difficulty": DIF.EASY,
            "passage": sent,
            "section": "Statistics",
            "qtype": QTYPES.statistic,
        }
        questions.append(q)
    return questions


# ── Option post-processing ────────────────────────────────────────────────────

def _fix_model_options(parsed: dict, chunk: str, domain: str, script_concepts: list) -> dict:
    """Replace empty / bad options with script-derived distractors."""
    correct_letter = parsed.get("correct")
    correct_text   = parsed["options"].get(correct_letter, "") if correct_letter else ""

    # Gather replacement candidates from chunk
    candidates = re.findall(r"\b[A-Z][a-z]{3,}\b", chunk)
    candidates = list(dict.fromkeys(candidates))  # deduplicate, preserve order
    random.shuffle(candidates)
    cand_iter = iter(candidates)

    for letter in LETTERS:
        opt = parsed["options"].get(letter, "")
        if not _is_valid_option(opt):
            replacement = next(cand_iter, None) or "None of the above"
            parsed["options"][letter] = replacement

    return parsed


# ── Main generation function ──────────────────────────────────────────────────

def generate_mcqs_from_script(
    script_text: str,
    tokenizer,
    model,
    device,
    num_questions: Optional[int] = None,
    chunk_words: int = 400,
    temperature: float = 0.5,
    num_beams: int = 8,
    num_candidates: int = 3,
    dedup_threshold: float = 0.68,
    exclude_questions: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Hybrid MCQ pipeline — any academic field.

    Parameters
    ----------
    script_text        : The full generated video script.
    tokenizer / model  : Fine-tuned FLAN-T5 instances from ModelRegistry.
    device             : torch.device
    num_questions      : None = maximum coverage (all chunks).
    exclude_questions  : Questions already used in K-type slides (will be filtered out).
    dedup_threshold    : Jaccard similarity above which two questions are considered duplicates.

    Returns
    -------
    List of question dicts, each with keys:
        question, options, correct, correct_answer, difficulty, passage, section, qtype
    """
    if model is None or tokenizer is None:
        logger.error("Quiz model not loaded")
        return []

    model.eval()
    all_questions: List[dict] = []
    seen_fps: set = set()

    # Build exclude set from K-slide questions
    exclude_fps: set = set()
    if exclude_questions:
        for q in exclude_questions:
            exclude_fps.add(_fingerprint(q.get("question", "")))

    def _add(q_dict: dict):
        if not q_dict:
            return
        q_text = q_dict.get("question", "")
        if not q_text or len(q_text) < 15:
            return
        fp = _fingerprint(q_text)
        if fp in seen_fps or fp in exclude_fps:
            return
        seen_fps.add(fp)
        all_questions.append(q_dict)

    logger.info("STEP 0: Indexing script — concepts + domain...")
    domain   = detect_script_domain(script_text)
    concepts = _extract_concepts(script_text)

    # ── Concept / definition questions ────────────────────────
    all_defs = extract_key_definitions(script_text)
    logger.info(f"  {len(all_defs)} definitions found")
    if all_defs:
        for q in generate_concept_questions(all_defs, all_defs, concepts, domain):
            _add(q)

    # ── Code questions ─────────────────────────────────────────
    code_blocks = detect_code_blocks(script_text)
    for cb in code_blocks:
        for q in generate_code_questions(cb["code"], cb["context_before"]):
            _add(q)

    # ── Comparison questions ───────────────────────────────────
    for sent in detect_comparisons(script_text)[:5]:
        cq = make_comparison_question(sent, script_text, concepts, domain)
        if cq:
            _add(cq)

    # ── Statistics questions ───────────────────────────────────
    for q in generate_stat_questions(detect_statistics(script_text), script_text, concepts, domain):
        _add(q)

    # ── Model-based prose questions ────────────────────────────
    logger.info("STEP 1: Model-based generation for prose chunks...")
    prose = re.sub(
        r"```(?:python|pseudocode|code)?\n.*?\n```",
        "[CODE BLOCK]", script_text, flags=re.DOTALL | re.IGNORECASE
    )
    chunks = chunk_script(prose, target_words=chunk_words)
    logger.info(f"  {len(chunks)} prose chunks")

    if num_questions:
        remaining = max(1, num_questions - len(all_questions))
        indices   = np.linspace(0, len(chunks) - 1, min(remaining, len(chunks)), dtype=int)
        chunks    = [chunks[i] for i in sorted(set(indices.tolist()))]

    for ci, chunk in enumerate(chunks):
        prompt = make_prompt(chunk)
        inputs = tokenizer(
            prompt, return_tensors="pt", max_length=MAX_INPUT, truncation=True
        ).to(device)

        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=MAX_TARGET,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=min(num_candidates, num_beams),
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        best, best_score = None, -9999
        for out in outs:
            decoded = tokenizer.decode(out, skip_special_tokens=True)
            parsed  = parse_mcq_output(decoded)
            valid_opts = [v for v in parsed["options"].values() if _is_valid_option(v)]
            score = len(valid_opts) * 4
            if parsed.get("correct"):                          score += 10
            if "?" in parsed.get("question", ""):             score += 4
            if len(parsed.get("question", "")) > 25:          score += 3
            if _is_surface_question(parsed.get("question", "")): score -= 12
            if _question_too_long(parsed.get("question", "")): score -= 15
            unique = {v.lower().strip() for v in parsed["options"].values() if v}
            if len(unique) <= 1: score -= 20
            if len(unique) == 2: score -= 8
            if score > best_score:
                best_score, best = score, parsed

        if best is None:
            continue

        correct_letter = best.get("correct")
        correct_text   = best["options"].get(correct_letter, "") if correct_letter else ""
        if not correct_letter or not _is_valid_option(correct_text):
            continue
        if _question_too_long(best.get("question", "")):
            continue

        best = _fix_model_options(best, chunk, domain, concepts)
        correct_letter = best.get("correct")
        correct_text   = best["options"].get(correct_letter, "") if correct_letter else ""
        if not correct_letter or not _is_valid_option(correct_text):
            continue

        diff_label = _classify_difficulty(best["question"], correct_text)
        _add({
            "question":       best["question"],
            "options":        best["options"],
            "correct":        best["correct"],
            "correct_answer": correct_text,
            "difficulty":     diff_label,
            "passage":        chunk[:400],
            "section":        f"Prose chunk {ci + 1}",
            "qtype":          QTYPES.recall,
        })
        logger.info(f"  ✅ Chunk {ci+1}/{len(chunks)}: {best['question'][:60]}…")

    # ── Deduplication ─────────────────────────────────────────
    logger.info("STEP 3: Cross-question deduplication...")
    final: List[dict] = []
    for candidate in all_questions:
        dup = any(
            _jaccard_similarity(candidate["question"], kept["question"]) >= dedup_threshold
            for kept in final
        )
        if not dup:
            final.append(candidate)

    logger.info(f"  {len(all_questions) - len(final)} duplicates removed → {len(final)} unique questions")

    diff_order = {DIF.EASY: 0, DIF.MEDIUM: 1, DIF.HARD: 2}
    final.sort(key=lambda x: diff_order.get(x.get("difficulty", DIF.MEDIUM), 1))

    return final


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_concepts(script: str) -> list:
    concepts: list = []
    concepts += re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
        r"(?:Algorithm|Method|Search|Model|System|Technique|Function|Rule|Theorem|Law|Principle)\b",
        script
    )
    concepts += re.findall(r"\b([A-Z]{2,6})\b", script)
    concepts += re.findall(r"[`'\"]([A-Za-z][A-Za-z_\s]{2,40})[`'\"]", script)
    seen, clean = set(), []
    for c in concepts:
        c = c.strip().strip("'\"` .,")
        cl = c.lower()
        if cl not in seen and 2 < len(c) < 80:
            seen.add(cl)
            clean.append(c)
    return clean