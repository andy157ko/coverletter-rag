"""
RAG pipeline: retrieval + prompt construction + generation.

Rubric:
- Sentence embeddings for retrieval
- Retrieval-augmented generation system
"""

from typing import List, Dict, Tuple
import re
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


def extract_title_and_company(job_text: str) -> Tuple[str, str]:
    """
    Heuristic extraction of job title and company from the job description.
    """
    lines = [l.strip() for l in job_text.splitlines() if l.strip()]

    title = ""
    company = ""

    skip_phrases = {"about the job", "introduction"}
    filtered = [l for l in lines if l.lower() not in skip_phrases]

    # 1) Title guess: first non-generic line
    if filtered:
        title = filtered[0]

    # 2) Company guess: look for IBM or other company-like patterns
    for line in lines[:6]:
        if re.search(r"\bIBM\b", line):
            company = "IBM"
            break
        if re.search(r"\b(inc\.?|corp\.?|llc|ltd\.?)\b", line, flags=re.IGNORECASE):
            company = line
            break

    return title, company


def extract_resume_bullets(resume_text: str, max_bullets: int = 8) -> List[str]:
    """
    Extract 'bullet-like' lines from the resume.
    """
    bullets: List[str] = []
    seen = set()
    lines = [l.rstrip() for l in resume_text.splitlines()]
    current_section = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        upper = stripped.upper()
        if upper in {"EXPERIENCE", "WORK EXPERIENCE", "PROJECTS", "LEADERSHIP", "ACTIVITIES"}:
            current_section = upper
            continue

        if "@" in stripped:
            continue
        if "linkedin.com" in stripped.lower():
            continue
        if re.search(r"\b\d{3}[-)\s]\d{3}[-\s]\d{4}\b", stripped):
            continue
        if "RELEVANT COURSEWORK" in upper:
            continue
        if "EDUCATION" in upper:
            continue

        if stripped.startswith(("-", "•")):
            candidate = stripped.lstrip("-•").strip()
        else:
            if len(stripped.split()) < 5:
                continue
            # Only treat as candidate if in a “good” section
            if current_section == "":
                continue
            candidate = stripped

        if candidate and candidate not in seen:
            seen.add(candidate)
            bullets.append(candidate)

        if len(bullets) >= max_bullets:
            break

    return bullets


def build_safe_opening(job_title: str, job_company: str) -> str:
    pieces: List[str] = []

    if job_title and job_company:
        pieces.append(
            f"I am excited to apply for the {job_title} position at {job_company}."
        )
    elif job_title:
        pieces.append(
            f"I am excited to apply for the {job_title} position."
        )
    else:
        pieces.append(
            "I am excited to submit my application for this opportunity."
        )

    pieces.append(
        "With the experiences and skills outlined in my attached resume, "
        "I am confident I can contribute meaningfully to your team."
    )

    return " ".join(pieces)


def clean_and_filter_sentences(text: str) -> str:
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    keep: List[str] = []

    for s in raw_sentences:
        s_strip = s.strip()
        if not s_strip:
            continue

        if re.search(r"\bI am currently\b", s_strip, flags=re.IGNORECASE):
            continue
        if re.search(r"\bI am a data scientist\b", s_strip, flags=re.IGNORECASE):
            continue
        if re.search(r"\bBachelor'?s degree\b", s_strip, flags=re.IGNORECASE):
            continue
        if re.search(r"\bMaster'?s degree\b", s_strip, flags=re.IGNORECASE):
            continue
        if re.search(r"\byears of experience\b", s_strip, flags=re.IGNORECASE):
            continue

        keep.append(s_strip)

    return " ".join(keep)


def build_rule_based_letter(
    job_title: str,
    job_company: str,
    resume_bullets: List[str],
) -> str:
    if job_title and job_company:
        opening = (
            f"I am excited to apply for the {job_title} position at {job_company}. "
            "With the experiences and skills outlined in my attached resume, "
            "I am confident I can contribute meaningfully to your team."
        )
    elif job_title:
        opening = (
            f"I am excited to apply for the {job_title} position. "
            "With the experiences and skills outlined in my attached resume, "
            "I am confident I can contribute meaningfully to your team."
        )
    else:
        opening = (
            "I am excited to submit my application for this opportunity. "
            "With the experiences and skills outlined in my attached resume, "
            "I am confident I can contribute meaningfully to your team."
        )

    body_parts: List[str] = []
    if resume_bullets:
        body_parts.append(
            "In my recent experiences, I have developed strengths that align well with this role, including:"
        )
        for b in resume_bullets[:3]:
            body_parts.append(f"- {b}")
        body = "\n".join(body_parts)
    else:
        body = (
            "My background includes academic projects and professional experiences that have helped me "
            "build strong analytical, communication, and collaboration skills."
        )

    if job_company:
        closing = (
            f"I am enthusiastic about the opportunity to contribute to {job_company} "
            "and to learn from a team that values growth and innovation. "
            "Thank you for considering my application; I would welcome the chance to discuss how my background "
            "can support your goals."
        )
    else:
        closing = (
            "I am enthusiastic about the opportunity to contribute to your team and to learn in a dynamic environment. "
            "Thank you for considering my application; I would welcome the chance to discuss how my background "
            "can support your goals."
        )

    return f"{opening}\n\n{body}\n\n{closing}"


# -------------------------------------------------------------------
# RAG Pipeline
# -------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, generator, tokenizer_name: str, embed_model_name: str, index, metadata_store):
        self.generator = generator
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.embed_model = SentenceTransformer(embed_model_name)
        self.index = index  # FAISS or similar
        self.metadata_store = metadata_store  # list[dict], one per training example

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_model.encode([text])[0]

    def retrieve(self, resume_text: str, job_text: str, k: int = 5) -> List[Dict]:
        """
        Classic RAG-style retrieval over the *current* resume + job description.

        We build a small in-memory corpus of chunks:
        - Resume bullets (from extract_resume_bullets)
        - Sentences from the job description

        Then we embed all chunks and do cosine-similarity search against
        a query embedding (job_text + resume_text).
        """

        # 1) Build candidate chunks
        candidates: List[Dict] = []

        # (a) Resume bullets
        resume_bullets = extract_resume_bullets(resume_text)
        for b in resume_bullets:
            candidates.append(
                {
                    "source": "resume",
                    "text": b,
                }
            )

        # (b) Job description sentences
        job_sentences = re.split(r"(?<=[.!?])\s+", job_text.strip())
        for s in job_sentences:
            s_strip = s.strip()
            if len(s_strip.split()) < 5:
                continue  # skip very short/noisy bits
            candidates.append(
                {
                    "source": "job",
                    "text": s_strip,
                }
            )

        if not candidates:
            print("===== RETRIEVAL DEBUG =====")
            print("No candidate chunks found from resume/job; returning empty retrieval.")
            return []

        texts = [c["text"] for c in candidates]

        # 2) Embed all candidate chunks
        cand_embs = self.embed_model.encode(texts, convert_to_numpy=True).astype("float32")

        # 3) Embed the query (resume + job together)
        query = resume_text + "\n\n" + job_text
        q_vec = self.embed_query(query).astype("float32")

        # 4) Cosine similarity
        #    cos_sim = (A·B) / (||A|| * ||B||)
        cand_norms = np.linalg.norm(cand_embs, axis=1) + 1e-8
        q_norm = np.linalg.norm(q_vec) + 1e-8
        sims = (cand_embs @ q_vec) / (cand_norms * q_norm)

        # 5) Take top-k
        k = min(k, len(candidates))
        top_idx = np.argsort(-sims)[:k]

        print("\n===== RETRIEVAL DEBUG (classic RAG) =====")
        print("Num candidates:", len(candidates))
        print("Top-k indices:", top_idx)
        print("Top-k sims:", sims[top_idx])

        print("\n--- Retrieved Chunk Previews (up to k) ---")
        for rank, idx in enumerate(top_idx):
            c = candidates[idx]
            print(f"\n[Rank {rank} | source={c['source']}]")
            print(c["text"][:300].replace("\n", " "))
        print("================================\n")

        # Return the selected candidate dicts
        retrieved = [candidates[i] for i in top_idx]
        return retrieved

    def build_prompt(self, resume_text: str, job_text: str, retrieved_examples: List[Dict]) -> str:
        """
        Build prompt that:
        - Highlights TARGET JOB title/company
        - Shows key bullets from TARGET RESUME
        - Shows RAG-retrieved job qualifications + resume experiences
        - Asks the model to connect them in full paragraphs
        """

        job_title, job_company = extract_title_and_company(job_text)
        resume_bullets = extract_resume_bullets(resume_text)

        # --- 1) Format bullets from the user's resume ---
        if resume_bullets:
            bullets_text = "\n".join(f"- {b}" for b in resume_bullets)
        else:
            bullets_text = (
                "No explicit bullet points detected; instead, carefully use specific "
                "projects, roles, and achievements from the TARGET RESUME text."
            )

        # --- 2) Split retrieved chunks into job vs resume buckets ---
        job_chunks = [ex["text"] for ex in retrieved_examples if ex.get("source") == "job"]
        resume_chunks = [ex["text"] for ex in retrieved_examples if ex.get("source") == "resume"]

        # Keep only a few most relevant from each side for clarity
        job_chunks = job_chunks[:3]
        resume_chunks = resume_chunks[:4]

        if job_chunks:
            job_focus_text = "\n".join(f"- {c}" for c in job_chunks)
            job_focus_block = (
                "[KEY JOB QUALIFICATIONS]\n"
                "These lines summarize the most important qualifications and responsibilities for the role:\n"
                f"{job_focus_text}\n\n"
            )
        else:
            job_focus_block = ""

        if resume_chunks:
            resume_focus_text = "\n".join(f"- {c}" for c in resume_chunks)
            resume_focus_block = (
                "[MOST RELEVANT EXPERIENCES]\n"
                "These lines summarize the most relevant experiences from the candidate's resume:\n"
                f"{resume_focus_text}\n\n"
            )
        else:
            resume_focus_block = ""

        prompt = (
            "You are an expert career coach helping students apply to jobs and internships.\n"
            "Your task is to write a NEW, personalized, professional cover letter based ONLY on:\n"
            "- The TARGET JOB section\n"
            "- The TARGET RESUME section\n\n"
            "IMPORTANT RULES:\n"
            "1. Completely ignore any previous jobs, titles, or companies you may have seen before.\n"
            "2. The job title and company in the cover letter MUST match the TARGET JOB section.\n"
            "   Use these exact strings where appropriate if available:\n"
            f"   - Job title guess: '{job_title}'\n"
            f"   - Company guess: '{job_company}'\n"
            "3. You MUST NOT invent degrees, companies, or experience that are not in the TARGET RESUME.\n"
            "4. Do NOT claim specific years of experience unless they are explicitly stated.\n"
            "5. VERY IMPORTANT: Do NOT mention tools, programming languages, or software that do not appear "
            "in the TARGET RESUME bullet list below.\n\n"
            "Style:\n"
            "- Professional but not overly formal.\n"
            "- Clear, concise, and specific.\n"
            "- 3–5 paragraphs.\n"
            "- Roughly one page.\n\n"
            "How to use the information:\n"
            "- Treat [KEY JOB QUALIFICATIONS] as what the company is looking for.\n"
            "- Treat [MOST RELEVANT EXPERIENCES] and the TARGET RESUME bullets as evidence that the candidate fits.\n"
            "- Write in full sentences and paragraphs, not bullets.\n"
            "- Paraphrase the resume bullets instead of copying them verbatim.\n"
            "- For at least two of the key job qualifications, explicitly describe a matching experience from the resume.\n\n"
            "Below are key experiences from the TARGET RESUME that you MUST explicitly reference. "
            "Paraphrase at least three of them in your own words, and do not invent new tools or experiences.\n"
            f"{bullets_text}\n\n"
            # RAG context blocks:
            f"{job_focus_block}"
            f"{resume_focus_block}"
            "Now write a NEW cover letter for the following:\n\n"
            "[TARGET JOB]\n"
            f"{job_text}\n\n"
            "[TARGET RESUME]\n"
            f"{resume_text}\n\n"
            "Write ONLY the cover letter text, as 3–5 paragraphs. Do not use bullet points. "
            "In the first paragraph, mention the exact job title and company from the TARGET JOB if they can be inferred.\n"
        )

        return prompt


    def generate(self, resume_text: str, job_text: str, k: int = 3, max_new_tokens: int = 512) -> str:
        """
        Full RAG + generation + light post-processing, with a
        rule-based fallback if the model output is low quality.
        """
        resume_bullets = extract_resume_bullets(resume_text)
        job_title, job_company = extract_title_and_company(job_text)

        retrieved = self.retrieve(resume_text, job_text, k=k)
        print(f"Retrieved {len(retrieved)} examples for RAG.")

        prompt = self.build_prompt(resume_text, job_text, retrieved)

        print("\n===== PROMPT DEBUG (first 600 chars) =====")
        print(prompt[:600])
        print("==========================================\n")

        device = next(self.generator.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=120,    # NEW: encourage >= ~120 tokens of output
                do_sample=False,       # deterministic
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3 # reduce copy-paste of same phrase
            )

        raw_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("==== DEBUG raw_text (first 500 chars) ====")
        print(raw_text[:500])
        print("==========================================")

        raw_stripped = raw_text.strip()
        word_count = len(raw_stripped.split())
        too_short = word_count < 40
        looks_like_header = raw_stripped.lower().startswith("hiring company:")
        mentions_training_company = bool(
            re.search(r"\b(XYZ Analytics|Innovation Inc)\b", raw_stripped)
        )

        print(f"DEBUG: Generated word count = {word_count}")
        print(
            f"DEBUG flags: too_short={too_short}, "
            f"looks_like_header={looks_like_header}, "
            f"mentions_training_company={mentions_training_company}"
        )
        if too_short or looks_like_header or mentions_training_company:
            print("DEBUG: Using rule-based fallback letter.")
            return build_rule_based_letter(job_title, job_company, resume_bullets)

        # ---- Otherwise, lightly clean and prepend safe opening ----
        # 1) Drop obviously risky / untrue sentences
        text = clean_and_filter_sentences(raw_text)

        # If everything got filtered out, fall back
        if not text.strip():
            print("DEBUG: All generated sentences filtered; using rule-based fallback.")
            return build_rule_based_letter(job_title, job_company, resume_bullets)

        # 2) Fix known wrong titles/companies from training artifacts
        if job_title:
            text = re.sub(r"\bData Scientist\b", job_title, text)
            text = re.sub(r"\bSenior Support Engineer\b", job_title, text)
            # Also fix this hallucinated title:
            text = re.sub(r"\bSenior Project Manager\b", job_title, text)

        if job_company:
            text = re.sub(r"\bXYZ Analytics\b", job_company, text)
            text = re.sub(r"\bInnovation Inc\b", job_company, text)

        # 3) Optionally remove generic “I am writing to express my interest...”
        #    if it might refer to the wrong role/company
        text = re.sub(
            r"I am writing to express my interest in the [^.]+?\.",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # 4) Clean prompt scaffolding and tags
        text = text.replace("TARGET JOB section", "role")
        text = text.replace("TARGET RESUME", "my background")
        text = re.sub(
            r"^Hiring Company:.*\n?", "",
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )
        text = text.replace("[RESUME]", "").replace("[JOB]", "")
        text = text.strip()

        # Remove an initial "I am a/an ..." sentence if it's redundant with the opening
        first_sentence_match = re.match(r"^I am (an|a) [^.]+?\.", text)
        if first_sentence_match:
            text = text[first_sentence_match.end():].lstrip()

        # IMPORTANT: leave this commented out; it was mangling things
        # text = text.replace(" - ", ". ")

        # Drop any dangling "(" at the very end
        text = re.sub(r"\(\s*$", "", text).strip()

        # 5) Build safe opening paragraph grounded in the job info
        safe_opening = build_safe_opening(job_title, job_company)

        if text:
            final = safe_opening + " " + text
        else:
            final = safe_opening

        return final.strip()
