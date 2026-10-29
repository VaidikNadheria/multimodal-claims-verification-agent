# Multimodal Claims Verification Agent

## Project at a glance

- **Context:** HackerRank Orchestrate, June 2026; the repository records a final
  placement of **20th out of 1,773 participants** (approximately the top 1.1%).
- **Problem:** Determine whether images support, contradict, or provide
  insufficient information for car, laptop, and package damage/content claims.
- **Implementation:** A custom Python CLI that combines claim conversations,
  local images, minimum evidence requirements, and structured user history.
- **Model strategy:** Gemini primary review; OpenAI fallback and optional
  suspicious-row adjudication; shared deterministic safeguards and validation.
- **Documented workload:** 44 final claims with 82 submitted images.
- **Development evaluation:** 20 labeled claims with 29 images; claim-status
  agreement improved from 15/20 to 18/20 with targeted adjudication. Strict
  row-exact match stayed at 8/20 for both configurations.

## Resume-ready entry

**Multimodal Claims Verification Agent | Python, Gemini API, OpenAI API,
Pillow, pandas | June 2026**

- Ranked **20th of 1,773 participants** in HackerRank Orchestrate, building a
  multimodal pipeline to review image-backed car, laptop, and package claims.
- Integrated Gemini primary review with OpenAI fallback and targeted
  adjudication, improving claim-status agreement from **75% to 90%** on a
  **20-claim labeled development sample**.
- Implemented SHA-256 response caching, resumable CSV processing, retry/backoff,
  image-format normalization, and strict **14-column output validation** for
  a documented submission workload of **44 claims and 82 images**.

For a two-bullet resume, keep the first bullet and combine the model strategy
with the reliability features from the third. For an achievement section, use
the competition placement separately and spend the project bullets on design
and implementation.

## Longer project description

Built an image-first evidence-review system that processes a claim conversation
and its submitted photographs alongside object-specific evidence requirements
and claimant history. The pipeline produces one structured CSV row per claim,
including evidence sufficiency, risk flags, issue type, object part, claim
status, explanation, supporting image IDs, image validity, and severity.

A structured JSON prompt supplies policy, allowed output values, and relevant
context to Gemini. Optional OpenAI adjudication reviews suspicious or internally
inconsistent results using the images and preliminary prediction. Generic
deterministic rules handle material object-identity conflicts, unclear package
contents, wrong-object taxonomy, limited taxonomy calibration, and history flags.
History affects risk context rather than overriding visual evidence.

The engineering layer hashes image contents for cache identity, normalizes
unsupported image formats, retries transient API failures, reuses compatible
completed rows, and validates output fields and supporting image references.
Adjudication metadata records second-pass outcomes for inspection. Credentials
are supplied through environment variables.

## Technologies and techniques

| Category | Actual implementation |
|---|---|
| Language and runtime | Python 3.10+, dataclasses, type hints, argparse, logging, pathlib |
| AI services | Google Gemini API; OpenAI Chat Completions API with vision inputs |
| Documented run models | `gemini-2.5-flash` and `gpt-5-mini`, selected through environment variables |
| Image processing | Pillow; MIME detection from bytes; EXIF orientation; resize/JPEG conversion; base64 |
| Data processing | pandas with standard-library CSV fallback; JSON and JSONL |
| Networking | Direct REST/HTTPS requests through `urllib`; certifi CA bundle; bounded retry/backoff |
| Reliability | SHA-256 caching, resume support, conservative fallback, schema and enum validation |
| AI engineering | Structured prompting, conditional model routing, evidence attribution, prompt-injection precautions, deterministic post-processing |
| Tooling | Git and GitHub |

## Metrics and claim boundaries

The development report compares two configurations on the provided 20-row
labeled sample. With adjudication, claim status matched labels on 18/20 rows
(90%), evidence sufficiency on 20/20, risk flags on 14/20, and severity on 15/20.
Row-exact agreement across all eight scored structured fields was 8/20 (40%)
for both configurations. A field-level improvement does not mean every field
in a row was correct.

These are historical development results recorded in the repository, not a new
benchmark run or hidden-test accuracy claim. The public evaluation entry point
is a dry-run scaffold rather than a scoring utility, and the private labels and
generated predictions are excluded. The public repository therefore does not
independently reproduce the reported scores.

Do not claim model training/fine-tuning, LangChain/LangGraph, RAG, a vector
database, cloud deployment, a web application, measured cost or latency savings,
or production-scale accuracy: those are not demonstrated by this repository.
The competition placement is documented in the repository README; retain your
official result evidence when using it as an achievement.

## Interview explanation

“I built a Python pipeline that checks damage claims against submitted images.
Gemini performs the primary review, and OpenAI provides fallback or a second
pass when policy triggers identify suspicious results. I focused on reliable
integration: content-aware image preparation, caching and resume, deterministic
evidence safeguards, and a strict output contract. Targeted adjudication
improved claim-status agreement from 75% to 90% on the 20-row development
sample, while the full-row score stayed unchanged. The project placed 20th out
of 1,773 participants in HackerRank Orchestrate.”
