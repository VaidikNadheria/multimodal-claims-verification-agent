# Multimodal Evidence Review

This Python system reviews car, laptop, and package claims from a claim
conversation, submitted images, user history, and evidence requirements. It
uses Gemini as the primary vision model and OpenAI-compatible vision only as a
fallback or second-pass adjudicator for suspicious rows.

The terminal entry point is `code/main.py`. The final prediction file is
written to `output.csv` at the repository root.

## Project Overview

The pipeline combines strict schema validation, local image handling,
provider-agnostic JSON clients, optional adjudication, and a narrow rule layer
for known evidence-sufficiency and taxonomy edge cases. It is designed to be
reproducible, cache-aware, and easy to inspect during evaluation.

## Setup

From the repository root, install the runtime dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set model credentials through environment variables only:

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."  # Needed only for fallback/adjudication.
```

Optional model overrides:

- `GEMINI_MODEL` (the final run used `gemini-2.5-flash`)
- `OPENAI_MODEL` (the final run used `gpt-5-mini`)

Never place credentials in source code, CSVs, documentation, or git.

## Final Prediction Run

Run the final claims pass from the repository root:

```bash
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" \
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}" \
python3 code/main.py \
  --input dataset/claims.csv \
  --output output.csv \
  --primary-provider gemini \
  --fallback-provider openai \
  --adjudicate-suspicious \
  --resume
```

`--resume` reuses compatible existing output rows and response cache entries.
Rows with failed or stale adjudication metadata are replayed so local policy
updates can be applied without unnecessarily repeating model calls.

The current `output.csv` has been validated with 44 rows, the exact required
column order, no blank required cells, allowed values only, lowercase boolean
strings, and row-valid supporting image IDs.

## Sample Evaluation

Gemini-only diagnostic pass:

```bash
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" \
python3 code/main.py \
  --input dataset/sample_claims.csv \
  --output code/evaluation/sample_predictions_gemini_only.csv \
  --primary-provider gemini \
  --fallback-provider none \
  --no-adjudication
```

Gemini plus suspicious-only OpenAI adjudication:

```bash
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}" \
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-mini}" \
python3 code/main.py \
  --input dataset/sample_claims.csv \
  --output code/evaluation/sample_predictions_openai_adjudicated.csv \
  --primary-provider gemini \
  --fallback-provider openai \
  --adjudicate-suspicious
```

Measured results and limitations are documented in
[`code/evaluation/evaluation_report.md`](./evaluation/evaluation_report.md).

## Architecture

- **Input and schema:** CSV input/output helpers enforce the contract and
  validate allowed values, field order, booleans, and image IDs.
- **Image handling:** image content is detected from bytes rather than filename
  extension. OpenAI-compatible requests normalize unsupported or uncertain
  formats, including AVIF, HEIC, and WebP, to high-quality JPEG when needed.
- **Primary review:** Gemini receives an image-first, strict-JSON prompt using
  `evidence-review-v7`.
- **Adjudication:** OpenAI is called only for suspicious, inconsistent, or
  failed primary outcomes. Its result goes through the same validator and rule
  layer as Gemini.
- **Evidence policy:** one clear relevant image can support a claim when other
  images are neutral context. Strong material object-identity conflicts can
  force `not_enough_information`.
- **Rules:** deterministic post-processing handles missing package contents,
  cross-image material conflicts, wrong-object taxonomy/severity, and
  structured user-history flags.
- **Caching and reliability:** provider/model/prompt/claim/history/evidence
  metadata plus image hashes determine cache keys. Transient API errors use
  retry/backoff. Deterministic fallback is logged as lower quality and is not
  cached as a successful model response.

## Evidence and Risk Handling

`evidence_requirements.csv` supplies object, part, and issue guidance to both
models. Images are the source of truth; conversations identify what to inspect.

`user_history.csv` is joined by `user_id`, included in both provider prompts
and cache keys, and used only for `user_history_risk` and
`manual_review_required` context. It never serves as visual proof or overrides
clear image evidence.

Instruction text in the claim or image is ignored as an instruction and may
add `text_instruction_present`. Stock, watermark, and provenance concerns add
risk when visually supported, but do not automatically prevent a supported
decision. If the image clearly shows a different object class, the rule layer
keeps taxonomy tied to the claimed object and sets severity to `unknown` rather
than describing damage on the unrelated object.

## Limitations

- Vision-model interpretation can still be uncertain for ambiguous angles,
  provenance, severe blur, and multi-part conversations compressed into one
  required output row.
- The system does not perform external provenance lookup or image forensics.
- `not_enough_information` is intentionally preferred when the evidence cannot
  be reliably linked to the claimed object or content.
- OpenAI adjudication improves several sample fields but does not improve the
  measured sample row-level exact score; it remains optional and
  suspicious-only by design.
