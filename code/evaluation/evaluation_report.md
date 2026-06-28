# Sample Evaluation Report

## Approach

The system is an image-first multimodal evidence reviewer for car, laptop, and
package claims. Claim conversations identify the alleged object, part, issue,
and material details. Images decide whether the claim is supported,
contradicted, or lacks enough information.

Gemini is the primary reviewer. OpenAI-compatible vision is an optional,
suspicious-only adjudicator or fallback. Both providers receive the same claim,
image IDs, evidence requirements, and structured user-history context, then
return strict JSON that passes through shared validation and deterministic
rules.

Prompt version: `evidence-review-v7`.

## Label-Guided Policy

`dataset/sample_claims.csv` was used to evaluate general behaviors, not to
hardcode answers. The resulting policy is:

- A single clear, relevant image can support a claim when remaining images are
  neutral, blurry, contextual, or otherwise non-material.
- A material identity conflict, such as clearly different vehicles, laptops,
  or packages that prevent linking evidence to the claim, can require
  `not_enough_information`.
- Clear wrong-object or wrong-part evidence can contradict a claim.
- Missing-content claims require clear opened-package and contents evidence.
- User history raises risk/manual-review context but is not visual proof.
- Embedded instructions are ignored as instructions and flagged; they do not
  determine the outcome.
- Stock-like or watermark/provenance concerns add risk when supported by the
  image, but do not automatically block support from a separate credible image.

## Sample Metrics

Rows evaluated: 20.

| Field | Gemini Only | Gemini + OpenAI | Delta |
|---|---:|---:|---:|
| `evidence_standard_met` | 16/20 | 20/20 | +4 |
| `risk_flags` | 12/20 | 14/20 | +2 |
| `issue_type` | 13/20 | 13/20 | +0 |
| `object_part` | 18/20 | 17/20 | -1 |
| `claim_status` | 15/20 | 18/20 | +3 |
| `supporting_image_ids` | 16/20 | 18/20 | +2 |
| `valid_image` | 19/20 | 19/20 | +0 |
| `severity` | 12/20 | 15/20 | +3 |

Row-level exact match across those scored structured fields:

- Gemini only: 8/20 (40.00%)
- Gemini + suspicious-only OpenAI: 8/20 (40.00%)

OpenAI improves evidence sufficiency, claim status, supporting image IDs, and
severity on this sample, while not improving row-level exact match. It is
therefore kept optional and limited to suspicious rows rather than used as a
default second call for every claim.

## Operational Design

- **Measured scope and calls:** the labeled sample contains 20 rows and 29
  submitted images; the final claims file contains 44 rows and 82 submitted
  images. A no-cache Gemini-only sample pass needs 20 primary reviews. The
  suspicious-only sample configuration needs 20 primary reviews plus 12
  adjudications. The persisted final adjudication metadata contains 42
  successful suspicious-row adjudications; a nominal no-cache final pass is
  therefore 44 primary reviews plus those targeted second passes.
- **Caching:** responses are cached under `.cache/vlm/` using provider, model,
  prompt, claim, user history, evidence requirements, and image content hashes.
  This avoids repeated calls during replays. The latest metric refresh reused
  cached responses and made no live model calls.
- **Reliability:** transient API failures use bounded retry/backoff. A
  deterministic lower-quality fallback is available when providers cannot
  return usable JSON and is not stored as a successful model response.
- **Image transport:** content-based image detection avoids trusting file
  extensions. OpenAI-compatible requests convert unsupported or uncertain
  inputs such as AVIF, HEIC, and WebP to JPEG when needed.
- **Cost-conscious use:** Gemini handles primary review. OpenAI is reserved for
  fallback and adjudication triggers such as failures, material ambiguity,
  suspicious provenance, and inconsistent evidence/status combinations. Calls
  are issued row by row with retry/backoff rather than aggressive parallel
  bursts, which keeps TPM/RPM behavior predictable. Provider token telemetry,
  billed cost, and uncached network latency were not captured in this
  repository, so no invented token, dollar, or runtime estimates are claimed.

## Risk Handling and Rules

The shared validator guarantees exact output schema/order, allowed values,
lowercase boolean strings, non-empty required cells, and supporting image IDs
that exist in the corresponding input row.

The deterministic rule layer preserves the image-first policy:

- It keeps `user_history_risk` tied to explicit structured history flags.
- It preserves `not_enough_information` for explicit material cross-image
  object conflicts without requiring every image to show damage.
- It makes package missing-content decisions conservative unless contents are
  directly inspectable.
- It prevents visible damage on an unrelated object from controlling the
  predicted issue taxonomy or severity for the claimed object.
- It treats instruction text as risk context, never as a command.

## Limitations and Failure Modes

- Some multi-image rows remain hard when identity, provenance, or object-side
  linkage cannot be determined visually.
- The required one-row schema cannot represent separate final statuses for
  each component of a genuinely multi-part claim.
- VLM outputs can disagree on taxonomy and authenticity. Rules reduce known
  failure modes but do not replace human review for high-risk rows.
- No external image provenance service, OCR verification service, or product
  identity database is used.

## Final Submission Status

The final `output.csv` has been validated against `dataset/claims.csv`:

- 44 prediction rows for 44 input rows
- exact required columns in exact order
- no blank required cells
- allowed values and lowercase boolean strings
- semicolon-formatted risk/supporting-ID fields
- supporting image IDs restricted to each row's submitted images
- clean pandas CSV read

Gemini is the final primary provider and OpenAI remains suspicious-only
fallback/adjudication. No secrets are included in the repository or
documentation.
