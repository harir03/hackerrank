# Multi-Domain Support Triage Agent

## Setup
```bash
cd code/
pip install -r requirements.txt
cp ../.env.example ../.env
# Add GEMINI_API_KEY_1 through GEMINI_API_KEY_5 to .env
```

## Run
```bash
python code/main.py
```

Reads:  `support_tickets/support_tickets.csv`
Writes: `support_tickets/output.csv`

## Architecture
5-layer pipeline: ingest → classify → retrieve → generate → validate.
Each ticket passes through safety checks before any API call is made.
Answers are grounded in the provided `data/` corpus only — the generator
is instructed to output CORPUS_GAP if the answer is not in the docs,
which triggers escalation instead of hallucination.

## Layers

**Layer 1 — Ingest:** PII scrubbing, injection detection, junk
filtering, language detection, multi-issue flagging. No API calls.

**Layer 2 — Classify:** One Gemini API call, temperature=0, structured
JSON output. Returns domain, request_type, product_area, severity,
escalate flag, confidence score, search_query.

**Layer 3 — Retrieve:** BM25 over 8088 chunks from `data/**/*.md`.
Built once at startup. Returns top-5 chunks + coverage score.

**Layer 4 — Generate:** One Gemini API call with retrieved chunks as
the only knowledge source. CORPUS_GAP signal escalates if corpus
does not cover the query.

**Layer 5 — Validate:** status and request_type validated against
allowed enums before any row touches disk. Hard fail on violation.

## Escalation gates (in priority order)
1. Prompt injection detected in ticket body
2. Fraud, stolen card, account compromise, legal threat in content
3. Classifier confidence < 0.72
4. Company field contradicts issue content (cross-domain mismatch)
5. Corpus coverage score < 0.25
6. Generator returns CORPUS_GAP

## Design decisions
- **CORPUS_GAP**: machine-parseable signal — never hallucinate,
  always escalate when corpus doesn't cover the query
- **Feature requests** skip retrieval entirely — templated response
  to avoid hallucinating non-existent features
- **BM25 per domain** — Visa queries never retrieve HackerRank chunks
- **Two API calls max per ticket** — classifier + generator
- **Junk/injection/language** rejected in Layer 1 before any API call
- **API key rotation** — round-robins across 5 keys to stay within
  free tier rate limits (20 RPD per key × 5 = 100 calls)

## Dependencies
```
google-genai, rank-bm25, langdetect, pandas, python-dotenv
```