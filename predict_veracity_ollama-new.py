# Saved 9430 predictions to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\1-veracity-done.json
# Saved unmatched/fallback details to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\1-veracity-done-unmatched.json

# Saved 2826 predictions to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\2-veracity-done.json
# Saved unmatched/fallback details to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\2-veracity-done-unmatched.json

# Saved 3124 predictions to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-veracity-done.json
# Saved unmatched/fallback details to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-veracity-done-unmatched.json

# Saved 3124 predictions to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-veracity-done.json
# Saved unmatched/fallback details to: E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-veracity-done-unmatched.json

import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

RERANKED_PATH = Path(
    r"E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-processed-reranked.json"
)
DECOMPOSED_PATH = Path(
    r"E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-decomposed-claims.json"
)
OUTPUT_PATH = Path(
    r"E:\Projects\Numerical Fact Checking\CheckThat 2025\task3\data\1-my dataset\3-veracity-done.json"
)

ALLOWED_LABELS = {"True", "False", "Conflicting"}
BATCH_SIZE = 1
SAVE_EVERY = 1
REQUEST_TIMEOUT = 240
MATCH_THRESHOLD = 0.92
MAX_DOCS = 5


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "Ã¢â‚¬â„¢": "'",
        "Ã¢â‚¬Å“": '"',
        "Ã¢â‚¬Â": '"',
        "ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢": "'",
        "ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ": '"',
        "ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_for_similarity(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9%$.,:/ -]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_for_similarity(a), normalize_for_similarity(b)).ratio()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return default

    return json.loads(text)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def build_decomposed_index(
    decomposed_items: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id: dict[int, dict[str, Any]] = {}
    by_claim: dict[str, list[dict[str, Any]]] = {}

    for item in decomposed_items:
        original_id = item.get("original_id")
        original_claim = item.get("original_claim", "")

        if isinstance(original_id, int):
            by_id[original_id] = item

        claim_key = normalize_text(original_claim)
        by_claim.setdefault(claim_key, []).append(item)

    return by_id, by_claim


def find_best_claim_match(
    claim: str, decomposed_items: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    best_item = None
    best_score = -1.0

    for item in decomposed_items:
        score = similarity(claim, item.get("original_claim", ""))
        if score > best_score:
            best_item = item
            best_score = score

    return best_item, best_score


def match_record(
    reranked_item: dict[str, Any],
    decomposed_items: list[dict[str, Any]],
    decomposed_by_id: dict[int, dict[str, Any]],
    decomposed_by_claim: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    original_id = reranked_item.get("original_id")
    claim = reranked_item.get("claim", "")
    normalized_claim = normalize_text(claim)

    candidate = decomposed_by_id.get(original_id)
    if candidate and normalize_text(candidate.get("original_claim", "")) == normalized_claim:
        return candidate, "id+exact-claim"

    for item in decomposed_by_claim.get(normalized_claim, []):
        if item.get("original_id") == original_id:
            return item, "claim-list+id"

    if candidate:
        score = similarity(claim, candidate.get("original_claim", ""))
        if score >= MATCH_THRESHOLD:
            return candidate, f"id+similar-claim({score:.3f})"

    claim_matches = decomposed_by_claim.get(normalized_claim, [])
    if len(claim_matches) == 1:
        return claim_matches[0], "exact-claim-only"

    best_item, best_score = find_best_claim_match(claim, decomposed_items)
    if best_item and best_score >= MATCH_THRESHOLD:
        return best_item, f"global-similar-claim({best_score:.3f})"

    return None, "unmatched"


def build_single_prompt(record: dict[str, Any]) -> str:
    questions_block = "\n".join(
        f"{idx}. {question}" for idx, question in enumerate(record["decomposed_questions"][:5], start=1)
    )
    docs_block = "\n\n".join(
        f"Document {idx}:\n{doc.strip()}" for idx, doc in enumerate(record["docs"][:MAX_DOCS], start=1)
    )

    return f"""You are a precise numerical fact-checking assistant. Your job is to verify a claim against evidence documents and assign exactly one label: True, False, or Conflicting.

---

## LABEL DEFINITIONS

**True** - The evidence directly and clearly supports the claim. The key facts, numbers, dates, or quantities in the claim match what the documents state. Minor irrelevant differences are acceptable.

**False** - The evidence directly contradicts the claim. At least one key fact, number, date, percentage, or quantity in the claim is clearly wrong according to the documents.

**Conflicting** - Use ONLY when:
  - Different documents give contradictory information about the same fact, OR
  - The evidence is genuinely ambiguous and cannot lean toward True or False.
  - Do NOT use Conflicting just because documents are incomplete or don't cover every detail.

---

## DECISION PROCESS - follow these steps in order

**Step 1 - Identify the core claim.**
What is the single most important factual assertion? What specific number, date, percentage, or fact is being claimed?

**Step 2 - Extract all key numerical facts from the claim.**
List every important number, year, percentage, count, amount, rank, duration, or comparison in the claim. These are the first things you must verify.

**Step 3 - Check each decomposed question against the documents.**
For each question, note: Does any document answer it? Does the answer support or contradict the claim?

**Step 4 - Compare claim facts with evidence facts.**
For each key fact, decide whether the documents:
- support the exact value or fact,
- contradict it with a different value or fact, or
- do not mention it.

**Step 5 - Apply the label.**
- If at least one document clearly confirms the core claim -> **True**
- If at least one document clearly contradicts a key fact in the claim -> **False**
- If two documents give genuinely conflicting answers to the same fact -> **Conflicting**
- If one document contradicts the claim and the other documents are merely silent, prefer **False**, not **Conflicting**
- If documents are silent or only tangentially related, judge based on what IS present. Silence alone is not Conflicting.

**Step 6 - Check your answer.**
Ask yourself:
- Am I defaulting to Conflicting out of uncertainty?
- Did I see a mismatched number/date/quantity and fail to mark it False?
If so, re-read the documents and force a lean toward True or False if any evidence points that way.

---

## STRICT RULES

1. Numerical/date mismatches are decisive. A wrong year, percentage, count, or amount -> **False**, even if the topic is otherwise correct.
2. Do NOT use outside knowledge. Judge only from the provided documents.
3. Do NOT use Conflicting as a default for weak or partial evidence. Only use it for genuine evidence conflict.
4. A document that partially supports the claim still counts as evidence - weigh it.
5. If documents are truncated or low quality but still indicate the claim direction, use that signal.
6. For numerical fact checking, a single reliable contradiction on a key quantity is enough for **False**.
7. Silence, missing detail, or incomplete coverage is not a contradiction and not a reason by itself to use **Conflicting**.

---

## INPUT

**Original ID:** {record["original_id"]}
**Claim:** {record["claim"]}

**Decomposed Questions:**
{questions_block}

**Evidence Documents:**
{docs_block}

---

## OUTPUT

Think step by step silently, then output ONLY this JSON and nothing else:
{{"label": "True|False|Conflicting", "reason": "1-2 sentence explanation citing the specific fact or document that decided the label", "supports_count": 0, "contradicts_count": 0, "conflicts_across_docs": false}}
"""


def build_batch_prompt(records: list[dict[str, Any]]) -> str:
    sections: list[str] = []

    for batch_index, record in enumerate(records, start=1):
        questions_block = "\n".join(
            f"{idx}. {question}" for idx, question in enumerate(record["decomposed_questions"][:5], start=1)
        )
        docs_block = "\n\n".join(
            f"Document {idx}:\n{doc.strip()}" for idx, doc in enumerate(record["docs"][:MAX_DOCS], start=1)
        )
        sections.append(
            f"""Case {batch_index}
Original ID: {record["original_id"]}
Claim: {record["claim"]}

Decomposed Questions:
{questions_block}

Evidence Documents:
{docs_block}"""
        )

    cases_block = "\n\n" + ("\n\n" + ("=" * 80) + "\n\n").join(sections)

    return f"""You are a precise numerical fact-checking assistant. For each case, verify the claim against the provided evidence and assign exactly one label: True, False, or Conflicting.

Use this policy for every case:
1. Extract all key quantities and factual slots from the claim: numbers, dates, percentages, counts, money amounts, ranks, durations, and named entities.
2. Compare those facts against the evidence documents.
3. If any key fact is clearly contradicted by a document, prefer False.
4. Use Conflicting ONLY when documents contradict each other about the same fact.
5. If one document contradicts the claim and the others are merely silent or partial, label False.
6. Silence, incomplete evidence, or tangential evidence is not enough for Conflicting.
7. Numerical/date mismatches are decisive and should strongly favor False.

Cases:
{cases_block}

Output ONLY valid JSON as an array of objects in this exact schema:
[{{"original_id":123,"label":"True|False|Conflicting","reason":"short explanation","supports_count":0,"contradicts_count":0,"conflicts_across_docs":false}}]
"""


def call_ollama(prompt: str, timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 700,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_json_payload(text: str) -> Any:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Model output did not contain JSON: {text[:500]}")

    return json.loads(match.group(0))


def normalize_label(label: str) -> str:
    raw = normalize_text(label)
    mapping = {
        "true": "True",
        "false": "False",
        "conflicting": "Conflicting",
        "conflict": "Conflicting",
        "half true/false": "Conflicting",
        "half true false": "Conflicting",
        "mixed": "Conflicting",
        "partially true": "Conflicting",
        "partly true": "Conflicting",
        "mostly true": "True",
        "mostly false": "False",
        "contradicted": "False",
        "incorrect": "False",
    }
    normalized = mapping.get(raw)
    if normalized not in ALLOWED_LABELS:
        raise ValueError(f"Unexpected label from model: {label}")
    return normalized


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return normalize_text(value) in {"true", "yes", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def resolve_label(response_json: dict[str, Any]) -> str:
    label = normalize_label(str(response_json.get("label", "")))
    supports_count = coerce_int(response_json.get("supports_count", 0))
    contradicts_count = coerce_int(response_json.get("contradicts_count", 0))
    conflicts_across_docs = coerce_bool(response_json.get("conflicts_across_docs", False))

    # Contradiction without a genuine cross-document disagreement should lean False.
    if (
        label == "Conflicting"
        and contradicts_count > 0
        and supports_count == 0
        and not conflicts_across_docs
    ):
        return "False"

    return label


def predict_single_label(record: dict[str, Any]) -> str:
    print(f"Sending single request for ID {record['original_id']}...", flush=True)
    prompt = build_single_prompt(record)
    response_payload = call_ollama(prompt)
    response_text = response_payload.get("response", "")
    response_json = extract_json_payload(response_text)
    if not isinstance(response_json, dict):
        raise ValueError("Single-case output was not a JSON object.")
    return resolve_label(response_json)


def predict_batch_labels(records: list[dict[str, Any]]) -> dict[int, str]:
    record_ids = [record["original_id"] for record in records]
    print(f"Sending batch request for IDs {record_ids}...", flush=True)
    prompt = build_batch_prompt(records)
    response_payload = call_ollama(prompt)
    response_text = response_payload.get("response", "")
    response_json = extract_json_payload(response_text)

    if not isinstance(response_json, list):
        raise ValueError("Batch output was not a JSON array.")

    parsed: dict[int, str] = {}
    for item in response_json:
        if not isinstance(item, dict):
            continue
        original_id = item.get("original_id")
        if not isinstance(original_id, int):
            continue
        parsed[original_id] = resolve_label(item)

    return parsed


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def load_existing_predictions() -> tuple[list[dict[str, Any]], dict[int, str]]:
    existing = load_json(OUTPUT_PATH, default=[])
    if not isinstance(existing, list):
        return [], {}

    cleaned: list[dict[str, Any]] = []
    by_id: dict[int, str] = {}

    for item in existing:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        label = item.get("label")
        if isinstance(item_id, int) and isinstance(label, str):
            try:
                normalized = normalize_label(label)
            except ValueError:
                continue
            cleaned_item = {"id": item_id, "label": normalized}
            cleaned.append(cleaned_item)
            by_id[item_id] = normalized

    return cleaned, by_id


def main() -> None:
    reranked_items = load_json(RERANKED_PATH, default=[])
    decomposed_items = load_json(DECOMPOSED_PATH, default=[])

    if not isinstance(reranked_items, list) or not isinstance(decomposed_items, list):
        raise ValueError("Input files must both contain JSON arrays.")

    decomposed_by_id, decomposed_by_claim = build_decomposed_index(decomposed_items)
    predictions, existing_by_id = load_existing_predictions()

    pending_records: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    skipped_existing = 0

    for reranked_item in reranked_items:
        matched, match_mode = match_record(
            reranked_item, decomposed_items, decomposed_by_id, decomposed_by_claim
        )

        reranked_id = reranked_item.get("original_id")
        if isinstance(reranked_id, int) and reranked_id in existing_by_id:
            skipped_existing += 1
            continue

        if not matched:
            original_id = reranked_item.get("original_id")
            if isinstance(original_id, int):
                predictions.append({"id": original_id, "label": "Conflicting"})
                existing_by_id[original_id] = "Conflicting"
            unmatched.append(
                {
                    "original_id": reranked_item.get("original_id"),
                    "claim": reranked_item.get("claim", ""),
                    "match_mode": match_mode,
                }
            )
            continue

        original_id = int(matched["original_id"])
        if original_id in existing_by_id:
            skipped_existing += 1
            continue

        claim = matched.get("original_claim", reranked_item.get("claim", ""))
        decomposed_questions = matched.get("decomposed_questions", [])
        docs = reranked_item.get("docs", [])[:MAX_DOCS]

        if not decomposed_questions or not docs:
            predictions.append({"id": original_id, "label": "Conflicting"})
            existing_by_id[original_id] = "Conflicting"
            unmatched.append(
                {
                    "original_id": original_id,
                    "claim": claim,
                    "match_mode": f"{match_mode}+missing-fields",
                }
            )
            continue

        pending_records.append(
            {
                "original_id": original_id,
                "claim": claim,
                "decomposed_questions": decomposed_questions,
                "docs": docs,
                "match_mode": match_mode,
            }
        )

    print(f"Loaded existing predictions: {len(existing_by_id)}")
    print(f"Skipped already predicted IDs: {skipped_existing}")
    print(f"Pending records to predict: {len(pending_records)}")

    batches = chunked(pending_records, BATCH_SIZE)
    batch_iter = (
        tqdm(batches, desc="Predicting batches", unit="batch")
        if tqdm is not None
        else batches
    )

    for batch_index, batch in enumerate(batch_iter, start=1):
        try:
            batch_ids = [record["original_id"] for record in batch]
            print(
                f"[batch {batch_index}/{len(batches)}] starting batch for IDs {batch_ids}",
                flush=True,
            )
            batch_predictions = predict_batch_labels(batch)

            for record in batch:
                label = batch_predictions.get(record["original_id"])
                if not label:
                    label = predict_single_label(record)
                predictions.append({"id": record["original_id"], "label": label})
                existing_by_id[record["original_id"]] = label
                print(
                    f"[batch {batch_index}/{len(batches)}] "
                    f"{record['original_id']}: {label} via {record['match_mode']}"
                )

        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://localhost:11434. "
                "Make sure `ollama serve` is running and the model is available."
            ) from exc
        except Exception as batch_exc:
            print(
                f"[batch {batch_index}/{len(batches)}] "
                f"batch failed, retrying one by one: {batch_exc}",
                flush=True,
            )
            for record in batch:
                try:
                    label = predict_single_label(record)
                except Exception as single_exc:
                    print(
                        f"[batch {batch_index}/{len(batches)}] "
                        f"{record['original_id']}: fallback to Conflicting due to error: {single_exc}",
                        flush=True,
                    )
                    label = "Conflicting"
                predictions.append({"id": record["original_id"], "label": label})
                existing_by_id[record["original_id"]] = label

        if batch_index % SAVE_EVERY == 0:
            predictions = sorted(predictions, key=lambda item: item["id"])
            save_json(OUTPUT_PATH, predictions)
            print(
                f"[batch {batch_index}/{len(batches)}] checkpoint saved to {OUTPUT_PATH}",
                flush=True,
            )
            time.sleep(0.1)

    predictions = sorted({item["id"]: item for item in predictions}.values(), key=lambda item: item["id"])
    save_json(OUTPUT_PATH, predictions)
    print(f"\nSaved {len(predictions)} predictions to: {OUTPUT_PATH}")

    unmatched_path = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "-unmatched.json")
    save_json(unmatched_path, unmatched)
    print(f"Saved unmatched/fallback details to: {unmatched_path}")


if __name__ == "__main__":
    main()
