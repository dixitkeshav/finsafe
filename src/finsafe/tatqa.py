"""Reproducible, numeric-only TAT-QA adaptation; no synthetic answers."""
import hashlib
import json
import math
import random
import urllib.request
from collections import Counter
from pathlib import Path
from .data import canonical, prepare, validate, write_jsonl

REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
SOURCE = "https://github.com/NExTplusplus/TAT-QA"
LICENSE_URL = "https://nextplusplus.github.io/TAT-QA/"


def convert_context(context, split):
    table = context["table"]["table"]
    table_text = "\n".join(" | ".join(str(cell) for cell in row) for row in table)
    paragraphs = "\n".join(p["text"] for p in sorted(context["paragraphs"], key=lambda p: p["order"]))
    # Content-derived group catches identical tables with different upstream UUIDs.
    group = "tatqa-table-" + hashlib.sha256(canonical(table_text).encode()).hexdigest()[:24]
    rows = []
    for question in context["questions"]:
        answer = question.get("answer")
        if question.get("answer_type") not in {"arithmetic", "count"}:
            continue
        try:
            if isinstance(answer, bool):
                continue
            answer = float(str(answer).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(answer):
            continue
        scale = question.get("scale", "") or ""
        if scale not in {"", "thousand", "million", "billion", "percent"}:
            continue
        prompt = ("Answer the financial question using only the supplied table and context.\n"
                  "Return only a JSON object with keys answer (a number) and scale "
                  '(one of "", "thousand", "million", "billion", "percent"). '
                  "Use the scale implied by the question and source. Do not add an explanation.\n\n"
                  f"Table:\n{table_text}\n\nContext:\n{paragraphs}\n\nQuestion: {question['question']}")
        rows.append({"id": "tatqa-" + question["uid"], "group_id": group,
                     "prompt": prompt, "kind": "finance", "split": split,
                     "source": SOURCE, "license": "CC-BY-4.0", "language": "en",
                     "answer_type": "numeric_scale", "target": {"answer": answer, "scale": scale},
                     "tolerance": .01, "metadata": {"upstream_question_id": question["uid"],
                     "upstream_table_id": context["table"]["uid"], "revision": REVISION,
                     "upstream_answer_type": question["answer_type"], "question": question["question"]}})
    return rows


def build(raw_dir, output_dir, limits=None, seed=42, max_prompt_chars=0):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    if output_dir.exists():
        raise ValueError("Choose a fresh output directory to preserve previous artifacts")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources, candidates, filtered = {}, {}, Counter()
    for upstream, split in (("train", "train"), ("dev", "validation"), ("test_gold", "test")):
        filename = f"tatqa_dataset_{upstream}.json"
        url = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{REVISION}/dataset_raw/{filename}"
        # Use a revision-specific cache to avoid trusting files from another release.
        pinned_path = raw_dir / REVISION / filename
        if not pinned_path.exists():
            pinned_path.parent.mkdir(parents=True, exist_ok=True)
            pinned_path.write_bytes(urllib.request.urlopen(url, timeout=90).read())
        content = pinned_path.read_bytes()
        sources[split] = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
        contexts = json.loads(content)
        rows = [r for c in contexts for r in convert_context(c, split)]
        candidates[split] = rows
    # Give test and validation priority over train if a source table is repeated.
    seen_groups, seen_prompts, seen_ids = set(), set(), set()
    accepted = []
    counts = {}
    for split in ("test", "validation", "train"):
        rows = candidates[split]
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        keys = sorted(groups)
        random.Random(seed).shuffle(keys)
        current = []
        for key in keys:
            if key in seen_groups:
                filtered["cross_split_table_group"] += len(groups[key]); continue
            # Reserve every upstream group even if it is not sampled for this run.
            seen_groups.add(key)
            eligible = []
            for row in groups[key]:
                prompt_key = canonical(row["prompt"])
                if row["id"] in seen_ids or prompt_key in seen_prompts:
                    filtered["duplicate_id_or_prompt"] += 1; continue
                seen_ids.add(row["id"]); seen_prompts.add(prompt_key)
                if max_prompt_chars and len(row["prompt"]) > max_prompt_chars:
                    filtered["character_length_filter"] += 1; continue
                eligible.append(row)
            limit = (limits or {}).get(split, 0)
            if limit and len(current) + len(eligible) > limit:
                continue
            current.extend(eligible)
        counts[split] = len(current)
        accepted.extend(current)
    validate(accepted)
    if not all(counts.values()):
        raise ValueError(f"A split has no examples: {counts}")
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "master.jsonl", accepted)
    prepare(output_dir / "master.jsonl", output_dir)
    manifest = {"dataset": "TAT-QA numeric subset", "revision": REVISION, "license": "CC-BY-4.0",
                "attribution": "Fengbin Zhu et al., TAT-QA, ACL 2021", "license_source": LICENSE_URL,
                "sources": sources, "seed": seed, "limits": limits, "counts": counts,
                "numeric_candidates": {s: len(v) for s,v in candidates.items()},
                "filtered": dict(filtered), "max_prompt_chars": max_prompt_chars,
                "notes": ["Preserves official train/dev/test_gold partition roles.",
                          "Groups by normalized table content; report/company and semantic leakage still need review.",
                          "Character filter is not a tokenizer length guarantee.",
                          "Numeric accuracy requires both value and scale; not full official TAT-QA scoring.",
                          "Finance capability data only. Separate safety data remains required."]}
    (output_dir / "tatqa_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
