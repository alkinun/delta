import json
import os
import sys
import difflib
import random
import requests
from tqdm import tqdm

BASE_URL = "http://0.0.0.0:8080"
MODEL = "local"

PROMPT_STYLES = [
    # Formality
    "written as a formal technical requirement (e.g. 'Refactor X to improve Y')",
    "written as a casual developer note (e.g. 'just clean up the X stuff')",
    # Granularity
    "high-level and conceptual, describing the goal without implementation details",
    "low-level and specific, mentioning exact functions, variables, or files changed",
    # Tone / person
    "written as a direct imperative command (e.g. 'Add error handling to...')",
    "written as a question or request (e.g. 'Can you make X do Y?')",
    # Style quirks
    "written with typos and informal shorthand, like a quick Slack message",
    "written as a git commit message (short subject line style)",
]


def make_prompt(data, prompt_style):
    before = data["before"].splitlines(keepends=True)
    after = data["after"].splitlines(keepends=True)

    diff = difflib.unified_diff(
        before, after,
        fromfile=f"before: {data['filename']}",
        tofile=f"after: {data['filename']}",
        n=8,
    )
    diff_text = "".join(diff)

    if len(diff_text) > 100000:
        diff_text = diff_text[:100000] + "\n... (truncated)"

    return f"""# Code Diff

```diff
{diff_text}
```

---

Please generate a super short instruction/prompt that would have resulted in this code diff. Respond only with the instruction/prompt. Make sure that the prompt you generate is {prompt_style}."""


def generate(path="dataset/diffs_raw.jsonl", out="dataset/dataset.jsonl", n=None):
    processed_shas = set()
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_shas.add(data["sha"])
        print(f"Resuming — {len(processed_shas)} existing instructions found in {out}")

    with open(path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    if n is not None:
        total_lines = min(total_lines, n)

    count = 0
    skipped = 0
    with open(path, "r", encoding="utf-8") as f, open(out, "a", encoding="utf-8") as fout:
        pbar = tqdm(total=total_lines, desc="Generating", unit="sample")
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break

            data = json.loads(line)

            if data["sha"] in processed_shas:
                skipped += 1
                pbar.update(1)
                continue

            prompt_style = random.choice(PROMPT_STYLES)
            prompt = make_prompt(data, prompt_style)

            print(f"\n{'='*70}")
            print(f"#{i+1} {data['sha'][:7]} | {data['message'][:50]}")
            print(f"Style: {prompt_style}")
            print(f"{'='*70}")

            try:
                resp = requests.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 1024,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                instruction = resp.json()["choices"][0]["message"]["content"].strip()

                data["instruction"] = instruction
                data["prompt_style"] = prompt_style
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"Instruction: {instruction}\n\n")
            except:
                print("Instruction: FAILED TO GENERATE\n\n")

            count += 1
            pbar.update(1)

    pbar.close()
    print(f"\nNew: {count} | Skipped: {skipped} | Total in file: {len(processed_shas) + count}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    generate(n=limit)
