import json
import os
import sys
import difflib
import requests
from tqdm import tqdm

BASE_URL = "http://0.0.0.0:8080"
MODEL = "local"

def make_prompt(data):
    before = data["before"].splitlines(keepends=True)
    after = data["after"].splitlines(keepends=True)

    diff = difflib.unified_diff(
        before, after,
        fromfile=f"before: {data['filename']}",
        tofile=f"after: {data['filename']}",
        n=8,
    )
    diff_text = "".join(diff)

    # Truncate if still too long
    if len(diff_text) > 4000:
        diff_text = diff_text[:4000] + "\n... (truncated)"

    return f"""# Code Diff

```diff
{diff_text}
```

---

Please generate a super short instruction/prompt that would have resulted in this code diff. Respond only with the instruction/prompt."""

def generate(path="diffs_raw.jsonl", out="dataset.jsonl", n=None):
    # Load existing processed SHAs for resume support
    processed_shas = set()
    if os.path.exists(out):
        with open(out, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                processed_shas.add(data["sha"])
        print(f"Resuming — {len(processed_shas)} existing instructions found in {out}")

    # Count total lines for progress bar
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

            # Skip already processed
            if data["sha"] in processed_shas:
                skipped += 1
                pbar.update(1)
                continue

            prompt = make_prompt(data)

            print(f"\n{'='*70}")
            print(f"#{i+1} {data['sha'][:7]} | {data['message'][:50]}")
            print(f"{'='*70}")

            resp = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 200,
                },
                timeout=120,
            )
            resp.raise_for_status()
            instruction = resp.json()["choices"][0]["message"]["content"].strip()

            print(f"Instruction: {instruction}")

            # Write original data + instruction to output
            data["instruction"] = instruction
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            fout.flush()
            count += 1
            pbar.update(1)

    pbar.close()
    print(f"\nNew: {count} | Skipped: {skipped} | Total in file: {len(processed_shas) + count}")
    print(f"Saved → {out}")

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    generate(n=limit)
