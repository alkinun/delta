import json
import difflib

def find_region(before_lines, after_lines):
    """Find the changed region and return (before_start, before_end, after_start, after_end)."""
    sm = difflib.SequenceMatcher(None, before_lines, after_lines)
    changes = [op for op in sm.get_opcodes() if op[0] != "equal"]

    if not changes:
        return None

    # Find the span of all changes
    before_start = min(op[1] for op in changes)
    before_end = max(op[2] for op in changes)
    after_start = min(op[3] for op in changes)
    after_end = max(op[4] for op in changes)

    return (before_start, before_end, after_start, after_end)

def format_sample(data):
    before_lines = data["before"].splitlines()
    after_lines = data["after"].splitlines()

    region = find_region(before_lines, after_lines)
    if region is None:
        return None

    b_start, b_end, a_start, a_end = region

    # Region (original code to transform)
    region_code = "\n".join(before_lines[b_start:b_end])

    # Output (transformed region)
    output = "\n".join(after_lines[a_start:a_end])

    # Format
    text = f"""[REGION]
{region_code}
[/REGION]
[INSTRUCTION]
{data["instruction"]}
[/INSTRUCTION]
[OUTPUT]
{output}
[/OUTPUT]"""

    return {
        "text": text,
        "repo": data["repo"],
        "sha": data["sha"],
        "message": data["message"],
        "filename": data["filename"],
        "before": data["before"],
        "after": data["after"],
        "instruction": data["instruction"],
        "region": region_code,
        "output": output,
    }

def main(path="dataset.jsonl", out="finetune.json"):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            sample = format_sample(data)

            if sample is None:
                print(f"  ⊘ #{i+1} no changes found, skipping")
                continue

            samples.append(sample)
            print(f"  ✓ #{i+1} formatted")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(samples)} samples → {out}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "finetune.json"
    main(path, out)
