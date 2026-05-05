import os
import json
import time
import ast
import requests
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "repos": [
        ("django", "django"),
        ("pallets", "flask"),
        ("psf", "requests"),
        ("encode", "httpx"),
    ],
    "out": "diffs_raw.jsonl",
    "max_pages": 100000,
    "min_changes": 2,
    "max_changes": 100,
    "max_examples": 100000000,
    "sleep": 0,
    "skip_words": {
        "bump", "release", "merge", "revert", "changelog", "version",
        "wip", "fixup", "whitespace", "format", "lint", "style", "chore",
        "isort", "black", "flake8", "ci", "docs", "readme", "typo",
    },
    "token": os.environ.get("GITHUB_TOKEN"),
}

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"Bearer {CONFIG['token']}"} if CONFIG['token'] else {}),
}

# ── Data Model ────────────────────────────────────────────────────────────────
@dataclass
class DiffExample:
    repo: str
    sha: str
    message: str
    filename: str
    before: str
    after: str
    additions: int
    deletions: int

# ── GitHub API ───────────────────────────────────────────────────────────────
def gh_get(url: str, params: dict = None) -> Optional[dict]:
    """Simple GET with rate-limit handling."""
    for attempt in range(3):
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)

        # Show rate limit info
        if "X-RateLimit-Remaining" in r.headers:
            remaining = r.headers["X-RateLimit-Remaining"]
            limit = r.headers.get("X-RateLimit-Limit", "?")
            if int(remaining) < 100:
                print(f"  ⚠ Rate limit: {remaining}/{limit} remaining")

        if r.status_code == 200:
            return r.json()
        if r.status_code == 403:
            wait = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time(), 5)
            print(f"  Rate limited — sleeping {wait:.0f}s")
            time.sleep(wait)
        else:
            print(f"  HTTP {r.status_code} for {url}")
            time.sleep(2 ** attempt)
    return None

def iter_commits(owner: str, repo: str, max_pages: int):
    for page in range(1, max_pages + 1):
        data = gh_get(f"https://api.github.com/repos/{owner}/{repo}/commits",
                      params={"per_page": 100, "page": page})
        if not data:
            break
        yield from data
        if len(data) < 100:
            break
        time.sleep(CONFIG['sleep'])

def get_file_content(owner: str, repo: str, commit_sha: str, path: str) -> Optional[str]:
    """Get file content at a specific commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    params = {"ref": commit_sha}
    data = gh_get(url, params=params)
    if not data:
        return None

    if data.get("encoding") == "base64":
        import base64
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            return None
    return None

def parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# ── Filters ───────────────────────────────────────────────────────────────────
def is_good_message(msg: str) -> bool:
    first = msg.split("\n")[0].strip()
    words = first.split()
    if not (2 <= len(words) <= 20) or len(first) < 10:
        return False
    if any(w in first.lower() for w in CONFIG['skip_words']):
        return False
    return not first.startswith(("/", ".", "#"))

def is_good_file(file: dict) -> bool:
    filename = file.get("filename", "")
    if not filename.endswith(".py") or file.get("patch") is None:
        return False
    if filename.startswith("test"):
        return False
    # Exclude generated files
    return not any(s in filename for s in ("_pb2.py", "conf.py", ".pb2"))

def is_good_diff(file: dict) -> bool:
    adds, dels = file.get("additions", 0), file.get("deletions", 0)
    total = adds + dels
    return CONFIG['min_changes'] <= total <= CONFIG['max_changes']

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def scrape_gen(owner: str, repo: str):
    """Generator that yields DiffExample one at a time from a repo."""
    stats = {"seen": 0, "skipped": 0}

    print(f"  [{owner}/{repo}] Starting scrape")

    for commit in iter_commits(owner, repo, CONFIG['max_pages']):
        sha, message = commit["sha"], commit["commit"]["message"]
        stats["seen"] += 1

        if not is_good_message(message):
            continue

        detail = gh_get(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}")
        if not detail:
            continue

        # Rate limiting sleep
        if stats["seen"] % 5 == 0:
            time.sleep(CONFIG['sleep'] * 3)

        py_files = [f for f in detail.get("files", []) if is_good_file(f)]
        if not py_files:
            continue

        # Take the first good Python file
        file = py_files[0]
        if not is_good_diff(file):
            continue

        filename = file["filename"]

        # Get parent commit for "before" content
        parents = commit.get("parents", [])
        if not parents:
            continue
        parent_sha = parents[0]["sha"]

        # Fetch "after" content (current commit)
        after_content = get_file_content(owner, repo, sha, filename)
        if after_content is None:
            continue

        # Fetch "before" content (parent commit)
        before_content = get_file_content(owner, repo, parent_sha, filename)
        if before_content is None:
            before_content = ""

        # Validate Python (only validate non-empty content)
        if before_content and not parses_as_python(before_content):
            continue
        if after_content and not parses_as_python(after_content):
            continue

        first_line = message.split("\n")[0].strip()
        example = DiffExample(
            repo=f"{owner}/{repo}", sha=sha, message=first_line,
            filename=filename, before=before_content, after=after_content,
            additions=file["additions"], deletions=file["deletions"],
        )
        print(f"  ✓ [{owner}/{repo}] {sha[:7]}  [{file['additions']:+d}/{file['deletions']:+d}]  {first_line[:60]}")
        yield example

    print(f"  [{owner}/{repo}] Exhausted after {stats['seen']} commits")

def save_jsonl(examples: list[DiffExample], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            data = asdict(ex)
            # Truncate large content for storage (first 5000 lines each)
            data["before"] = "\n".join(data["before"].split("\n")[:5000])
            data["after"] = "\n".join(data["after"].split("\n")[:5000])
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(examples)} examples → {path}")

def print_sample(examples: list[DiffExample], n: int = 2):
    print("\n── Sample ──")
    for ex in examples[:n]:
        print(f"\nSHA: {ex.sha[:7]} | File: {ex.filename} | {ex.message}")
        print(f"Changes: +{ex.additions}/-{ex.deletions}")
        before_lines = ex.before.split("\n")
        after_lines = ex.after.split("\n")
        print(f"--- BEFORE ({len(before_lines)} lines) ---")
        print("\n".join(before_lines[:20]))
        if len(before_lines) > 20:
            print(f"... ({len(before_lines) - 20} more lines)")
        print(f"--- AFTER ({len(after_lines)} lines) ---")
        print("\n".join(after_lines[:20]))
        if len(after_lines) > 20:
            print(f"... ({len(after_lines) - 20} more lines)")
        print(f"{'─'*60}")

if __name__ == "__main__":
    import hashlib
    from itertools import zip_longest

    # Load existing examples for resume support
    all_examples = []
    seen_hashes = set()
    seen_shas = set()

    if os.path.exists(CONFIG["out"]):
        with open(CONFIG["out"], "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                h = hashlib.sha256(f"{data['before']}|||{data['after']}".encode()).hexdigest()[:16]
                seen_hashes.add(h)
                seen_shas.add(data["sha"])
        print(f"Resuming — {len(seen_shas)} existing examples found in {CONFIG['out']}")

    remaining = CONFIG["max_examples"] - len(seen_shas)

    if remaining <= 0:
        print(f"Already have {len(seen_shas)} examples (target: {CONFIG['max_examples']}). Nothing to do.")
        exit(0)

    print(f"Scraping {len(CONFIG['repos'])} repos — {remaining} more needed (target: {CONFIG['max_examples']})")
    print("Repos: " + ", ".join(f"{o}/{r}" for o, r in CONFIG["repos"]))

    # Create one generator per repo
    generators = [scrape_gen(owner, repo) for owner, repo in CONFIG["repos"]]

    # Progress bar (starts from existing count)
    pbar = tqdm(total=CONFIG["max_examples"], desc="Collecting", unit="example",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                initial=len(seen_shas))

    # Open file for appending (not overwriting)
    outfile = open(CONFIG["out"], "a", encoding="utf-8")

    def save_example(ex):
        data = asdict(ex)
        data["before"] = "\n".join(data["before"].split("\n")[:5000])
        data["after"] = "\n".join(data["after"].split("\n")[:5000])
        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
        outfile.flush()

    # Round-robin: take one from each repo in turn
    for round_num, results in enumerate(zip_longest(*generators)):
        for result in results:
            if result is None:
                continue

            # Skip already seen SHAs
            if result.sha in seen_shas:
                continue

            # Global dedup
            h = hashlib.sha256(f"{result.before}|||{result.after}".encode()).hexdigest()[:16]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            seen_shas.add(result.sha)

            all_examples.append(result)
            save_example(result)
            pbar.update(1)

            if len(seen_shas) >= CONFIG["max_examples"]:
                break

        if len(seen_shas) >= CONFIG["max_examples"]:
            break

    pbar.close()
    outfile.close()

    total_collected = len(seen_shas)
    new_collected = len(all_examples)

    print(f"\n── Done ──")
    print(f"  New: {new_collected} | Total: {total_collected} examples from {len(CONFIG['repos'])} repos")
    print(f"  Saved to: {CONFIG['out']}")

    if all_examples:
        print_sample(all_examples)
    elif new_collected == 0:
        print("No new examples collected — already up to date or check token/filters.")
