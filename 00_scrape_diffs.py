import os
import re
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
        # Python
        ("django", "django",       "python"),
        ("pallets", "flask",       "python"),
        ("psf", "requests",        "python"),
        ("encode", "httpx",        "python"),

        # Rust
        ("tokio-rs", "tokio",      "rust"),
        ("serde-rs", "serde",      "rust"),
        ("actix", "actix-web",     "rust"),
        ("BurntSushi", "ripgrep",  "rust"),

        # Go
        ("gin-gonic", "gin",       "go"),
        ("gofiber", "fiber",       "go"),
        ("labstack", "echo",       "go"),
        ("go-chi", "chi",          "go"),
    ],
    "out": "dataset/diffs_raw.jsonl",
    "max_pages": 100000,
    "min_changes": 2,
    "max_changes": 100,
    "max_examples": 100000000,
    "sleep": 0,
    "skip_words": {
        "bump", "release", "merge", "revert", "changelog", "version",
        "wip", "fixup", "whitespace", "format", "lint", "style", "chore",
        "isort", "black", "flake8", "ci", "docs", "readme", "typo",
        "rustfmt", "gofmt", "clippy", "golint",
    },
    "token": os.environ.get("GITHUB_TOKEN"),
}

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"Bearer {CONFIG['token']}"} if CONFIG['token'] else {}),
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
}

# ── Data Model ────────────────────────────────────────────────────────────────
@dataclass
class DiffExample:
    repo: str
    sha: str
    message: str
    filename: str
    language: str
    before: str
    after: str
    additions: int
    deletions: int

# ── GitHub API ────────────────────────────────────────────────────────────────
def gh_get(url: str, params: dict = None) -> Optional[dict]:
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"  Network error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
            continue

        if "X-RateLimit-Remaining" in r.headers:
            remaining = r.headers["X-RateLimit-Remaining"]
            limit = r.headers.get("X-RateLimit-Limit", "?")
            if int(remaining) < 100:
                print(f"  ⚠ Rate limit: {remaining}/{limit} remaining")

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as e:
                print(f"  Failed to parse JSON from {url}: {e}")
                return None
        if r.status_code == 403:
            wait = max(int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time(), 5)
            print(f"  Rate limited — sleeping {wait:.0f}s")
            time.sleep(wait)
        elif r.status_code == 404:
            print(f"  Not found: {url}")
            return None
        else:
            print(f"  HTTP {r.status_code} for {url}")
            time.sleep(2 ** attempt)

    print(f"  Giving up on {url} after 3 attempts")
    return None

def iter_commits(owner: str, repo: str, max_pages: int):
    for page in range(1, max_pages + 1):
        data = gh_get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            params={"per_page": 100, "page": page},
        )
        if not data:
            break
        if not isinstance(data, list):
            print(f"  Unexpected response type for commits: {type(data)}")
            break
        yield from data
        if len(data) < 100:
            break
        time.sleep(CONFIG["sleep"])

def get_file_content(owner: str, repo: str, commit_sha: str, path: str) -> Optional[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    data = gh_get(url, params={"ref": commit_sha})
    if not data:
        return None
    if data.get("encoding") == "base64":
        import base64
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except Exception as e:
            print(f"  Failed to decode file content for {path}: {e}")
            return None
    return None

# ── Language Validators ───────────────────────────────────────────────────────
def parses_as_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def _braces_balanced(code: str) -> bool:
    depth = 0
    in_line_comment = False
    in_block_comment = False
    i = 0
    while i < len(code):
        c = code[i]
        if not in_line_comment and not in_block_comment and code[i:i+2] == "/*":
            in_block_comment = True
            i += 2
            continue
        if in_block_comment:
            if code[i:i+2] == "*/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if not in_block_comment and code[i:i+2] == "//":
            in_line_comment = True
            i += 2
            continue
        if in_line_comment and c == "\n":
            in_line_comment = False
        if in_line_comment:
            i += 1
            continue
        if c in ('"', "'"):
            quote = c
            i += 1
            while i < len(code) and code[i] != quote:
                if code[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0

def validates_as_rust(code: str) -> bool:
    if not code.strip():
        return True
    if not _braces_balanced(code):
        return False
    return bool(re.search(r"\b(fn|let|mut|pub|use|mod|impl|struct|enum|trait|match|async|await)\b", code))

def validates_as_go(code: str) -> bool:
    if not code.strip():
        return True
    if not _braces_balanced(code):
        return False
    return bool(re.search(r"^\s*package\s+\w+", code, re.MULTILINE))

VALIDATORS = {
    "python": parses_as_python,
    "rust":   validates_as_rust,
    "go":     validates_as_go,
}

def is_valid_source(code: str, language: str) -> bool:
    validator = VALIDATORS.get(language)
    return validator(code) if validator else True

# ── Filters ───────────────────────────────────────────────────────────────────
_SKIP_PATTERNS: dict[str, list[str]] = {
    "python": ["_pb2.py", "conf.py", ".pb2"],
    "rust":   ["build.rs"],
    "go":     [".pb.go", "_gen.go", "mock_", "bindata.go"],
}

def _get_lang_from_filename(filename: str) -> Optional[str]:
    for ext, lang in EXT_TO_LANG.items():
        if filename.endswith(ext):
            return lang
    return None

def is_good_message(msg: str) -> bool:
    first = msg.split("\n")[0].strip()
    words = first.split()
    if not (2 <= len(words) <= 20) or len(first) < 10:
        return False
    if any(w in first.lower() for w in CONFIG["skip_words"]):
        return False
    return not first.startswith(("/", ".", "#"))

def is_good_file(file: dict, repo_lang: str) -> bool:
    filename = file.get("filename", "")
    detected = _get_lang_from_filename(filename)
    if detected is None or detected != repo_lang:
        return False
    if file.get("patch") is None:
        return False
    basename = os.path.basename(filename)
    if basename.startswith("test") or basename.endswith("_test.go") or "_test." in basename:
        return False
    skip_pats = _SKIP_PATTERNS.get(repo_lang, [])
    return not any(s in filename for s in skip_pats)

def is_good_diff(file: dict) -> bool:
    adds, dels = file.get("additions", 0), file.get("deletions", 0)
    return CONFIG["min_changes"] <= adds + dels <= CONFIG["max_changes"]

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def scrape_gen(owner: str, repo: str, repo_lang: str):
    print(f"  [{owner}/{repo}] Starting scrape (language: {repo_lang})")
    seen = 0

    for commit in iter_commits(owner, repo, CONFIG["max_pages"]):
        try:
            sha, message = commit["sha"], commit["commit"]["message"]
        except (KeyError, TypeError) as e:
            print(f"  Malformed commit object, skipping: {e}")
            continue

        seen += 1

        if not is_good_message(message):
            continue

        detail = gh_get(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}")
        if not detail:
            continue

        if seen % 5 == 0:
            time.sleep(CONFIG["sleep"] * 3)

        py_files = [f for f in detail.get("files", []) if is_good_file(f, repo_lang)]
        if not py_files:
            continue

        file = py_files[0]
        if not is_good_diff(file):
            continue

        filename = file["filename"]
        lang = _get_lang_from_filename(filename) or repo_lang

        parents = commit.get("parents", [])
        if not parents:
            continue
        parent_sha = parents[0]["sha"]

        after_content = get_file_content(owner, repo, sha, filename)
        if after_content is None:
            continue

        before_content = get_file_content(owner, repo, parent_sha, filename) or ""

        if before_content and not is_valid_source(before_content, lang):
            continue
        if after_content and not is_valid_source(after_content, lang):
            continue

        first_line = message.split("\n")[0].strip()
        example = DiffExample(
            repo=f"{owner}/{repo}",
            sha=sha,
            message=first_line,
            filename=filename,
            language=lang,
            before=before_content,
            after=after_content,
            additions=file["additions"],
            deletions=file["deletions"],
        )
        print(
            f"  ✓ [{owner}/{repo}|{lang}] {sha[:7]}"
            f"  [{file['additions']:+d}/{file['deletions']:+d}]"
            f"  {first_line[:60]}"
        )
        yield example

    print(f"  [{owner}/{repo}] Exhausted after {seen} commits")

def print_sample(examples: list[DiffExample], n: int = 2):
    print("\n── Sample ──")
    for ex in examples[:n]:
        print(f"\nSHA: {ex.sha[:7]} | Lang: {ex.language} | File: {ex.filename} | {ex.message}")
        print(f"Changes: +{ex.additions}/-{ex.deletions}")
        before_lines = ex.before.split("\n")
        after_lines  = ex.after.split("\n")
        print(f"--- BEFORE ({len(before_lines)} lines) ---")
        print("\n".join(before_lines[:20]))
        if len(before_lines) > 20:
            print(f"... ({len(before_lines) - 20} more lines)")
        print(f"--- AFTER ({len(after_lines)} lines) ---")
        print("\n".join(after_lines[:20]))
        if len(after_lines) > 20:
            print(f"... ({len(after_lines) - 20} more lines)")
        print("─" * 60)

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import hashlib
    from itertools import zip_longest

    os.makedirs(os.path.dirname(CONFIG["out"]), exist_ok=True)

    print(f"Scraping {len(CONFIG['repos'])} repos (target: {CONFIG['max_examples']} examples)")
    print("Repos: " + ", ".join(f"{o}/{r} ({l})" for o, r, l in CONFIG["repos"]))

    generators = [scrape_gen(owner, repo, lang) for owner, repo, lang in CONFIG["repos"]]
    examples: list[DiffExample] = []
    seen_hashes: set[str] = set()

    pbar = tqdm(
        total=CONFIG["max_examples"],
        desc="Collecting",
        unit="example",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    try:
        with open(CONFIG["out"], "w", encoding="utf-8") as outfile:
            for results in zip_longest(*generators):
                for result in results:
                    if result is None:
                        continue

                    h = hashlib.sha256(
                        f"{result.before}|||{result.after}".encode()
                    ).hexdigest()[:16]
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    examples.append(result)
                    try:
                        data = asdict(result)
                        data["before"] = "\n".join(data["before"].split("\n")[:5000])
                        data["after"]  = "\n".join(data["after"].split("\n")[:5000])
                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                        outfile.flush()
                    except (TypeError, ValueError, OSError) as e:
                        print(f"  Failed to write example {result.sha[:7]}: {e}")
                        continue

                    pbar.update(1)

                    if len(examples) >= CONFIG["max_examples"]:
                        break

                if len(examples) >= CONFIG["max_examples"]:
                    break

    except KeyboardInterrupt:
        print(f"\n  Interrupted — saved {len(examples)} examples so far")
    finally:
        pbar.close()

    print("\n── Done ──")
    print(f"  Total: {len(examples)} examples from {len(CONFIG['repos'])} repos")
    print(f"  Saved to: {CONFIG['out']}")

    if examples:
        print_sample(examples)
