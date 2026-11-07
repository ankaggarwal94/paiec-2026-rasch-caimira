"""Publish a reviewed PAIEC release through the user's authenticated GitHub CLI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

OWNER = "ankaggarwal94"
NAME = "paiec-2026-rasch-caimira"
COLLABORATOR = "maharshi95"
TARGET = f"{OWNER}/{NAME}"
URL = f"https://github.com/{TARGET}"
EXCLUDED = {".git", ".venv", ".venvs", "venv", "venvs", "env", "envs",
            "virtualenv", "virtualenvs", ".virtualenvs", ".tox", ".nox", "node_modules",
            "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
            "internal_leads", ".ds_store", ".clean_venv", ".clean-venv", ".hf-cache", ".cache"}
ENV_PREFIXES = (".venv-", ".venv_", "venv-", "venv_", ".virtualenv-",
                ".virtualenv_", "virtualenv-", "virtualenv_", ".clean_venv-",
                ".clean_venv_", ".clean-venv-", ".clean-venv_")
EXCLUDED_DIRECTORY_SUFFIXES = {("data", "raw"), ("data", "materialized"),
                               ("development", "outputs")}
BUILD_SUFFIXES = (".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz",
                  ".toc", ".lof", ".lot", ".blg", ".nav", ".snm", ".vrb", ".pyc", ".pyo")
SECRET_MARKERS = re.compile(rb"(?:github_pat_[A-Za-z0-9_]{35,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{40,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default=".", help="Reviewed release directory; defaults to the current directory")
parser.add_argument("--expected-owner", required=True)
parser.add_argument("--expected-name", required=True)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--dry-run", action="store_true", help="Stage and validate; perform read-only GitHub checks")
mode.add_argument("--verify-only", action="store_true", help="Read public visibility and current collaborator status")
args = parser.parse_args()
receipt = {"target": TARGET, "collaborator": COLLABORATOR,
           "started_utc": datetime.now(timezone.utc).isoformat(), "events": []}
workspace = None
env = dict(os.environ)
for key in list(env):
    if key.startswith("GIT_") or key in {"GH_DEBUG", "GH_FORCE_TTY"}:
        env.pop(key)
env.update({"GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1",
            "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull})

def fail(message):
    raise RuntimeError(message)

def command(argv, *, cwd=None, input_data=None, allowed=(0,)):
    result = subprocess.run(argv, cwd=cwd, env=env, input=input_data,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode in allowed:
        return result
    fail(f"{Path(argv[0]).name} failed with exit {result.returncode}; captured command output was withheld to protect credentials.")

def api(endpoint, method="GET", payload=None, allowed=(200,)):
    argv = ["gh", "api", "--hostname", "github.com", "--include", "--method", method,
            "--header", "Accept: application/vnd.github+json", endpoint]
    data = None
    if payload is None:
        pass
    else:
        argv.extend(["--input", "-"])
        data = json.dumps(payload).encode()
    result = command(argv, input_data=data, allowed=(0, 1))
    raw = result.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    headers, separator, body = raw.partition("\n\n")
    match = re.search(r"^HTTP/\S+\s+(\d{3})", headers, re.MULTILINE)
    if match is None or separator == "":
        fail("GitHub response lacks a readable HTTP status; check gh authentication and connectivity.")
    status = int(match.group(1))
    if status in allowed:
        return status, json.loads(body) if body.strip() else None
    fail(f"GitHub API returned HTTP {status} for {method} {endpoint.split('?')[0]}.")

def repo_state(allow_absent=True):
    allowed = (200, 404) if allow_absent else (200,)
    status, value = api(f"repos/{TARGET}", allowed=allowed)
    if status == 404:
        return None
    if value.get("full_name") != TARGET or value.get("owner", {}).get("login") != OWNER:
        fail("Repository identity differs from the explicit target.")
    if value.get("private") is False and value.get("visibility") == "public":
        pass
    else:
        fail("The existing target requires public visibility; automatic visibility changes are disabled.")
    if value.get("archived") or value.get("disabled") or value.get("fork"):
        fail("The target is archived, disabled, or a fork; review it separately.")
    if value.get("permissions", {}).get("admin") is True:
        return value
    fail("The authenticated account lacks repository administration permission.")

def require_empty():
    """Check advertised Git references, independent of CLI JSON null encoding."""
    result = command(["git", "ls-remote", "--refs", URL + ".git"])
    refs = result.stdout.strip()
    if refs:
        fail("The target already contains Git references; publication stopped before any push.")
    # A successful empty ls-remote establishes that no advertised refs exist.
    # Connection, authentication and server errors are rejected by command().
    return

def access_state():
    status, value = api(f"repos/{TARGET}/collaborators/{COLLABORATOR}/permission", allowed=(200, 404))
    if status == 200 and value.get("permission") in {"write", "admin"}:
        if value.get("user", {}).get("login") != COLLABORATOR:
            fail("Collaborator identity differs from the expected username.")
        return {"status": "accepted", "permission": value["permission"],
                "role_name": value.get("role_name")}
    for page in range(1, 101):
        _, invites = api(f"repos/{TARGET}/invitations?per_page=100&page={page}")
        for invite in invites:
            if invite.get("invitee", {}).get("login") == COLLABORATOR:
                return {"status": "pending", "invitation_id": invite.get("id"),
                        "permission": invite.get("permissions")}
        if len(invites) < 100:
            return {"status": "access_unconfirmed"}
    fail("Invitation pagination exceeded its safety bound.")

def save_receipt():
    if workspace is None:
        return
    (workspace / "publication_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

def event(name, **fields):
    receipt["events"].append({"step": name, **fields})
    save_receipt()

def collect(source, destination=None):
    if (source / "pyvenv.cfg").is_file():
        fail("The selected source is a Python virtual environment; choose the release directory.")
    manifest = {}
    exclusions = 0
    for base, dirs, files in os.walk(source, followlinks=False):
        base_path = Path(base)
        for name in list(dirs):
            path = base_path / name
            parts = tuple(part.lower() for part in path.relative_to(source).parts)
            if (name.lower() in EXCLUDED or name.lower().startswith(ENV_PREFIXES)
                    or parts[-2:] in EXCLUDED_DIRECTORY_SUFFIXES):
                dirs.remove(name)
                exclusions += 1
            elif path.is_symlink():
                fail("A source directory is a symbolic link; use a self-contained release tree.")
            elif (path / "pyvenv.cfg").is_file():
                dirs.remove(name)
                exclusions += 1
        for name in sorted(files):
            path = base_path / name
            relative = path.relative_to(source)
            lower = name.lower()
            parts = tuple(part.lower() for part in relative.parts)
            mutable_runtime_output = (parts[-3:-1] == ("runtime", "results")
                                      and (lower == "smoke_results.json"
                                           or (lower.startswith("fresh_cli_") and lower.endswith(".jsonl"))))
            if (lower in EXCLUDED or lower.endswith(BUILD_SUFFIXES)
                    or lower == "publication-receipt.json" or mutable_runtime_output):
                exclusions += 1
                continue
            if path.is_symlink() or stat.S_ISREG(path.stat().st_mode) is False:
                fail("A source entry is a symbolic link or special file.")
            if lower == ".env" or lower.startswith(".env.") or lower in {".netrc", ".git-credentials", "id_rsa", "id_ed25519", ".gitmodules"}:
                fail("A credential, environment, or submodule file requires removal from the public source tree.")
            if any(ord(char) < 32 for char in str(relative)):
                fail("A source path contains control characters.")
            if path.stat().st_size > 100 * 1024 * 1024:
                fail("A source file exceeds GitHub's 100 MiB Git limit.")
            content = path.read_bytes()
            if SECRET_MARKERS.search(content):
                fail("A credential-shaped value requires review; its contents were withheld.")
            manifest[relative.as_posix()] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                           "executable": bool(path.stat().st_mode & 0o111)}
            if destination is None:
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o755 if manifest[relative.as_posix()]["executable"] else 0o644)
    return manifest, exclusions

try:
    if args.expected_owner != OWNER or args.expected_name != NAME:
        fail(f"This helper supports only {TARGET}.")
    for program in ("gh", "git"):
        if shutil.which(program) is None:
            fail(f"Install {program} before running this helper.")
    _, identity = api("user")
    if identity.get("login") != OWNER or identity.get("type") != "User":
        fail(f"Authenticate gh as the personal account {OWNER}; the current identity differs.")
    state = repo_state()
    if args.verify_only:
        if state is None:
            fail("The target repository is absent.")
        result = {"repository": URL, "visibility": "public", "collaborator": COLLABORATOR, **access_state()}
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["status"] in {"accepted", "pending"} else 2)
    source_arg = Path(args.source).expanduser()
    if source_arg.is_symlink():
        fail("The source directory is a symbolic link; select the reviewed release directory.")
    source = source_arg.resolve(strict=True)
    if source.is_dir() is False or source == Path(source.anchor):
        fail("Supply a specific release directory.")
    if state is None:
        pass
    else:
        require_empty()
    workspace = Path(tempfile.mkdtemp(prefix="paiec-publication-"))
    stage = workspace / "repository"
    stage.mkdir()
    manifest, exclusions = collect(source, stage)
    if len(manifest) == 0:
        fail("The reviewed source contains zero eligible files.")
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (workspace / "source_manifest.json").write_bytes(manifest_bytes)
    receipt.update({"source": str(source), "staging_directory": str(stage),
                    "files": len(manifest), "excluded_entries": exclusions,
                    "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "dry_run": args.dry_run})
    env["GIT_CONFIG_GLOBAL"] = str(workspace / "isolated_gitconfig")
    command(["git", "init", "--initial-branch=main", str(stage)])
    def git(*values):
        return command(["git", "-c", "core.hooksPath=" + os.devnull,
                        "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false",
                        *values], cwd=stage)
    git("config", "user.name", OWNER)
    git("config", "user.email", f"{identity['id']}+{OWNER}@users.noreply.github.com")
    # Include every reviewed eligible file, including files covered by .gitignore.
    git("add", "--all", "--force", "--", ".")
    git("commit", "--message", "Release PAIEC 2026 reproducibility package")
    commit = git("rev-parse", "HEAD").stdout.decode().strip()
    entries = git("ls-tree", "-r", "-z", "HEAD").stdout.split(b"\0")
    committed = {}
    for entry in entries:
        if len(entry) == 0:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        file_mode, kind, object_id = metadata.decode().split()
        if kind != "blob":
            fail("The commit contains an unexpected Git object type.")
        content = git("cat-file", "blob", object_id).stdout
        committed[raw_path.decode()] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                        "executable": file_mode == "100755"}
    if committed != manifest:
        fail("Committed bytes differ from the reviewed snapshot; inspect Git attributes.")
    current, _ = collect(source)
    if current != manifest:
        fail("Source files changed during staging; rerun after edits finish.")
    receipt["commit"] = commit
    event("local_snapshot_verified")
    print(f"Verified {len(manifest)} files; excluded {exclusions} local metadata/cache entries.")
    print(f"Local receipt directory: {workspace}")
    if args.dry_run:
        event("dry_run_complete")
        print("Dry run complete. Remote writes: zero.")
        sys.exit(0)
    command(["gh", "auth", "setup-git", "--hostname", "github.com"])
    latest = repo_state()
    if latest is None:
        command(["gh", "repo", "create", TARGET, "--public", "--description",
                 "PAIEC 2026 Rasch and CAIMIRA report, inference artifacts, and reproducibility materials"])
        event("public_repository_created")
        repo_state(allow_absent=False)
    require_empty()
    git("remote", "add", "origin", URL + ".git")
    git("push", "--set-upstream", "origin", "main:main")
    event("main_pushed", commit=commit)
    remote = repo_state(allow_absent=False)
    _, branch = api(f"repos/{TARGET}/git/ref/heads/main")
    if branch.get("object", {}).get("sha") != commit:
        fail("Remote main differs from the prepared commit.")
    if remote.get("default_branch") != "main":
        fail("Remote default branch differs from main; review repository settings.")
    event("public_commit_verified")
    invitation_status, invitation = api(f"repos/{TARGET}/collaborators/{COLLABORATOR}",
                                         method="PUT", payload={}, allowed=(201, 204))
    event("collaborator_request", http_status=invitation_status,
          invitation_id=invitation.get("id") if isinstance(invitation, dict) else None)
    access = access_state()
    event("collaborator_verified", **access)
    print(f"Published {URL} at commit {commit}.")
    if access["status"] == "accepted":
        print(f"{COLLABORATOR}: accepted collaborator with {access['permission']} access.")
    elif access["status"] == "pending":
        print(f"{COLLABORATOR}: invitation pending; push access activates after acceptance.")
    else:
        fail("Publication succeeded; collaborator access requires follow-up verification.")
except (RuntimeError, OSError, ValueError, KeyError) as error:
    receipt["failure"] = str(error)
    save_receipt()
    print("Stopped: " + str(error), file=sys.stderr)
    if workspace is None:
        pass
    else:
        print(f"Recovery receipt: {workspace / 'publication_receipt.json'}", file=sys.stderr)
    sys.exit(1)
