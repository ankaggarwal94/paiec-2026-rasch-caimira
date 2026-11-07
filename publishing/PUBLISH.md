# Publish the PAIEC release from Git Bash on Windows

Target: https://github.com/ankaggarwal94/paiec-2026-rasch-caimira
Collaborator: `maharshi95`, with the personal-repository collaborator role and push access.

The target was verified as public and empty during this repair. Keep it empty until publication. This helper supports an absent target or an empty public target. Existing Git references stop publication, preserving any remote history.

## Commands from the extracted release directory

```bash
bash publishing/publish.sh --source . --expected-owner ankaggarwal94 --expected-name paiec-2026-rasch-caimira --dry-run
bash publishing/publish.sh --source . --expected-owner ankaggarwal94 --expected-name paiec-2026-rasch-caimira
bash publishing/publish.sh --expected-owner ankaggarwal94 --expected-name paiec-2026-rasch-caimira --verify-only
```

Run the preview first, then publish after it succeeds. The launcher probes `python3`, `python`, and `py -3` and uses a working Python 3.9+ interpreter. Git and GitHub CLI must already be available and GitHub CLI authenticated as `ankaggarwal94`. Using `--source .` lets native Python resolve its own working directory and avoids Unix-to-Windows absolute path conversion. The older `publish.zsh` name remains a compatibility entry point when invoked through Bash.

The helper checks account identity, public visibility, administration permission, source files, and staging identity. It copies eligible files into a temporary repository and uses a normal push. Source history, virtual environments, caches, credential-shaped data, and raw-data download trees retain their existing safeguards. It performs zero force pushes and zero remote deletions. A private or populated target requires separate review.

## Empty-repository repair

The earlier helper required `defaultBranchRef` from `gh repo view` to equal JSON `null`. That check rejected the user's empty target. The replacement calls `git ls-remote --refs` and requires both a successful command and empty output. Any advertised branch or tag blocks publication. A connection or permission error also stops the operation.

The terminal excerpt establishes that staging succeeded before the guard failed. A live GitHub check then returned HTTP 409 with the message `Git Repository is empty.` The exact JSON value emitted by the user's installed CLI remains outside the captured transcript; the repair removes dependence on that representation entirely.

## Invitation and recovery

The publication step requests collaborator access after verifying the remote commit. A successful invitation appears as pending until Maharshi accepts it. Run `--verify-only` after acceptance to check effective write access.

Each run saves a `publication_receipt.json` and source manifest in the displayed temporary directory. The original source folder stays unchanged. A populated target causes a safe stop on subsequent publication attempts; use verification mode to inspect a successful push. If only collaborator setup remains, use the repository's collaborator settings to complete the invitation without republishing.

## Tests and scope

`python publishing/test_publisher.py` runs the offline regression suite. Twelve cases passed during this repair, including branch/tag refusal, transport failure, secret-file refusal, and real local empty/populated Git repositories. Four additional end-to-end cases used a mocked GitHub API with real local Git commits and pushes. These checks cover publication mechanics; live authenticated Windows publication remains the owner-run step.

Official command documentation: https://git-scm.com/docs/git-ls-remote and https://cli.github.com/manual/gh_repo_view . Personal-account collaborator permissions: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository .
