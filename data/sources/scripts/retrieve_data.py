#!/usr/bin/env python3
"""Retrieve version-pinned native files and verify exact training projections."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or '\\' in relative or ':' in relative:
        raise ValueError('Manifest contains an unsafe relative path')
    destination = root.joinpath(*name.parts)
    try:
        destination.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError('Manifest path or symlink escapes the chosen root') from exc
    return destination


def records(manifest_path: Path, selected: list[str]) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text())
    entries = manifest['files']
    if len(entries) != manifest['file_count']:
        raise ValueError('Manifest file count differs from the entries')
    available = {item['benchmark'] for item in entries}
    unknown = set(selected) - available
    if unknown:
        raise ValueError('Benchmark selection is absent from this manifest: ' + ', '.join(sorted(unknown)))
    seen = set()
    for entry in entries:
        relative = entry.get('local_path', entry.get('path'))
        if relative in seen:
            raise ValueError('Manifest contains duplicate destination paths')
        seen.add(relative)
        if re.fullmatch('[0-9a-f]{64}', entry['sha256']) is None:
            raise ValueError('Manifest contains an invalid SHA256 value')
        if isinstance(entry['bytes'], int) is False or entry['bytes'] < 0:
            raise ValueError('Manifest contains an invalid byte count')
    return manifest, [entry for entry in entries if len(selected) == 0 or entry['benchmark'] in selected]


def check(path: Path, entry: dict) -> str:
    if path.is_file() is False:
        return 'missing'
    if path.stat().st_size != entry['bytes']:
        return 'size_mismatch'
    if sha256(path) != entry['sha256']:
        return 'sha256_mismatch'
    return 'verified'


def fetch(entry: dict, root: Path, cache: Path) -> str:
    destination = safe_path(root, entry['local_path'])
    if destination.exists():
        outcome = check(destination, entry)
        if outcome != 'verified':
            raise ValueError('Existing destination differs from the frozen source: ' + entry['local_path'])
        return 'verified_existing'
    if re.fullmatch('[0-9a-f]{40}', entry['revision']) is None:
        raise ValueError('A complete immutable repository revision is required')
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError('Install the pinned retrieval dependency from requirements-data.txt') from exc
    try:
        downloaded = Path(hf_hub_download(
            repo_id=entry['repo_id'], repo_type=entry['repo_type'],
            filename=entry['filename'], revision=entry['revision'],
            cache_dir=str(cache), token=None,
        ))
    except Exception as exc:
        # Provider exceptions can contain credential-bearing URLs.
        raise RuntimeError('Provider retrieval failed: ' + type(exc).__name__ +
                           '. Confirm repository access, acceptance of terms, credentials, and connectivity.') from None
    if check(downloaded, entry) != 'verified':
        raise ValueError('Provider bytes differ from the frozen source: ' + entry['filename'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix='.verified-source-', dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, 'wb') as target, downloaded.open('rb') as source:
            shutil.copyfileobj(source, target)
        if check(temporary, entry) != 'verified':
            raise ValueError('Copied file failed final verification')
        # A hard link creates the final name atomically and preserves existing files.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return 'downloaded_and_verified'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'fetch-native', 'verify-native', 'verify-projected'])
    parser.add_argument('--root', type=Path, help='External directory holding source or projected data')
    parser.add_argument('--benchmark', action='append', default=[])
    parser.add_argument('--manifest', type=Path, help='Explicit manifest, useful for an independently prepared verification set')
    parser.add_argument('--cache', type=Path, help='Hugging Face cache outside the release package')
    parser.add_argument('--receipt', type=Path, help='Create a fresh machine-readable receipt; existing files are preserved')
    args = parser.parse_args()
    kind = 'projected' if args.command == 'verify-projected' else 'native'
    manifest_path = args.manifest or PACKAGE / (kind + '_files.json')
    try:
        manifest, entries = records(manifest_path, args.benchmark)
        result = {
            'command': args.command, 'manifest_sha256': sha256(manifest_path),
            'selected_file_count': len(entries),
            'selected_benchmark_count': len({entry['benchmark'] for entry in entries}),
            'selected_bytes': sum(entry['bytes'] for entry in entries),
            'entries': [],
        }
        if args.command == 'plan':
            result['repository_revision'] = manifest.get('revision')
            result['status'] = 'plan_only'
        else:
            if args.root is None:
                raise ValueError('--root is required for retrieval and verification')
            root = args.root.expanduser().resolve()
            if args.command == 'fetch-native':
                try:
                    root.relative_to(PACKAGE)
                except ValueError:
                    pass
                else:
                    raise ValueError('Choose a raw-data root outside this public metadata package')
                root.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                relative = entry.get('local_path', entry.get('path'))
                destination = safe_path(root, relative)
                if args.command == 'fetch-native':
                    state = fetch(entry, root, args.cache or root / '.hf-cache')
                else:
                    state = check(destination, entry)
                result['entries'].append({'benchmark': entry['benchmark'], 'path': relative, 'status': state})
            failures = [item for item in result['entries']
                        if item['status'] in {'missing', 'size_mismatch', 'sha256_mismatch'}]
            result['verified_count'] = len(entries) - len(failures)
            result['failure_count'] = len(failures)
            result['status'] = 'verified' if len(failures) == 0 else 'incomplete'
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open('x') as output:
                json.dump(result, output, indent=2)
                output.write('\n')
        summary = {key: value for key, value in result.items() if key != 'entries'}
        print(json.dumps(summary, indent=2))
        if result['status'] == 'incomplete':
            for item in result['entries']:
                if item['status'] != 'verified':
                    print(item['status'] + ': ' + item['path'], file=sys.stderr)
            return 1
        return 0
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
