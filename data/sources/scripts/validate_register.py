#!/usr/bin/env python3
"""Cross-check the public register against its immutable historical receipts."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def require(condition, message):
    if condition is False:
        raise ValueError(message)


def main():
    rows = list(csv.DictReader((ROOT / 'DATA_SOURCES.csv').open()))
    freeze = read('references/freeze.json')
    layout = read('references/combined-sources.json')
    catalog = read('references/catalog-freeze-v6.json')['suites']
    historical = read('references/archives-native.json')['benchmarks']
    native = read('native_files.json')
    projected = read('projected_files.json')
    ledger = read('references/reference-ledger.json')
    for item in ledger['files']:
        content = (ROOT / item['path']).read_bytes()
        require(len(content) == item['bytes'], 'Reference size differs: ' + item['path'])
        require(hashlib.sha256(content).hexdigest() == item['sha256'],
                'Reference hash differs: ' + item['path'])
    names = {row['benchmark'] for row in rows}
    require(len(rows) == len(names) == 44, 'Register must contain 44 unique suites')
    require(names == set(layout['benchmarks']), 'Register differs from frozen suite membership')
    require(len({row['source_lineage'] for row in rows}) == 36, 'Lineage count differs')
    require(sum(int(row['retained_binary_trials']) for row in rows) == 10951005,
            'Retained trial count differs')
    require(len(native['files']) == native['file_count'] == 160, 'Native file count differs')
    require(len(projected['files']) == projected['file_count'] == 176, 'Projected file count differs')
    projected_hashes = {entry['path']: entry['sha256'] for entry in projected['files']}
    require(projected_hashes == freeze['sources'] == layout['expected_sha256'],
            'Projected manifest differs from historical freeze or layout')
    native_receipts = {**catalog, **historical}
    for entry in native['files']:
        stem = Path(entry['filename']).stem
        receipt = native_receipts[entry['benchmark']]['source'][stem]
        require(entry['bytes'] == receipt['bytes'] and entry['sha256'] == receipt['sha256'],
                'Native receipt differs: ' + entry['filename'])
    for row in rows:
        receipt = read(row['projection_receipt'])
        if row['benchmark'] in catalog:
            record = receipt['suites'][row['benchmark']]
        elif row['benchmark'] in historical:
            record = receipt['benchmarks'][row['benchmark']]
        else:
            record = receipt
        require(int(row['retained_binary_trials']) == record['counts']['binary_trials'],
                'Per-suite retained count differs: ' + row['benchmark'])
        for filename, entry in record['outputs'].items():
            path = row['projected_directory'] + '/' + filename
            require(projected_hashes[path] == entry['sha256'], 'Projection receipt differs: ' + path)
    source_metadata = read('references/publisher-source-metadata.json')['benchmarks']
    require(len(source_metadata) == 40, 'Publisher metadata count differs')
    gate = read('references/gate-build-info-extract.json')
    require(len(gate['benchmarks']) == 180, 'Extension source declaration count differs')
    print(json.dumps({'status': 'verified', 'historical_references': len(ledger['files']),
                      'suites': 44, 'lineages': 36, 'binary_trials': 10951005,
                      'native_files': 160, 'projected_files': 176,
                      'publisher_metadata_entries': 40,
                      'extension_declared_sources': 180}, indent=2))


if __name__ == '__main__':
    main()
