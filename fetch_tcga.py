"""Download public, versioned cBioPortal TCGA-BRCA files with SHA256 checks.

Raw files stay in --cache; the harmonised cohort and provenance are written
under --out. No credentials, private data, or outcome-guided gene selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import requests

REPO = "cBioPortal/datahub"
STUDY = "brca_tcga_pan_can_atlas_2018"


def get_json(url):
    r = requests.get(url, timeout=(15, 60))
    r.raise_for_status()
    return r.json()


def download_file(study, name, revision, destination):
    rel = f"public/{study}/{name}"
    raw_url = f"https://raw.githubusercontent.com/{REPO}/{revision}/{rel}"
    r = requests.get(raw_url, timeout=(15, 60))
    r.raise_for_status()
    pointer = r.text if len(r.content) < 1000 else ""
    expected = None
    if pointer.startswith("version https://git-lfs.github.com/spec/v1"):
        expected = next(x.split("sha256:")[1] for x in pointer.splitlines() if x.startswith("oid "))
        url = f"https://media.githubusercontent.com/media/{REPO}/{revision}/{rel}"
    else:
        url = raw_url
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected:
        digest = hashlib.file_digest(destination.open("rb"), "sha256").hexdigest()
        if digest == expected:
            return {"url": url, "sha256": digest, "bytes": destination.stat().st_size, "revision": revision}
    partial = destination.with_suffix(destination.suffix + ".partial")
    digest = hashlib.sha256()
    total = 0
    print(f"Downloading {study}/{name}", flush=True)
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with partial.open("wb") as fh:
            for block in response.iter_content(1024 * 1024):
                fh.write(block)
                digest.update(block)
                total += len(block)
                if total and total % (20 * 1024 * 1024) == 0:
                    print(f"  {name}: {total // (1024 * 1024)} MiB", flush=True)
    actual = digest.hexdigest()
    if expected and actual != expected:
        raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {actual}")
    partial.replace(destination)
    return {"url": url, "sha256": actual, "bytes": total, "revision": revision}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", default="work/tcga_source")
    p.add_argument("--out", default="external_data/tcga_brca_harmonized.csv")
    p.add_argument("--revision", help="Datahub commit SHA; otherwise resolve and record the latest commit")
    args = p.parse_args()
    cache = Path(args.cache)
    revision = args.revision or get_json(f"https://api.github.com/repos/{REPO}/commits?per_page=1")[0]["sha"]
    manifest = {"study": STUDY, "repository": f"https://github.com/{REPO}", "revision": revision,
                "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": "overall survival, 1=death", "files": {},
                "mutations": "not downloaded; unavailable, never encoded as wild-type"}
    for study, names in [(STUDY, ["data_clinical_patient.txt", "data_clinical_sample.txt", "data_mrna_seq_v2_rsem.txt"]),
                         ("brca_tcga", ["data_clinical_patient.txt", "data_clinical_sample.txt"])]:
        for name in names:
            manifest['files'][f'{study}/{name}'] = download_file(study, name, revision, cache / study / name)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from external_data import prepare_tcga
    out = Path(args.out)
    prepare_tcga(cache / STUDY, out, cache / "brca_tcga")
    manifest['harmonized_sha256'] = hashlib.file_digest(out.open('rb'), 'sha256').hexdigest()
    Path(str(out) + ".provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == '__main__':
    main()
