#!/bin/bash
# Inside the interactive build job of batch 3, after cotd_4b.sif is built:  bash fetch_model_4b.sh /staging/<netid>
# Downloads Qwen/Qwen3.5-4B (the post-trained release, Apache-2.0, not gated) at the revision of all our 4B runs with the
# image's python and writes /staging/<netid>/Qwen3.5-4B.tar (uncompressed tar, about 9 GB). The directory is named
# Qwen3.5-4B, so the run directories match ours (runs/<ARM>-<TAG>-Qwen3.5-4B-<data>-s<seed>).
# Checks (no sha256 of our HPC copy is at hand, so the Hub's own record of the pinned revision is the reference):
#   the Hub resolves the revision to itself; every file of that revision is on disk with the Hub's size; every LFS file
#   (the weights) has the Hub's sha256, every other file the Hub's git blob id; every file's download metadata names the
#   pinned commit. Then it writes Qwen3.5-4B.sha256 (sha256 and size of every file) next to the tar: send it to Sora.
set -euo pipefail
STAGING=${1:?usage: bash fetch_model_4b.sh /staging/<netid>}
SIF=${SIF:-cotd_4b.sif}
apptainer exec --env HF_HUB_OFFLINE=0 "$SIF" python - <<'PY'
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._local_folder import read_download_metadata
REPO, REV, DIR = "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "Qwen3.5-4B"
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
assert info.sha == REV, f"the Hub resolves {REV} to {info.sha}"
p = Path(snapshot_download(REPO, revision=REV, local_dir=DIR))
bad, lines = [], []
for s in sorted(info.siblings, key=lambda s: s.rfilename):
    f = p / s.rfilename
    if not f.is_file():
        bad.append(f"missing {s.rfilename}"); continue
    data_sha, blob, n = hashlib.sha256(), hashlib.sha1(), f.stat().st_size
    blob.update(b"blob %d\0" % n)
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            data_sha.update(b); blob.update(b)
    if s.size is not None and n != s.size:
        bad.append(f"size {s.rfilename}: {n} != {s.size}")
    if s.lfs is not None and data_sha.hexdigest() != s.lfs.sha256:
        bad.append(f"sha256 {s.rfilename}: {data_sha.hexdigest()} != {s.lfs.sha256}")
    if s.lfs is None and s.blob_id and blob.hexdigest() != s.blob_id:
        bad.append(f"blob id {s.rfilename}: {blob.hexdigest()} != {s.blob_id}")
    md = read_download_metadata(p, s.rfilename)
    if md is None or md.commit_hash != REV:
        bad.append(f"commit {s.rfilename}: {md.commit_hash if md else 'no metadata'}")
    lines.append(f"{data_sha.hexdigest()}  {n:>12}  {DIR}/{s.rfilename}")
extra = sorted(str(q.relative_to(p)) for q in p.rglob("*") if q.is_file() and ".cache" not in q.parts
               and str(q.relative_to(p)) not in {s.rfilename for s in info.siblings})
bad += [f"not in the revision: {e}" for e in extra]
assert not bad, "\n".join(bad)
assert (p / "config.json").is_file() and any(q.suffix == ".safetensors" for q in p.iterdir())
Path(f"{DIR}.sha256").write_text(f"# {REPO} @ {REV}\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
print(f"{DIR} ok at {REV}: {len(lines)} files")
PY
rm -rf Qwen3.5-4B/.cache
tar cf Qwen3.5-4B.tar Qwen3.5-4B
mv Qwen3.5-4B.tar Qwen3.5-4B.sha256 "$STAGING/"
ls -la "$STAGING/Qwen3.5-4B.tar" "$STAGING/Qwen3.5-4B.sha256"
