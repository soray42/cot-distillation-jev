#!/bin/bash
# Inside the interactive build job of batch 4, with the batch-3 image cotd_4b.sif in the working directory:
#   bash fetch_model_9b.sh /staging/<netid> [REPO REVISION DIR]
# Default: Qwen/Qwen3.5-9B (the post-trained release, Apache-2.0, not gated, 4 safetensors shards, about 19.3 GB) at the
# pinned revision below, written as /staging/<netid>/Qwen3.5-9B.tar (uncompressed). The directory is named Qwen3.5-9B,
# so the run directories read runs/<ARM>-<TAG>-Qwen3.5-9B-<data>-s<seed>. fetch_model_4b.sh generalised (same checks;
# `bash fetch_model_9b.sh /staging/<netid> Qwen/Qwen3.5-4B 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a Qwen3.5-4B`
# reproduces it). Checks, with the Hub's own record of the pinned revision as the reference:
#   the Hub resolves the revision to itself; every file of that revision is on disk with the Hub's size; every LFS file
#   (the weights) has the Hub's sha256, every other file the Hub's git blob id; every file's download metadata names the
#   pinned commit; nothing else is in the directory; with several shards, model.safetensors.index.json is there and
#   every shard it names is on disk. Then it writes <DIR>.sha256 (sha256 and size of every file) next to the tar: send
#   it to Sora.
set -euo pipefail
STAGING=${1:?usage: bash fetch_model_9b.sh /staging/<netid> [REPO REVISION DIR]}
REPO=${2:-Qwen/Qwen3.5-9B}; REV=${3:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}; DIR=${4:-Qwen3.5-9B}
SIF=${SIF:-cotd_4b.sif}
apptainer exec --env HF_HUB_OFFLINE=0 "$SIF" python - "$REPO" "$REV" "$DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub._local_folder import read_download_metadata
REPO, REV, DIR = sys.argv[1:4]
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
shards = sorted(q.name for q in p.iterdir() if q.suffix == ".safetensors")
if len(shards) > 1:
    idx = p / "model.safetensors.index.json"
    if not idx.is_file():
        bad.append(f"{len(shards)} shards but no model.safetensors.index.json")
    else:
        named = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        bad += [f"index names {q}, not on disk" for q in named if q not in shards]
        bad += [f"shard {q} not in the index" for q in shards if q not in named]
assert not bad, "\n".join(bad)
assert (p / "config.json").is_file() and shards
Path(f"{DIR}.sha256").write_text(f"# {REPO} @ {REV}\n" + "\n".join(lines) + "\n")
print("\n".join(lines))
print(f"{DIR} ok at {REV}: {len(lines)} files, {len(shards)} safetensors shard(s), "
      f"{sum((p / s).stat().st_size for s in shards) / 1e9:.1f} GB of weights")
PY
rm -rf "$DIR/.cache"
tar cf "$DIR.tar" "$DIR"
mv "$DIR.tar" "$DIR.sha256" "$STAGING/"
ls -la "$STAGING/$DIR.tar" "$STAGING/$DIR.sha256"
