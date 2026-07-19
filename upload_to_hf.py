"""
Upload AvatarMoE trained checkpoints to the Hugging Face Hub.

Prereqs:
    pip install -U huggingface_hub
    hf auth login          # paste a WRITE token from https://huggingface.co/settings/tokens

Local layout (as produced by train.py):

    exp/
    ├── ps_female_3-best/
    │   ├── .hydra/            # resolved Hydra config
    │   ├── test-pose/         # rendered eval outputs  (NOT uploaded)
    │   ├── ckpt15000.pth      # checkpoint
    │   └── render.log         # (NOT uploaded)
    ├── zju_377_mono-best/
    │   └── ... (may contain hundreds of render images -- NOT uploaded)
    └── ...

Uses an ALLOW-list, so ONLY the checkpoint (and the .hydra config) go up,
regardless of what render folders exist inside each subject dir.
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ----------------------------------------------------------------------------
# Config -- edit these
# ----------------------------------------------------------------------------
REPO_ID      = "CODINGHYE/AvatarMoE"
EXP_ROOT     = Path("exp")
PRIVATE      = False
SUBJECT_GLOB = "*-best"

# ONLY these are uploaded. Drop ".hydra/*" if you want the .pth alone.
ALLOW_PATTERNS = ["ckpt*.pth", ".hydra/*"]

# Wipe checkpoints/ on the Hub first, so previously-uploaded junk (render
# images) is removed and the repo ends up clean. Set False to skip.
CLEAN_FIRST = True
# ----------------------------------------------------------------------------


def main():
    api = HfApi()

    create_repo(REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
    print(f"[ok] repo ready: https://huggingface.co/{REPO_ID}")

    # 1) Optional clean slate: delete the whole checkpoints/ folder on the Hub
    if CLEAN_FIRST:
        try:
            api.delete_folder(
                path_in_repo="checkpoints",
                repo_id=REPO_ID,
                repo_type="model",
                commit_message="Clean checkpoints/ before re-upload",
            )
            print("[ok] removed existing checkpoints/ on the Hub")
        except Exception as e:
            print(f"[skip] nothing to clean ({type(e).__name__})")

    # 2) Collect subject folders
    subject_dirs = sorted(p for p in EXP_ROOT.glob(SUBJECT_GLOB) if p.is_dir())
    if not subject_dirs:
        raise SystemExit(f"No '{SUBJECT_GLOB}' folders found under {EXP_ROOT.resolve()}")

    # 3) Upload only the allowed files, mirroring folder names
    for d in subject_dirs:
        path_in_repo = f"checkpoints/{d.name}"
        print(f"[up]  {d}  ->  {path_in_repo}")
        api.upload_folder(
            folder_path=str(d),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=ALLOW_PATTERNS,
            commit_message=f"Add checkpoint: {d.name}",
        )

    # 4) Upload the model card if present next to this script
    card = Path("README.md")
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
        print("[ok] model card uploaded")

    print(f"\nDone: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()