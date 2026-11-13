"""Explicit publication of one reviewed artifact directory; never uploads workspace wholesale."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi
p = argparse.ArgumentParser()
p.add_argument("--folder", required=True)
p.add_argument("--repo", required=True, help="namespace/repo")
p.add_argument("--type", choices=["model", "dataset"], required=True)
p.add_argument("--public", action="store_true", help="Required acknowledgement for public submission artifacts")
a = p.parse_args()
folder = Path(a.folder).resolve()
if not a.public:
    p.error("Inspect the artifact folder, then pass --public to publish")
if not folder.is_dir() or not (folder / "README.md").is_file():
    p.error("Artifact directory must exist and include a completed model/dataset card")
for file in folder.rglob("*"):
    if file.is_symlink() or file.name.startswith(".env") or ".git" in file.parts or ".venv" in file.parts:
        p.error(f"Remove symlinks/credentials/environment folders before publication: {file}")
api = HfApi()
api.create_repo(a.repo, repo_type=a.type, private=False, exist_ok=True)
api.update_repo_settings(a.repo, repo_type=a.type, private=False, gated=False)
api.upload_folder(folder_path=str(folder), repo_id=a.repo, repo_type=a.type,
                   commit_message="Publish FinSafe reviewed experiment artifacts")
print(f"https://huggingface.co/{'datasets/' if a.type == 'dataset' else ''}{a.repo}")
