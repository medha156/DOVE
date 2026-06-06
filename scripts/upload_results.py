"""
DOVE — Upload results to git remote.

Stages results/ (and optionally extra folders), commits with a timestamp tag,
and pushes to origin/main.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("upload_results")

REPO_ROOT = Path(__file__).parent.parent

try:
    import git
    _GIT_AVAILABLE = True
except ImportError:
    _GIT_AVAILABLE = False
    logger.error("gitpython not installed — run: pip install gitpython")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-folder", action="append", default=[],
                        dest="extra_folders", help="Additional result folders to stage")
    args = parser.parse_args()

    if not _GIT_AVAILABLE:
        sys.exit(1)

    try:
        repo = git.Repo(REPO_ROOT)
    except git.InvalidGitRepositoryError:
        logger.error("Not a git repository: %s", REPO_ROOT)
        sys.exit(1)

    folders = ["results", "configs"] + args.extra_folders
    logger.info("Staging %s (excluding checkpoints)", ", ".join(folders))
    for folder in folders:
        folder_path = REPO_ROOT / folder
        if not folder_path.exists():
            continue
        for fpath in folder_path.rglob("*"):
            if fpath.is_file() and fpath.suffix.lower() not in (".pt", ".pth"):
                try:
                    repo.index.add([str(fpath)])
                except Exception as e:
                    logger.warning("Could not stage %s: %s", fpath, e)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit_msg = f"results: auto-upload at {timestamp}"
    if repo.index.diff("HEAD") or repo.untracked_files:
        commit = repo.index.commit(commit_msg)
        logger.info("Committed: %s — %s", commit.hexsha[:8], commit_msg)
    else:
        logger.info("Nothing to commit — working tree clean")
        return

    tag_name = f"results-{timestamp}"
    repo.create_tag(tag_name, message=commit_msg)
    logger.info("Created tag: %s", tag_name)

    import subprocess, os
    origin = repo.remote("origin")
    logger.info("Pushing to origin main with tags…")
    try:
        origin.push(refspec="main:main")
        origin.push(tags=True)
    except Exception as e:
        logger.warning("gitpython push failed (%s), retrying with git CLI", e)
        env = dict(os.environ)
        subprocess.run(["git", "push", "origin", "main", "--tags"],
                       cwd=str(REPO_ROOT), check=True, env=env)
    logger.info("Push complete.")


if __name__ == "__main__":
    main()
