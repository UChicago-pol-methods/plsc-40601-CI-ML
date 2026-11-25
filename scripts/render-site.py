#!/usr/bin/env python3
"""Render and validate the Quarto website before replacing docs/. Never deploy."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from published_projects import public_manifest


ROOT = Path(__file__).resolve().parent.parent


def main():
    projects = public_manifest()["projects"]
    candidates = [os.environ.get("QUARTO"), shutil.which("quarto"),
                  "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto"]
    quarto = next((str(p) for p in candidates if p and Path(p).is_file()), None)
    if not quarto:
        raise SystemExit("Quarto was not found. Install it or set QUARTO to its executable path.")
    if len(sys.argv) > 1:
        raise SystemExit("Usage: bash scripts/render-site.sh (set QUARTO to override the executable)")
    built = ROOT / "site/_site"
    # Quarto can retain orphaned resource copies; never carry removed downloads forward.
    if (built / "projects").exists():
        shutil.rmtree(built / "projects")
    subprocess.run([quarto, "render", "site"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/check-project-preview.py"),
                    "--directory", str(built)], check=True)
    # Verify prepared tutorials and display assets survived the resource copy.
    public_assets = [ROOT / "site/projects/2026/index.html"]
    for directory in [ROOT / "site/projects/shared"] + [
            ROOT / "site/projects/2026" / project["slug"] for project in projects]:
        public_assets.extend(directory.rglob("*"))
    for source in public_assets:
        if not source.is_file() or any(part.startswith(".") for part in source.relative_to(ROOT / "site").parts):
            continue
        target = built / source.relative_to(ROOT / "site")
        if not target.is_file() or source.read_bytes() != target.read_bytes():
            raise SystemExit(f"Prepared project asset changed or is missing: {source.relative_to(ROOT)}")
    with tempfile.TemporaryDirectory(prefix=".site-build-", dir=ROOT) as scratch:
        stage = Path(scratch) / "new"
        shutil.copytree(built, stage, ignore=shutil.ignore_patterns(".DS_Store"))
        old = Path(scratch) / "previous"
        docs = ROOT / "docs"
        if docs.exists():
            docs.rename(old)
        try:
            stage.rename(docs)
        except BaseException:
            if old.exists():
                old.rename(docs)
            raise
    shutil.rmtree(built)
    print(f"Rebuilt docs/ from site/. {len(projects)} approved project(s) and their assets are intact. Nothing was deployed.")


if __name__ == "__main__":
    main()
