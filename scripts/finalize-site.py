#!/usr/bin/env python3
"""Index the preserved student tutorials after Quarto renders the course pages."""

import json
import os
from pathlib import Path
from published_projects import public_manifest

ROOT = Path(__file__).resolve().parent.parent


def main():
    output = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
    if not output.is_absolute():
        output = ROOT / "site" / output
    search_path = output / "search.json"
    search = json.loads(search_path.read_text())
    search = [entry for entry in search if not entry.get("href", "").startswith("projects/2026/")]
    manifest = public_manifest()
    projects = manifest["projects"]
    project_root = output / "projects/2026"
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for project in projects:
        href = f'projects/2026/{project["slug"]}/tutorial.html'
        search.append({"objectID": href, "href": href, "title": project["title"],
                       "section": "", "text": project["author"] + ". " + project["title"]})
    search_path.write_text(json.dumps(search, ensure_ascii=False, indent=2) + "\n")
    (output / ".nojekyll").touch()


if __name__ == "__main__":
    main()
