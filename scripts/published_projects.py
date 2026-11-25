"""The public project list is independent of private sources and review records."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
PUBLICATION = ROOT / "site/published-projects.json"


def public_manifest(path=PUBLICATION):
    """Require explicit consent and return only public display metadata."""
    manifest = json.loads(path.read_text())
    if manifest.get("year") != 2026 or not isinstance(manifest.get("projects"), list):
        raise ValueError("Expected a 2026 publication list")
    projects, slugs = [], set()
    for project in manifest["projects"]:
        if not isinstance(project, dict):
            raise ValueError("Invalid publication record")
        slug = project.get("slug", "")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in slugs:
            raise ValueError("Invalid or duplicate publication slug")
        if project.get("publication_approved") is not True:
            raise ValueError(f"Publication consent is required: {slug}")
        if any(not isinstance(project.get(key), str) or not project[key].strip()
               for key in ("author", "title")):
            raise ValueError(f"Missing public display metadata: {slug}")
        slugs.add(slug)
        projects.append({key: project[key] for key in ("slug", "author", "title")}
                        | {"has_source": False, "has_pdf": False})
    return {"year": 2026, "projects": projects}
