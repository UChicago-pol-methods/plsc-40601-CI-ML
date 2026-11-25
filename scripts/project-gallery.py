#!/usr/bin/env python3
"""Generate the Quarto gallery include from the prepared project manifest."""

import html
from pathlib import Path
from published_projects import public_manifest

ROOT = Path(__file__).resolve().parent.parent


def build():
    projects = public_manifest()["projects"]
    cards = []
    for project in projects:
        href = f'projects/2026/{project["slug"]}/tutorial.html'
        cards.append(
            '<li><a class="project-title" href="' + html.escape(href, quote=True) + '">'
            + html.escape(project["title"]) + '</a><span class="project-byline">'
            + html.escape(project["author"]) + '</span></li>'
        )
    gallery = (
        '<section id="project-gallery"><h2>Student projects · 2026</h2>'
        '<p class="project-intro">Tutorials on causal inference and machine learning, '
        'written by students in PLSC 40601 and shared with their permission.</p><ul class="project-list">'
        + "\n".join(cards) + '</ul></section>'
    )
    (ROOT / "site/_project-gallery.qmd").write_text("```{=html}\n" + gallery + "\n```\n")


if __name__ == "__main__":
    build()
