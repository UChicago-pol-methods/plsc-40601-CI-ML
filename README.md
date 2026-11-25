# PLSC 40601 — Advanced Topics in Causal Inference (Spring 2026)

Repository for the Spring 2026 syllabus and public course materials for `PLSC 40601` at the University of Chicago.

[Course website](https://uchicago-pol-methods.github.io/plsc-40601-CI-ML/)

Readings are distributed privately and are intentionally not stored in this public repository.

## Repository structure

- `assets/`: shared bibliography, LaTeX style files, and figure assets
- `site/`: editable Quarto course pages and approved tutorial editions
- `docs/`: generated GitHub Pages website; rebuild from `site/`
- `slides/`: lecture deck source files and slide-authoring guidance
- `materials/`: standalone worked examples and method notes
- `assignments/`: assignment and project prompts
- `code/`: in-class and section code
- `reference/`: supplemental reference notes and handouts
- `syllabus_PLSC40601-2026.tex`: current syllabus source

Historical 2024 materials remain in the repository while the 2026 refresh is underway.

## Rebuild and preview the website

Edit the course pages in `site/`, then run:

```sh
bash scripts/render-site.sh
python3 -m http.server --bind 127.0.0.1 --directory docs 40601
```

Open <http://127.0.0.1:40601/>. The build uses Quarto on PATH or RStudio's
bundled copy; set `QUARTO` to select another executable. Python 3.9 or later is
required. It checks publication consent, links, search entries, and assets
before replacing `docs/`. It does not run student analyses or deploy.
The build works from a clean checkout without private student materials.

## Student publication permission

The explicit list of tutorials approved for publication is
`site/published-projects.json`; every entry must have `publication_approved: true`.
The gallery, search index, and generated project manifest use this list.
`site/_quarto.yml` copies only the listed public tutorial directories and shared
display assets. The website checker rejects extra projects and source archives.

After an author explicitly agrees to publication:

1. Add their public metadata and approval to `site/published-projects.json`.
2. Add their prepared tutorial directory to `project.resources` in `site/_quarto.yml`.
3. Add matching exceptions in `.gitignore` for `site/projects/2026/<slug>/`
   and `docs/projects/2026/<slug>/`.
4. Rebuild, run `python3 scripts/test-publication.py`, and review the staged files.

Commit the approved prepared tutorials, website sources, and regenerated `docs/`
together. Building and committing do not push or publish the website.

Maintained authoring sources in `projects/`, private archives and review packages
in `local/`, the full local `site/projects/2026/manifest.json`, and unpublished
prepared tutorial directories are ignored by Git. These files stay on the local
machine. The public build does not read the full local manifest or use the private
preparation/review tools. Keep consent correspondence and internal notes local.

The website contains rendered tutorials and display assets. Source ZIPs, original
PDFs, build instructions, and review reports are excluded. Tutorials using MathJax
share one versioned renderer in `site/projects/shared/`.
