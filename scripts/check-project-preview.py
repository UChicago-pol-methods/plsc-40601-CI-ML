#!/usr/bin/env python3
"""Check course links and assets, and keep authoring files and internal notes private."""

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
from published_projects import public_manifest

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = Path("projects/2026")
SHARED_ASSETS = Path("projects/shared/mathjax-3.2.2")
MATH_SCRIPT_URL = "../../shared/mathjax-3.2.2/tex-svg-full.js"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.ids = [], set()
        self.gallery_refs = []
        self.gallery_depth = 0
        self.course_navigation_tags = []
        self.course_navigation_refs = []
        self.course_navigation_depth = 0
        self.redirects = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "section":
            if self.gallery_depth:
                self.gallery_depth += 1
            elif values.get("id") == "project-gallery":
                self.gallery_depth = 1
        if "id" in values:
            self.ids.add(values["id"])
        if values.get("id") == "course-project-navigation":
            self.course_navigation_tags.append(tag)
        if tag == "nav":
            if self.course_navigation_depth:
                self.course_navigation_depth += 1
            elif values.get("id") == "course-project-navigation":
                self.course_navigation_depth = 1
        if tag == "a" and "name" in values:
            self.ids.add(values["name"])
        if tag == "a" and self.gallery_depth and values.get("href"):
            self.gallery_refs.append(values["href"])
        if tag == "a" and self.course_navigation_depth and values.get("href"):
            self.course_navigation_refs.append(values["href"])
        if tag == "meta" and values.get("http-equiv", "").lower() == "refresh":
            self.redirects.append(values.get("content", ""))
        for attribute in ("href", "src", "data", "data-src"):
            if values.get(attribute) is not None and (attribute != "data" or tag == "object"):
                self.refs.append(values[attribute])

    def handle_endtag(self, tag):
        if tag == "section" and self.gallery_depth:
            self.gallery_depth -= 1
        if tag == "nav" and self.course_navigation_depth:
            self.course_navigation_depth -= 1


def local_target(web, page, reference):
    url = urlsplit(reference)
    if url.scheme or url.netloc:
        return url, None
    path = unquote(url.path)
    if not path:
        target = page
    elif path.startswith("/"):
        target = web / path.lstrip("/")
    else:
        target = page.parent / path
    return url, target.resolve()


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--directory", type=Path, default=ROOT / "docs",
                           help="Website directory to check (default: repository docs/)")
    arguments.add_argument("--report", type=Path,
                           help="Optionally save the JSON report at this path")
    options = arguments.parse_args()
    web = options.directory.resolve()
    approved = public_manifest()
    pages, problems = {}, []
    for page in web.rglob("*.html"):
        parser = References()
        parser.feed(page.read_text(errors="replace"))
        pages[page.resolve()] = parser
    for page, parser in pages.items():
        for reference in parser.refs:
            url, target = local_target(web, page, reference)
            if target is None:
                continue
            if "local" in Path(unquote(url.path)).parts:
                problems.append([str(page.relative_to(web)), "private local/ reference", reference])
            if not target.is_relative_to(web) or not target.exists():
                problems.append([str(page.relative_to(web)), "missing local reference", reference])
            elif url.fragment and target in pages and page.is_relative_to(web / PROJECTS):
                if unquote(url.fragment) not in pages[target].ids:
                    problems.append([str(page.relative_to(web)), "missing section anchor", reference])
    for css in web.rglob("*.css"):
        for reference in re.findall(r'url\([\s"\']*([^\)"\']+)', css.read_text(errors="replace")):
            url, target = local_target(web, css, reference.strip())
            if target is None or not url.path:
                continue
            if "local" in Path(unquote(url.path)).parts:
                problems.append([str(css.relative_to(web)), "private local/ reference", reference])
            if not target.is_relative_to(web) or not target.exists():
                problems.append([str(css.relative_to(web)), "missing CSS asset", reference])
    try:
        manifest = json.loads((web / PROJECTS / "manifest.json").read_text())
        projects = manifest["projects"]
        if not isinstance(projects, list):
            raise ValueError("projects must be a list")
        if manifest != approved:
            problems.append(["manifest.json", "website manifest differs from approved publication list"])
    except (OSError, ValueError, KeyError) as error:
        projects = []
        problems.append([str(PROJECTS / "manifest.json"), "project manifest unavailable", str(error)])
    gallery = pages.get(web / "projects.html")
    gallery_targets = []
    if gallery is None:
        problems.append(["projects.html", "missing gallery page"])
    else:
        gallery_targets = [local_target(web, web / "projects.html", reference)[1]
                           for reference in gallery.gallery_refs]
    bundles = 0
    slugs = set()
    expected_gallery_targets = set()
    for project in projects:
        if not isinstance(project, dict) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project.get("slug", "")):
            problems.append(["manifest.json", "invalid project slug", str(project)])
            continue
        slug = project["slug"]
        if slug in slugs:
            problems.append([slug, "duplicate project slug"])
        slugs.add(slug)
        directory = web / PROJECTS / slug
        expected_gallery_targets.add(directory / "tutorial.html")
        if gallery_targets.count(directory / "tutorial.html") != 1:
            problems.append([slug, "gallery must link directly to tutorial exactly once"])
        if not (directory / "index.html").is_file():
            problems.append([slug, "missing legacy project redirect"])
        else:
            redirect = pages[directory / "index.html"]
            if (len(redirect.redirects) != 1 or not re.fullmatch(
                    r"\s*0\s*;\s*url\s*=\s*['\"]?tutorial\.html['\"]?\s*",
                    redirect.redirects[0], re.IGNORECASE)):
                problems.append([slug, "legacy project page must redirect to tutorial.html"])
            if any(urlsplit(ref).path in {"tutorial.pdf", "source.zip"} for ref in redirect.refs):
                problems.append([slug, "legacy project page retains download menu"])
        if not (directory / "tutorial.html").is_file():
            problems.append([slug, "missing HTML tutorial"])
        else:
            tutorial = pages[directory / "tutorial.html"]
            html_text = (directory / "tutorial.html").read_text()
            if re.search(r'course-edition-note|class="(?:reproduction-note|conversion-note)"', html_text):
                problems.append([slug, "internal build/review commentary remains"])
            if re.search(r"mathjax\.rstudio\.com|mathjax/2\.", html_text):
                problems.append([slug, "legacy MathJax loader can lose script-letter glyphs"])
            if 'id="course-math-renderer"' in html_text:
                if MATH_SCRIPT_URL not in tutorial.refs:
                    problems.append([slug, "missing bundled SVG math renderer"])
            if tutorial.course_navigation_tags != ["nav"]:
                problems.append([slug, "tutorial must have exactly one course navigation element"])
            expected_navigation = {"../../../projects.html#project-gallery", "../../../index.html"}
            if (len(tutorial.course_navigation_refs) != 2 or
                    set(tutorial.course_navigation_refs) != expected_navigation):
                problems.append([slug, "tutorial must link back to Projects and course home"])
        if project.get("has_pdf"):
            problems.append([slug, "original PDFs belong in private review materials"])
        if project.get("has_source"):
            problems.append([slug, "public source downloads must be disabled"])
    for target in gallery_targets:
        if target not in expected_gallery_targets:
            problems.append(["projects.html", "unexpected gallery link", str(target)])
    project_root = web / PROJECTS
    if project_root.is_dir():
        for directory in project_root.iterdir():
            if directory.is_dir() and directory.name not in slugs:
                problems.append([str(directory.relative_to(web)), "project absent from manifest"])
        for path in (web / "projects").rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".pdf", ".zip", ".txt", ".md", ".r", ".rmd", ".qmd", ".ipynb", ".py"}:
                problems.append([str(path.relative_to(web)), "source archive or internal authoring/review file"])
    search = json.loads((web / "search.json").read_text())
    expected_hrefs = {f'projects/2026/{p["slug"]}/tutorial.html' for p in approved["projects"]}
    project_hrefs = [entry.get("href", "") for entry in search
                     if entry.get("href", "").startswith("projects/2026/")]
    if set(project_hrefs) != expected_hrefs or len(project_hrefs) != len(expected_hrefs):
        problems.append(["search.json", "search entries differ from approved publication list"])
    if any(MATH_SCRIPT_URL in parser.refs for parser in pages.values()):
        for name in ("tex-svg-full.js", "LICENSE"):
            asset = web / SHARED_ASSETS / name
            if not asset.is_file() or asset.read_bytes() != (ROOT / "site" / SHARED_ASSETS / name).read_bytes():
                problems.append([str(SHARED_ASSETS / name), "missing or changed shared MathJax asset"])
        if set((web / "projects").rglob("tex-svg-full.js")) != {web / SHARED_ASSETS / "tex-svg-full.js"}:
            problems.append(["projects", "expected exactly one shared MathJax renderer"])
    for path in web.rglob("*"):
        if path.is_file() and re.search(r"grades-and-notes|grading\.csv|AdvancedTopicsinCausalInfe|_\d{6}_\d{8}_", path.name):
            problems.append([str(path.relative_to(web)), "private/export identifier filename"])
    report = {"html_pages": len(pages), "projects": len(projects),
              "source_bundles": bundles, "problems": problems}
    if options.report:
        options.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
