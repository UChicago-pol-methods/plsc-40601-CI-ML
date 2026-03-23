#!/usr/bin/env bash

set -euo pipefail

quarto render site
rm -rf docs
mkdir docs
cp -R site/_site/. docs/
touch docs/.nojekyll
find docs -name .DS_Store -delete
