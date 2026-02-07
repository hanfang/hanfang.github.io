# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic portfolio website built with **Jekyll** (static site generator) and hosted on **GitHub Pages**. It's a customized version of the Academic Pages template designed for researchers to showcase publications, talks, and professional achievements. The site is deployed at https://hanfang.info.

## Development Commands

### Local Development

```bash
# Install Ruby dependencies
bundle install

# Serve locally with live reload (RESTART required after _config.yml changes)
jekyll serve -l -H localhost
# OR
bundle exec jekyll serve -l -H localhost

# Access at http://localhost:4000
```

### Docker Development (Alternative)

```bash
chmod -R 777 .
docker compose up
# Access at http://localhost:4000
```

### Content Generation

```bash
# Generate publication markdown files from TSV
cd markdown_generator
python publications.py
# OR use Jupyter notebook: publications.ipynb

# Generate talk markdown files from TSV
python talks.py
# OR use Jupyter notebook: talks.ipynb

# Convert CV markdown to JSON format
cd scripts
python cv_markdown_to_json.py
```

### Frontend Build (if modifying JavaScript)

```bash
# Install npm dependencies
npm install

# Minify JavaScript
npm run uglify

# Watch for JS changes
npm run watch:js
```

## Architecture

### Content Structure

The site uses **Jekyll Collections** to organize different content types:

- **`_pages/`**: Static pages (about, publications list, talks list, CV, blogs)
  - `about.md` serves as the homepage (`/`)
  - All pages use YAML frontmatter with `permalink` to define URLs

- **`_publications/`**: Individual publication entries (Jekyll collection)
  - Files named: `YYYY-MM-DD-slug.md`
  - Displayed on `/publications/` page
  - Categories: "manuscripts" (Journal Articles) and "conferences" (Conference Papers)

- **`_talks/`**: Conference presentations and talks (Jekyll collection)
  - Integrated with geographic talkmap visualization

- **`_posts/`**: Blog posts (standard Jekyll posts)
  - Files named: `YYYY-MM-DD-title.md`
  - Currently minimal content

- **`_teaching/`**, **`_portfolio/`**: Additional collections (lightly used)

### Configuration Files

- **`_config.yml`**: Master site configuration
  - Site metadata (title, URL, repository)
  - Author profile (bio, avatar, social links)
  - Collection definitions and default layouts
  - **Important**: Changes require server restart

- **`_data/navigation.yml`**: Controls header navigation menu
  - Easy to modify without touching templates

- **`_data/cv.json`**: Structured CV data for JSON-based CV page

### Templates & Styling

- **`_layouts/`**: Page templates (default.html, single.html, archive.html, talk.html, cv-layout.html)
- **`_includes/`**: Reusable HTML components (author-profile.html, archive-single.html, feature_row, etc.)
- **`_sass/`**: SCSS stylesheets (auto-compiled to `assets/css/`)
- **`assets/js/`**: JavaScript (jQuery-based, minified to main.min.js)

### Static Assets

- **`files/`**: Downloadable files (PDFs, CV, etc.) → accessible at `/files/filename.pdf`
- **`images/`**: Image assets (including profile photo)

## Adding Content

### Adding a Publication

**Option 1: Manual** (for single additions)
```bash
# Create file: _publications/YYYY-MM-DD-short-title.md
# Use this frontmatter template:
---
title: "Paper Title"
collection: publications
category: conferences  # or "manuscripts"
permalink: /publication/YYYY-MM-DD-short-title
excerpt: 'Brief description'
date: YYYY-MM-DD
venue: 'Conference/Journal Name'
paperurl: 'https://arxiv.org/abs/...'
citation: 'Authors. (Year). "Title." <i>Venue</i>.'
---

Optional paper description here.
```

**Option 2: Bulk** (using TSV conversion)
1. Edit `markdown_generator/publications.tsv`
2. Run `python markdown_generator/publications.py`
3. Generated files appear in `_publications/`

### Adding to Homepage "Recent Papers" Section

Edit `_pages/about.md` and add to the "Recent Papers" section following this format:

```markdown
* **Paper Title**
  *Authors (bold **Han Fang**)* · [**Venue**](arxiv-link) (Year)
```

Add new papers at the top of the list (most recent first).

### Adding a Blog Post

```bash
# Create file: _posts/YYYY-MM-DD-title.md
---
title: 'Post Title'
date: YYYY-MM-DD
permalink: /posts/YYYY/MM/title/
tags:
  - tag1
  - tag2
---

Post content here.
```

### Adding a Talk

Similar to publications, either:
1. Create `_talks/YYYY-MM-DD-title.md` manually
2. Edit `markdown_generator/talks.tsv` and run `talks.py`

## Git Workflow

### Committing Changes

```bash
git add <files>
git commit -m "Description of changes"
git push origin master
```

GitHub Pages automatically rebuilds and deploys the site after push to master.

### Deployment

- **Branch**: master
- **Domain**: hanfang.info (via CNAME file)
- **Build**: Automatic via GitHub Pages
- **Status**: Check repository Settings → Pages

## Key Technical Details

- **Jekyll Version**: 3.9+
- **Ruby Version**: 3.2+ (see Dockerfile)
- **Markdown Processor**: kramdown with GitHub Flavored Markdown (GFM)
- **Code Highlighting**: Rouge syntax highlighter
- **Theme Base**: Minimal Mistakes (customized, detached fork)

## Special Features

### Talkmap
- Geographic visualization of talk locations
- Generated via `talkmap/getorg.py` and `talkmap.ipynb`
- GitHub Actions workflow (`.github/workflows/scrape_talks.yml`) auto-updates on push

### Multi-format CV
- **Markdown version**: `_pages/cv.md`
- **JSON version**: `_pages/cv-json.md` (uses `_data/cv.json`)
- Conversion script: `scripts/cv_markdown_to_json.py`

### Publication Categories
- Defined in `_config.yml` under `publication_category`
- Current categories: "manuscripts" (Journal Articles) and "conferences" (Conference Papers)
- Display controlled in `_pages/publications.html`

## Common Gotchas

1. **_config.yml changes require server restart** - Live reload doesn't pick up config changes
2. **File naming matters** - Publications/talks must follow `YYYY-MM-DD-slug.md` format
3. **Gemfile.lock conflicts** - If bundle install fails, delete Gemfile.lock and retry
4. **Permissions on Docker** - Run `chmod -R 777 .` before `docker compose up`
5. **Navigation menu** - Modify `_data/navigation.yml`, not templates directly
6. **Homepage content** - Edit `_pages/about.md` for Recent Papers and News sections

## Development Environment Options

1. **Native Ruby/Jekyll** - Recommended for quick edits
2. **Docker** - Best for avoiding dependency issues
3. **VS Code DevContainer** - Automatic setup, open with F1 → "DevContainer: Reopen in Container"
