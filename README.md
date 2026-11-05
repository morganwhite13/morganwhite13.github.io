# morganwhite13.github.io

Source for [morganwhite13.github.io](https://morganwhite13.github.io), my portfolio site. It is a [Jekyll](https://jekyllrb.com/) site using the [minima](https://github.com/jekyll/minima) theme, built and hosted by GitHub Pages.

## Running locally

Requires Ruby 3.x and Bundler.

```sh
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000. The `github-pages` gem keeps local versions in line with what GitHub Pages uses.

To check for broken internal links:

```sh
bundle exec jekyll build
bundle exec htmlproofer _site --disable-external
```

To run the accessibility audit (axe-core, WCAG 2 A/AA, light and dark mode) on the built site:

```sh
npm ci
npx playwright install chromium
npm run check:a11y
```

Both checks run in CI on every push and pull request. A separate workflow checks external links weekly and on pull requests that change content.

The result charts on the project pages are generated SVGs. To change them, edit the numbers in `scripts/make_charts.py` and run:

```sh
python3 scripts/make_charts.py
```

Raster plots, such as the matplotlib output copied from the VMAS project, go in `assets/charts/` as PNGs. After adding or updating one, run this (it needs Pillow):

```sh
python3 scripts/prepare_plots.py
```

It quantizes each PNG to 256 colours (about a third of the size, with no visible change). It also gives every Markdown image that uses one, and every generated SVG chart, a width, height and lazy loading, so pages don't jump while they load. Wide plots (`.chart-wide`) also link to the full-size image, since multi-panel plots are hard to read at phone width.

## Layout

| Path | Contents |
| --- | --- |
| `index.md` | Home page |
| `about.md` | About page |
| `projects.md` | Project overview |
| `projects/*.md` | One detailed write-up per project |
| `_data/projects.yml` | Project list behind the home page cards, the projects page and the previous/next links |
| `_layouts/`, `_includes/` | Overrides of the minima theme (project layout, project cards, head tags) |
| `assets/` | Styles, resume PDF, icons, charts and the social preview image |
| `scripts/make_charts.py` | Generates the SVG charts in `assets/charts/` |
| `scripts/prepare_plots.py` | Shrinks the PNG plots in `assets/charts/` and sizes every chart image |
| `scripts/check-a11y.js` | Accessibility audit of the built site |
| `_config.yml` | Site settings, navigation and plugins |

## Adding a project

1. Create `projects/<slug>.md` with this front matter (the layout is set automatically):

   ```yaml
   ---
   title: Project Title
   permalink: /projects/<slug>/
   description: "One sentence for search results and link previews."
   ---
   ```

2. Add an entry to `_data/projects.yml` with the same `url`. Set `featured: true` to show it on the home page, and fill in `glance` (`problem`, `approach`, `result`) for the "At a glance" box at the top of the write-up.
