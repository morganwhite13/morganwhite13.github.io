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

This also runs in CI on every push and pull request. A separate workflow checks external links weekly and on pull requests that change content.

The result charts on the project pages are generated SVGs. To change them, edit the numbers in `scripts/make_charts.py` and run:

```sh
python3 scripts/make_charts.py
```

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

2. Add an entry to `_data/projects.yml` with the same `url`. Set `featured: true` to show it on the home page.
