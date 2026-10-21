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
bundle exec htmlproofer _site --disable-external --ignore-missing-alt
```

This also runs in CI on every push and pull request.

## Layout

| Path | Contents |
| --- | --- |
| `index.md` | Home page |
| `about.md` | About page |
| `projects.md` | Project overview |
| `projects/*.md` | One detailed write-up per project |
| `assets/` | Resume PDF and other static files |
| `_config.yml` | Site settings, navigation and plugins |

## Adding a project

1. Create `projects/<slug>.md` with this front matter:

   ```yaml
   ---
   layout: page
   title: Project Title
   permalink: /projects/<slug>/
   ---
   ```

2. Add a summary section to `projects.md` linking to `/projects/<slug>/`.
3. Optionally feature it on the home page in `index.md`.
