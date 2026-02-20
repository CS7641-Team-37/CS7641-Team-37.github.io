# CS7641-Team-37.github.io
Repository for CS7641 Team 37 (stock and polymarket prediction)

Please navigate to the following sites for further information:
- [Project Proposal](Project-Proposal/)
- [Midterm Presentation](Midterm-Presentation/)
- [Final Presentation](Final-Presentation/)

## Run the site locally

### 1) Prerequisites
- Ruby (recommended: latest stable 3.x)
- Bundler (`gem install bundler`)

### 2) Install dependencies
From the repository root:

```bash
bundle install
```

### 3) Build once (no server)

```bash
bundle exec jekyll clean
bundle exec jekyll build
```

Generated site files will be in `_site/`.

### 4) Start local preview server

```bash
bundle exec jekyll serve
```

Then open: `http://127.0.0.1:4000`

### 5) Optional: live reload while editing

```bash
bundle exec jekyll serve --livereload
```

If port `4000` is busy, run on a different port:

```bash
bundle exec jekyll serve --port 4002
```

Open: `http://127.0.0.1:4002`

## Troubleshooting

- **Could not locate Gemfile**
	- Make sure you are in this repo root before running bundle/jekyll commands.

- **no acceptor (port is in use or requires root privileges)**
	- Use another port, e.g. `bundle exec jekyll serve --port 4002`.

- **Faraday / GitHub metadata warnings**
	- These warnings are non-blocking for local preview and can usually be ignored.

