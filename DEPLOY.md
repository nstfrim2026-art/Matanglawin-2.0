# Deploying Matanglawin to a live public URL

The site is a Flask + PyTorch app, so it needs a host that can run a Python
web server. The recommended free host is **Hugging Face Spaces**, which is
built for ML demos and has enough memory for PyTorch (the tiny free tier that
some other hosts offer will run out of memory).

The repo already contains a `Dockerfile` that Hugging Face can run directly.

---

## Option A — Hugging Face Spaces (recommended, free, ~5 minutes)

You only need a free Hugging Face account. No credit card, no servers to manage.

### 1. Create the account
Sign up at https://huggingface.co/join

### 2. Create a new Space
- Go to https://huggingface.co/new-space
- **Space name:** e.g. `matanglawin`
- **License:** your choice
- **Select the SDK:** choose **Docker** → **Blank**
- **Hardware:** the free **CPU basic** is fine
- Click **Create Space**

### 3. Add the project files to the Space
A Hugging Face Space is itself a git repo. The simplest way to fill it:

**Easiest (web upload):**
- On your new Space page, click **Files** → **Add file** → **Upload files**
- Upload everything from this project:
  `Dockerfile`, `app.py`, `inference_core.py`, `requirements.txt`,
  `best.pt`, and the `templates/` and `static/` folders.
- Commit the changes.

**Or via git:**
```bash
git clone https://huggingface.co/spaces/<your-username>/matanglawin
cd matanglawin
# copy the project files into this folder (Dockerfile, app.py, best.pt, etc.)
git add .
git commit -m "Add Matanglawin crack-detection app"
git push
```

### 4. Confirm the Space config
Hugging Face reads a small YAML header at the top of the Space's `README.md`.
Make sure it looks like this (create/edit `README.md` in the Space if needed):

```yaml
---
title: Matanglawin
emoji: 👁️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---
```

The important lines are `sdk: docker` and `app_port: 7860` (the port the
app listens on).

### 5. Wait for the build
The Space will show "Building" while it builds the Docker image (a few
minutes the first time). When it flips to "Running", your site is live at:

```
https://<your-username>-matanglawin.hf.space
```

Share that link with anyone. It goes to sleep after inactivity and wakes
automatically on the next visit.

---

## Option B — Any Docker host (Render, Fly.io, a VPS, etc.)

Because the app is containerized, it runs anywhere Docker runs:

```bash
docker build -t matanglawin .
docker run -p 7860:7860 matanglawin
# open http://localhost:7860
```

Point your host of choice at this `Dockerfile`. Note the app reads the
`PORT` and `HOST` environment variables, so if a platform assigns a
different port, set `PORT` accordingly (it defaults to 7860 in the image,
and binds to `0.0.0.0`).

> Heads-up on memory: this app loads PyTorch, which needs roughly
> ~1 GB RAM. Free tiers with 512 MB (e.g. Render free) will crash on
> startup. Hugging Face Spaces' free CPU tier has plenty.
