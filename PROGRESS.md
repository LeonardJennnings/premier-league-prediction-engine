# Progress

## 2026-11-10 — Phase 0, steps 0.2–0.9

**Learned:** The terminal runs commands in a current folder (`pwd`, `cd`, `cd ..`). `python3` is the Python interpreter. Git records snapshots on this computer; GitHub is the website copy. `command not found` means the terminal could not find that program on the PATH. Cursor's open folder is the one the editor and this chat can see; the files also live on disk.

**Works now:** Python 3.14.10, Git 2.56.0, Git name/email/`main`, `SYLLABUS.md`, `hello.py`.

**Still shaky:** The difference between a folder on disk and the folder Cursor has open. It took a second explanation.

**Next:** Phase 1, step 1.1 — look at the project folder tree before creating files.

## 2026-11-10 — Phase 1, steps 1.1–1.6

**Learned:** File vs module vs package. Config is the shared rulebook. Imports pull names from another module. `python3 src/pipeline.py` can fail with ModuleNotFoundError; `python3 -m src.pipeline` runs it as a package module. Dashboard should import cleaning, not copy it.

**Works now:** Project skeleton folders/files, `src/config.py` paths, `python3 -m src.pipeline` prints ROOT, DATA_RAW, SEASON_CODES.

**Still shaky:** Exact wording of module vs package; import path vs file path.

**Next:** Phase 2 — Git and GitHub fundamentals.

## 2026-11-13 — Phase 2, Git and GitHub

**Learned:** `git add` stages; `git commit` saves locally; `git push` sends to GitHub. A branch is an alternate timeline. A PR asks to merge one branch into another. `origin` is the nickname for the GitHub remote. `git restore` undoes uncommitted changes.

**Works now:** Git name/email on this Mac; README commits; practice branch; pull request #1 merged; `git pull` on main; Python 3.13.7.

**Still shaky:** Remembering the exact order of add → commit → push.

**Next:** Phase 3 — virtual environments and packages.

## 2026-11-13 — Phase 3, virtual environments and packages

**Learned:** A venv is a private toolbox for this project. Activation makes the terminal use that toolbox. `requirements.txt` is the shopping list you commit; `.venv` stays local and is gitignored. `python -m pip` installs into the active environment.

**Works now:** `.venv` on Python 3.13.7; packages installed; `requirements.txt` frozen; README Setup section; deactivate breaks `import pandas`, activate fixes it.

**Still shaky:** Markdown code fences vs normal text.

**Next:** Phase 4 — getting the data.

## 2026-11-14 — Phase 4, getting the data

**Learned:** Config owns URL and path rules; download fetches and saves. Raw stays in `data/raw/` and is gitignored. Rerunning skips existing files unless forced. Credit the data source.

**Works now:** `season_url` / `raw_path` in config; `python -m src.download` pulls seasons 2223–2627; second run skips; README Data section.

**Still shaky:** (optional — one fuzzy thing, or leave as none)

**Next:** Phase 5 — cleaning with pandas.
