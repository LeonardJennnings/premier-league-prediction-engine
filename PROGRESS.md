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
