# GitHub project workflow

The user authorized creating a new repository and committing/pushing this project on 13 September 2026. The selected repository is private so unpublished work remains private by default.

GitHub CLI authentication uses the local credential manager. Passwords, access tokens, and device codes are never stored in project files. Use `gh auth login --hostname github.com --git-protocol https --web` to authenticate on another machine.

Track source, configurations, tests, dependency locks, documentation, and compact verified reports. Ignore original data, model weights, generated labels and images, environments, caches, credentials, and detailed per-location records. Keep third-party code in an ignored pinned checkout downloaded by bootstrap.

Before each push:

```powershell
& .\.venv\python.exe -m pytest -q
git diff --check
git status --short
git diff --cached --stat
& .\.venv\python.exe scripts/check_publish.py
git commit -m "Describe the concrete tested change"
git push
```

Update the implementation status only from actual run outputs. A synthetic GPU test is software/resource evidence, not real-image quality evidence. CI verifies protocol code on CPU and is separate from CUDA profiling.

Ordinary commits/pushes can continue when work is requested in this task. No recurring background updater or scheduled automation has been created.
