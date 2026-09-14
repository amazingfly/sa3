# Local storage and component boundary

Run commands from the repository root. Relative paths in configuration files are
resolved by the existing workflow; presets containing absolute paths describe
specific historical runs and must be adapted for another machine.

On the original workstation, `outputs`, `musicLibrary`, and `web_runs` are
symlinks into `/mnt/storage/sa3Archive/project/`. They remain untouched. A fresh
clone can use ordinary directories or link equivalent stores at those paths.
Keep model access tokens in the environment or your existing local token store.

The `sd15Files` compatibility directory is migrated to the images repository.
The current web server already uses `SD15_REPO` to locate that repository.
The LTX checkout's old copied SA3 Colab example is retained in its local backup;
this repository's `colab/` is the authoritative audio worker implementation.
