# Release Manifest

This manifest defines the intended contents of the publishable Xamt++ source
artifact.

## Include

- `README.md`
- `ARTIFACT.md`
- `RELEASE_MANIFEST.md`
- `CITATION.cff`
- `LICENSE`
- `requirements-main.txt`
- `MANIFEST.in`
- `tools/*.py`
- `bug_repros/*.py`
- `bug_repros/metadata.json`
- `bug_repros/README.md`
- Result summaries:
  - `PAIRWISE_ADAPTER_SUMMARY.md`
  - `BUG_CANDIDATES.md`
  - `ALL_BUG_CANDIDATES.md`
  - `REAL_BUG_AUDIT.md`
  - `CURRENT_BAD_TRIAGE.md`
  - `SELECTED_DIFF_CANDIDATES.md`
  - `POST_FIX_ERROR_REPLAY.md`
- Legacy XAMT compatibility code:
  - `functions/`
  - `inputs/`
  - `run_tasks/`
  - `tests/`
  - `utilities/`
  - `try.py`
  - `LEGACY_XAMT_README.md`

## Exclude

- `__pycache__/`
- `.pytest_cache/`
- `.git/`
- `rank_0/`
- `results/`
- `dist/`
- Temporary run files and local logs
- External Python environments
- Historical raw files under `/tmp`

## Pre-Release Checklist

Run these from `Xamt/`:

```bash
python -B tools/artifact_check.py
python -B tools/build_artifact.py --out dist/xamtplusplus-artifact.tar.gz
```

For a complete experimental rerun, configure the external worker variables
listed in `ARTIFACT.md` before running adapter-aware validation or timed fuzzing.
