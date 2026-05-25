# Post-Fix Error Replay

Snapshot: `2026-05-25 10:43:06 AEST`
Source run: `/tmp/xamt_wide_arity_nonfinite_20260525_091040_seed20260539_auto`

Replayed every current `ERROR` group seed against the patched adapter code.

| Replay State | Groups |
| --- | ---: |
| PASS | 37 |
| SKIP | 49 |
| DIFF | 2 |
| ERROR | 0 |

Notes:
- The original current-run JSONL still contains old `ERROR` rows because `seed20260539` started before these patches.
- New runs started after this patch should load the fixed adapter code.
- The two replay `DIFF` cases are not execution failures; they now need normal DIFF triage.
