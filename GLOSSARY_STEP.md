# Glossary step: how to run it

Files go in your project root (tests/ goes in your existing tests/ folder). No new packages: stdlib only.

1. Paste `config_additions.py` into `config.py`, then put your real email in `GLOSSARY_USER_AGENT`.
   Set the full PLAYLIST_ID first: the glossary folder is named after it.
2. Add this line to `.gitignore` (keep aliases.csv and domain_pack.json in git, ignore only the cache):
   `glossary/*/_cache/`
3. `python glossary_builder.py propose --topic "Mahabharata"`
   Open `glossary/<PLAYLIST_ID>/domain_pack.json`. For each category that fits: set `"use": true`, a `"type"`
   (person, place, text, concept, ...) and a `"depth"`. Delete the rest. `_pages`/`_subcats` show how big each is.
4. `python glossary_builder.py build`
   Check `build_report.json` first (look for `truncated: true` and the counts), then `aliases_candidates.csv`.
   Terms with no Telugu spelling are in `missing_telugu.csv`; shared spellings are in `ambiguous.csv`.
5. Review candidates. In the `decision` column write `accept` or `reject` where you have an opinion.
6. `python glossary_builder.py promote --accept label`   (safest: Wikidata labels and Telugu page titles only)
   `--accept decided` takes only rows you marked accept (the only mode that lets ambiguous spellings in).
   `--accept all` takes every non-rejected, non-ambiguous row.
   Promote only ever appends. Re-running it is safe.

In the corrector's loader, call `glossary_builder.require_alias_table(config.ALIAS_TABLE_PATH)` so a missing
or empty glossary gives a clear "build it first" message.

Tests: `python -m pytest tests/test_glossary_builder.py`  (no internet needed)
