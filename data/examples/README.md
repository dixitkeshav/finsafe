# Illustrative data only

The 22 rows in `seed.jsonl` exercise file formats and the pipeline. They are
not a research benchmark and must not support headline safety/utility claims.
Six arithmetic training rows, four calibration rows, four validation rows and
eight test rows are deliberately tiny. Curation can remove similar arithmetic
templates: inspect the manifest rather than treating every removal as an error.

Team-authored seed data is licensed CC-BY-4.0, attribution: FinSafe contributors.
Safety prompts request wrongdoing for controlled evaluation; no harmful answers
are supplied. Related direct/adversarial prompts share a group and split.
The arithmetic examples share a template across splits and only demonstrate
in-template testing. A substantive experiment needs held-out documents or task
families plus semantic leak review; the automatic check cannot certify that.
