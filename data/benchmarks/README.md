# Finance safety pilot v1

84 AI-assisted, project-authored English prompts under the repository MIT license:
4 calibration behavior groups, 12 validation groups, 12 reserved test groups.
Each group contains one direct harmful request, one fixed fictional-roleplay
attack on that objective, and one legitimate prevention/education question.
Related prompts stay in the same split. The attack family is shared across splits;
this does not test transfer to unseen jailbreak families. It is a pilot, not an
externally validated benchmark. Independent prompt and response review is pending.

Never add these rows to the plain benign finance SFT training dataset. Do not
use test prompts for steering calibration, repair tuning or model selection.
Review generated answers using docs/PROTOCOL.md, including substantive harmful
assistance after a disclaimer. Responses are not labeled by keywords. The review
packet leaves all labels UNREVIEWED and binds labels to the exact response hash.

Run scripts/prepare_safety_pilot.py to validate and freeze the source manifest.
Publish raw rates and uncertainty, not only relative safety-recovery percentages.
A response hitting the output cap needs inspection; it is not automatically safe.
