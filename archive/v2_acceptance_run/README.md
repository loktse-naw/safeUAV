# v2 acceptance-controlled run archive

This record identifies the immutable artifacts used by `feedback/08_P1受控复跑报告.md`.

- Implementation: `p1-prototype-v2.0.0-acceptance`
- Source digest recorded by every training manifest: `0f8b7800c9537cec`
- Uploaded source bundle: `pytorch_code/artifacts/securelink_v2_bundle.tar.gz`
- Source bundle SHA-256: `F78590691330C35B9A4E274124DD073F0176DC314714705A1FFD60CDC103160F`
- Downloaded result bundle: `pytorch_code/artifacts/securelink_v2_results.tar.gz`
- Result bundle SHA-256: `18B24F05E5D662841E4C5DF5D2A2FC2B24B30EDDE820D961C0C7C39F7B59178E`
- Legacy B01 config digest: `baab937d44ac27dd`
- Centered B01/B00/B10/B11 config digest: `62993829b6b793fa`
- Remote tests: 16 passed in 4.64 s
- Final-test status: seeds starting at 50001 reserved, not used

The exact resolved configurations used by the run remain inside each result directory. After the run, the editable source configuration corrected a prose-only note from “one 8x512 rollout” to the actual `16 x (4 environments x 512 steps)` accounting. No physical, learning, or evaluation parameter changed; the immutable result manifests retain their original digest.
