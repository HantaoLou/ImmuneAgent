# NetTCR Gotchas

1. **All 7 columns MUST exist in native format.** CSV must have peptide, A1, A2, A3, B1, B2, B3 even if A1/A2/B1/B2 are empty strings. Missing columns cause `KeyError`, not graceful fallback.

2. **Peptide length 8-15 AA only (MHC-I).** Longer peptides (MHC-II) are silently ignored — zero predictions produced with no error.

3. **Output path differs from input path.** The tool appends suffixes like `_nettcr_results.csv`. Always read `result_path` from the response JSON to find actual output.

4. **Duplicates not deduplicated.** 1000 copies of the same TCR → 1000 identical rows. Dedup upstream.

5. **Empty CDR columns become NaN floats.** When reading CSV, empty A1/A2/B1/B2 become `float('nan')`. The server now handles this (zero-padding), but downstream tools may not.

6. **Response is synchronous, NOT streaming_task.** Do not poll streaming URLs. The response comes back directly as a dict.

7. **Fabricated results when server unreachable.** If the MCP server returns 502 or is down, OpenCode's LLM may silently fabricate plausible-looking results. Always check output files actually exist.

8. **26 pretrained peptides only for percentile rank.** Other peptides use pan-specific model with lower accuracy. Always run `check_peptide_support` first.

9. **Column aliases from other tools don't work.** NetTCR expects exactly `peptide,A1,A2,A3,B1,B2,B3` or `peptide,CDR3a,CDR3b,TRA_v_gene,TRB_v_gene`. AIRR format columns like `cdr3_aa`, `v_call` must be renamed.
