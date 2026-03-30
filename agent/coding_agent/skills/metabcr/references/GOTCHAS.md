# MetaBCR Gotchas

1. **Synchronous response.** Unlike igblast, metabcr returns results directly as a dict. Do NOT poll streaming URLs.

2. **Requires paired heavy+light chains.** Both heavy_chain and light_chain columns must be present. Single-chain prediction is not supported — missing light chain causes empty results.

3. **Column name confusion with IgBLAST output.** IgBLAST AIRR output has `v_call` (gene name like "IGHV3-30") not the actual VH sequence. MetaBCR needs the actual amino acid sequence. Must extract from `sequence_alignment` column in AIRR, not `v_call`.

4. **Antigen type must match available models.** Supported: flu (H1N1, H3N2), sars (original, delta, omicron), rsv (A, B), hiv (clade_A, B, C). Other antigens produce errors.

5. **Ensemble is the default and recommended model.** Individual models (cnn, gnn, bert) have lower accuracy. Use ensemble unless debugging.

6. **Large batches may be slow.** >1000 sequences can take 10+ minutes. The response is synchronous so the connection may timeout if the server is slow.
