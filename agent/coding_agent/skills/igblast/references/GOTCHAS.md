# IgBLAST Gotchas

1. **Streaming response.** Unlike nettcr/metabcr, igblast returns a `streaming_task`. Must poll the streaming URL with bash (`curl`) until `type=result`. Without bash permission, polling fails.

2. **CSV column aliases are strict.** BCR: Heavy_DNA, Light_DNA, Heavy, Light, VH, VL, sequence. TCR: alpha_dna, beta_dna, TRA, TRB, CDR3a, CDR3b. Other names are not recognized.

3. **FASTA headers must be unique.** Duplicate sequence IDs cause silent overwrite in AIRR output.

4. **Organism must match germline database.** Using `mouse` sequences with `human` germline produces garbage V/D/J assignments with no warning.

5. **Timeout for large files.** Default 7200s (2 hours). Files with >10,000 sequences may need more. Set `timeout` explicitly.

6. **Output is AIRR TSV, not CSV.** Tab-separated, not comma-separated. Downstream tools must use `\t` delimiter.

7. **ANSI escape codes in logs.** When captured via nohup without TTY, output contains orphan `[0m` sequences.
