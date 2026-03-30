# NetTCR Error Patterns

## Missing CDR3 column
- match: KeyError: 'B3'
- failure_kind: MCP_TOOL_ERROR
- description: Input CSV missing required CDR3 beta column
- recovery: Input uses non-standard column names. Check for CDR3b, cdr3_beta, CDR3_TRB, junction_aa. Run prepare to normalize.

## Column B3 not found
- match: Column B3 not found
- failure_kind: MCP_TOOL_ERROR
- description: CSV header missing B3 column
- recovery: Ensure CSV has all 7 columns: peptide, A1, A2, A3, B1, B2, B3. Empty columns must still have headers.

## Unsupported peptide fallback
- match: not in pretrained list
- failure_kind: MCP_TOOL_ERROR
- description: Target peptide does not have a pretrained model
- recovery: Pan-specific model fallback is automatic but accuracy is lower. Run check_peptide_support first. This is expected behavior.

## Empty prediction output
- match: 0 predictions written
- failure_kind: OUTPUT_MISSING
- description: All input sequences filtered out during validation
- recovery: Run validate_tcr_input to identify bad rows. Check for non-standard amino acids (B, J, O, U, X, Z) and peptide length (must be 8-15 AA).

## NaN in CDR columns
- match: TypeError: float has no len
- failure_kind: MCP_TOOL_ERROR
- description: Empty CDR1/CDR2 values read as NaN floats
- recovery: Ensure empty CDR columns contain empty strings, not NaN. The server now handles this, but check the input CSV if using a custom pipeline.

## Args wrapping error
- match: not of type 'object'
- failure_kind: MCP_TOOL_ERROR
- description: Tool parameter serialization error from nested Pydantic model
- recovery: This is a server-side bug (fixed in Session 11). If seen, the MCP server needs updating. Use flat parameters.

## Connection refused
- match: ECONNREFUSED
- failure_kind: MCP_UNREACHABLE
- description: NetTCR MCP server not running
- recovery: Check that the nettcr server is running on the configured port. Verify URL in MCP config.

## Timeout
- match: timed out
- failure_kind: TIMEOUT
- description: Prediction took too long
- recovery: Large input files (>5000 sequences) may need longer timeout. Consider splitting input into batches.
