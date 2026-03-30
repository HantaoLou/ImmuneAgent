# MetaBCR Error Patterns

## Missing heavy chain column
- match: heavy_chain
- failure_kind: MCP_TOOL_ERROR
- description: Input CSV missing required heavy chain column
- recovery: Check for column aliases: Heavy, VH, heavy_dna, Heavy_DNA. Rename to heavy_chain if needed.

## Missing light chain column
- match: light_chain
- failure_kind: MCP_TOOL_ERROR
- description: Input CSV missing required light chain column
- recovery: Check for aliases: Light, VL, light_dna, Light_DNA. MetaBCR requires paired chains — single-chain not supported.

## Unsupported antigen type
- match: antigen_type
- failure_kind: MCP_TOOL_ERROR
- description: Requested antigen type not in supported list
- recovery: Use one of: flu, sars, rsv, hiv. Check spelling.

## Empty predictions
- match: 0 predictions
- failure_kind: OUTPUT_MISSING
- description: No binding predictions produced
- recovery: Input sequences may be invalid or too short. Check that heavy/light chain columns contain actual amino acid sequences, not gene names.

## Connection refused
- match: ECONNREFUSED
- failure_kind: MCP_UNREACHABLE
- description: MetaBCR MCP server not running
- recovery: Check that metabcr server is running on configured port.

## Timeout
- match: timed out
- failure_kind: TIMEOUT
- description: Prediction took too long for large input
- recovery: Reduce batch size. MetaBCR with ensemble model is slow for >1000 sequences.
