# IgBLAST Error Patterns

## No sequences found in FASTA
- match: No sequences found
- failure_kind: MCP_TOOL_ERROR
- description: Input FASTA file is empty or has invalid format
- recovery: Check FASTA format. Each sequence needs a > header line followed by the sequence. CSV files are auto-converted — check column names match expected aliases.

## Invalid organism
- match: germline database
- failure_kind: MCP_TOOL_ERROR
- description: Specified organism does not match available germline databases
- recovery: Use one of: human, mouse, rhesus_monkey. Check organism parameter.

## Timeout during batch processing
- match: timed out
- failure_kind: TIMEOUT
- description: V(D)J analysis took too long for large input
- recovery: Increase timeout parameter. For >10,000 sequences, use timeout=14400 (4 hours). Consider splitting input.

## Connection refused
- match: ECONNREFUSED
- failure_kind: MCP_UNREACHABLE
- description: IgBLAST MCP server not running
- recovery: Check that the igblast server is running on configured port.

## Args wrapping error
- match: not of type 'object'
- failure_kind: MCP_TOOL_ERROR
- description: Nested parameter serialization error
- recovery: Server may need the flat-params fix applied in Session 11 for nettcr. Check MCP server version.
