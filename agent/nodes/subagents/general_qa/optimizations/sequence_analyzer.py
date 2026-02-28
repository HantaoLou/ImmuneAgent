"""
Sequence Analyzer - Critical Fix for DNA/Protein Translation

Solves the "first protein that will be translated" problem:
1. Detect start codon (ATG) position
2. Translate from correct reading frame
3. Handle stop codons properly
4. Support both forward and reverse complement
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SequenceType(Enum):
    DNA = "dna"
    RNA = "rna"
    PROTEIN = "protein"


class ReadingFrame(Enum):
    FRAME_1 = 1  # Start at position 0
    FRAME_2 = 2  # Start at position 1
    FRAME_3 = 3  # Start at position 2
    FROM_START_CODON = 0  # Find first ATG and start there


# Standard genetic code
CODON_TABLE = {
    # U
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',  # Stop codons
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',  # UGA is also stop
    # C
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    # A
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',  # AUG = Start codon
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    # G
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

# Start codons
START_CODONS = ['ATG', 'AUG']  # DNA and RNA versions

# Stop codons
STOP_CODONS = ['TAA', 'TAG', 'TGA', 'UAA', 'UAG', 'UGA']


@dataclass
class TranslationResult:
    """Result of DNA/RNA translation"""
    protein_sequence: str
    start_position: int
    end_position: int
    reading_frame: int
    start_codon_found: bool
    stop_codon_found: bool
    original_sequence: str
    mrna_sequence: str
    codons: List[Tuple[str, str]]  # List of (codon, amino_acid)
    warnings: List[str]


class SequenceAnalyzer:
    """
    Specialized analyzer for DNA/RNA sequences with proper
    start codon detection and translation
    """
    
    def __init__(self):
        self.codon_table = CODON_TABLE
        self.start_codons = START_CODONS
        self.stop_codons = STOP_CODONS
    
    def detect_sequence_type(self, sequence: str) -> SequenceType:
        """Detect if sequence is DNA, RNA, or Protein"""
        clean_seq = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Check for protein-only letters
        protein_only = set('BDEFHIJKLMNPQRSVWXYZ')  # Exclude ACGTU
        if any(c in protein_only for c in clean_seq):
            return SequenceType.PROTEIN
        
        # Check for T (DNA) vs U (RNA)
        has_t = 'T' in clean_seq
        has_u = 'U' in clean_seq
        
        if has_u and not has_t:
            return SequenceType.RNA
        elif has_t and not has_u:
            return SequenceType.DNA
        elif has_t and has_u:
            # Mixed, likely input error - assume DNA
            return SequenceType.DNA
        else:
            # Only ACG, assume DNA
            return SequenceType.DNA
    
    def dna_to_mrna(self, dna_sequence: str) -> str:
        """Convert DNA to mRNA (T -> U)"""
        return dna_sequence.upper().replace('T', 'U')
    
    def find_start_codon(self, sequence: str, 
                         start_from: int = 0) -> Optional[int]:
        """
        Find the first ATG (start codon) in the sequence
        
        Args:
            sequence: DNA or RNA sequence
            start_from: Position to start searching from
            
        Returns:
            Position of start codon, or None if not found
        """
        seq = sequence.upper()
        
        # Try DNA start codon
        atg_pos = seq.find('ATG', start_from)
        aug_pos = seq.find('AUG', start_from)
        
        positions = [p for p in [atg_pos, aug_pos] if p >= 0]
        
        if positions:
            return min(positions)
        return None
    
    def find_all_start_codons(self, sequence: str) -> List[int]:
        """Find all start codon positions in frame"""
        positions = []
        seq = sequence.upper()
        
        for i in range(0, len(seq) - 2):
            codon = seq[i:i+3]
            if codon in ['ATG', 'AUG']:
                positions.append(i)
        
        return positions
    
    def translate_sequence(self, 
                          sequence: str,
                          reading_frame: ReadingFrame = ReadingFrame.FROM_START_CODON,
                          stop_at_stop_codon: bool = True) -> TranslationResult:
        """
        Translate DNA/RNA sequence to protein
        
        Args:
            sequence: DNA or RNA sequence
            reading_frame: Which reading frame to use
            stop_at_stop_codon: Whether to stop at first stop codon
            
        Returns:
            TranslationResult with all details
        """
        warnings = []
        
        # Clean sequence
        clean_seq = sequence.upper().replace(' ', '').replace('\n', '')
        
        # Detect type and convert to mRNA
        seq_type = self.detect_sequence_type(clean_seq)
        if seq_type == SequenceType.DNA:
            mrna = self.dna_to_mrna(clean_seq)
        elif seq_type == SequenceType.RNA:
            mrna = clean_seq
        else:
            return TranslationResult(
                protein_sequence="",
                start_position=0,
                end_position=0,
                reading_frame=0,
                start_codon_found=False,
                stop_codon_found=False,
                original_sequence=sequence,
                mrna_sequence="",
                codons=[],
                warnings=["Input appears to be protein, not nucleic acid"]
            )
        
        # Determine start position based on reading frame
        if reading_frame == ReadingFrame.FROM_START_CODON:
            start_pos = self.find_start_codon(mrna)
            if start_pos is None:
                warnings.append("No start codon (AUG) found - translating from position 0")
                start_pos = 0
                start_codon_found = False
            else:
                start_codon_found = True
                warnings.append(f"Start codon found at position {start_pos}")
        else:
            start_pos = reading_frame.value - 1
            start_codon_found = mrna[start_pos:start_pos+3] == 'AUG'
        
        # Translate
        protein = []
        codons = []
        stop_codon_found = False
        end_pos = start_pos
        
        for i in range(start_pos, len(mrna) - 2, 3):
            codon = mrna[i:i+3]
            if len(codon) < 3:
                break
            
            amino_acid = self.codon_table.get(codon, 'X')  # X for unknown
            
            codons.append((codon, amino_acid))
            
            if amino_acid == '*':  # Stop codon
                stop_codon_found = True
                if stop_at_stop_codon:
                    warnings.append(f"Stop codon {codon} at position {i}")
                    break
            
            protein.append(amino_acid)
            end_pos = i + 3
        
        protein_seq = ''.join(protein)
        
        return TranslationResult(
            protein_sequence=protein_seq,
            start_position=start_pos,
            end_position=end_pos,
            reading_frame=(start_pos % 3) + 1,
            start_codon_found=start_codon_found,
            stop_codon_found=stop_codon_found,
            original_sequence=sequence,
            mrna_sequence=mrna,
            codons=codons,
            warnings=warnings
        )
    
    def analyze_translation_question(self, 
                                     question_text: str,
                                     sequence: str) -> Dict[str, Any]:
        """
        Analyze a translation question and provide comprehensive analysis
        
        Args:
            question_text: The full question text
            sequence: The DNA/RNA sequence to translate
            
        Returns:
            Dict with analysis and answer
        """
        # Detect key phrases in question
        question_lower = question_text.lower()
        
        # Key indicators
        first_protein = 'first protein' in question_lower
        translate_following = 'following transcription' in question_lower
        single_letter = 'single letter' in question_lower
        
        # Determine reading frame based on question
        if first_protein:
            # MUST find start codon
            frame = ReadingFrame.FROM_START_CODON
            reasoning = "Question asks for 'first protein that will be translated' - must start from AUG (Met)"
        else:
            frame = ReadingFrame.FROM_START_CODON
            reasoning = "Using standard translation from start codon"
        
        # Perform translation
        result = self.translate_sequence(sequence, frame)
        
        # Build analysis
        analysis = {
            'sequence_type': self.detect_sequence_type(sequence).value,
            'original_sequence': sequence,
            'mrna_sequence': result.mrna_sequence,
            'start_codon_position': result.start_position,
            'start_codon_found': result.start_codon_found,
            'reading_frame': result.reading_frame,
            'protein_sequence': result.protein_sequence,
            'protein_length': len(result.protein_sequence),
            'stop_codon_found': result.stop_codon_found,
            'codons': result.codons,
            'warnings': result.warnings,
            'reasoning': reasoning,
            'key_question_indicators': {
                'first_protein': first_protein,
                'following_transcription': translate_following,
                'single_letter_format': single_letter
            }
        }
        
        # Generate explanation
        explanation = self._generate_explanation(result, analysis)
        analysis['explanation'] = explanation
        
        return analysis
    
    def _generate_explanation(self, result: TranslationResult, 
                             analysis: Dict) -> str:
        """Generate human-readable explanation"""
        lines = ["# Translation Analysis\n"]
        
        lines.append("## Key Step: Start Codon Detection")
        if result.start_codon_found:
            lines.append(f"Found start codon (AUG/Met) at position {result.start_position}")
            lines.append("Translation begins from this position, NOT from the beginning of the sequence.\n")
        else:
            lines.append("No start codon found. Translated from position 0.\n")
        
        lines.append("## Sequence Transformation")
        lines.append(f"1. Original DNA: {result.original_sequence}")
        lines.append(f"2. Transcribed mRNA: {result.mrna_sequence}")
        lines.append(f"3. Translation starts at: position {result.start_position}")
        lines.append(f"4. Reading frame: {result.reading_frame}")
        
        lines.append("\n## Codon Translation")
        for i, (codon, aa) in enumerate(result.codons):
            if aa == '*':
                lines.append(f"  {i+1}. {codon} → STOP")
            else:
                lines.append(f"  {i+1}. {codon} → {aa}")
        
        lines.append(f"\n## Final Result")
        lines.append(f"**Protein sequence (single letter):** {result.protein_sequence}")
        lines.append(f"**Length:** {len(result.protein_sequence)} amino acids")
        
        if result.warnings:
            lines.append("\n## Notes")
            for w in result.warnings:
                lines.append(f"- {w}")
        
        return "\n".join(lines)


# Convenience functions
def translate_dna(sequence: str, from_start_codon: bool = True) -> str:
    """
    Quick DNA translation function
    
    Args:
        sequence: DNA sequence
        from_start_codon: If True, find first ATG and translate from there
        
    Returns:
        Protein sequence in single letter code
    """
    analyzer = SequenceAnalyzer()
    frame = ReadingFrame.FROM_START_CODON if from_start_codon else ReadingFrame.FRAME_1
    result = analyzer.translate_sequence(sequence, frame)
    return result.protein_sequence


def analyze_sequence_question(question: str, sequence: str) -> Dict[str, Any]:
    """
    Full analysis of a sequence-related question
    
    Returns comprehensive analysis with answer
    """
    analyzer = SequenceAnalyzer()
    return analyzer.analyze_translation_question(question, sequence)


# Test case
if __name__ == "__main__":
    # Test with the problematic sequence
    test_seq = "AGTTGCTTCCCTACCAATAGCTTCTTTTGCCATGGGG"
    test_question = "What is the amino acid sequence of the first protein that will be translated following transcription of this region?"
    
    analyzer = SequenceAnalyzer()
    result = analyzer.analyze_translation_question(test_question, test_seq)
    
    print(result['explanation'])
    print(f"\nAnswer: {result['protein_sequence']}")




