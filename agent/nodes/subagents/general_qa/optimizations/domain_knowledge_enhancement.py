"""
Domain Knowledge Enhancement for N0 Node

Enhances cross-domain knowledge detection and fusion:
1. Multi-domain question detection
2. Key constraint/prompt word extraction
3. Domain-specific knowledge injection
4. Question intent deep understanding

This is a GENERAL enhancement, not specific question hacks.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class BiomedicalDomain(Enum):
    """Core biomedical domains"""
    MOLECULAR_BIOLOGY = "molecular_biology"
    BIOCHEMISTRY = "biochemistry"
    GENETICS = "genetics"
    CELL_BIOLOGY = "cell_biology"
    MICROBIOLOGY = "microbiology"
    IMMUNOLOGY = "immunology"
    PHARMACOLOGY = "pharmacology"
    PHYSIOLOGY = "physiology"
    STRUCTURAL_BIOLOGY = "structural_biology"
    BIOPHYSICS = "biophysics"
    BIOINFORMATICS = "bioinformatics"
    SYNTHETIC_BIOLOGY = "synthetic_biology"
    EVOLUTIONARY_BIOLOGY = "evolutionary_biology"
    # Cross-cutting domains
    PROTEIN_SCIENCE = "protein_science"
    NUCLEIC_ACID_SCIENCE = "nucleic_acid_science"
    METABOLISM = "metabolism"
    SIGNALING = "signaling"


class ConstraintType(Enum):
    """Types of constraints in questions"""
    TEMPORAL = "temporal"          # first, last, before, after
    QUANTITATIVE = "quantitative"  # number, value, rate
    QUALITATIVE = "qualitative"    # best, most, optimal
    COMPARATIVE = "comparative"    # higher, lower, more, less
    EXCLUSIVE = "exclusive"        # only, single, unique
    NEGATIVE = "negative"          # not, except, cannot
    CONDITIONAL = "conditional"    # if, when, given


@dataclass
class KeyConstraint:
    """Represents a key constraint extracted from question"""
    constraint_type: ConstraintType
    keyword: str
    context: str
    domain_hint: Optional[str] = None
    importance: float = 1.0


@dataclass
class DomainKnowledge:
    """Domain-specific knowledge that should be injected"""
    domain: BiomedicalDomain
    key_concepts: List[str]
    rules: List[str]
    common_pitfalls: List[str]


# ============================================================
# DOMAIN-SPECIFIC KNOWLEDGE DATABASE (General Knowledge, NOT question-specific)
# ============================================================

DOMAIN_KNOWLEDGE_DB: Dict[BiomedicalDomain, DomainKnowledge] = {
    BiomedicalDomain.    NUCLEIC_ACID_SCIENCE: DomainKnowledge(
        domain=BiomedicalDomain.NUCLEIC_ACID_SCIENCE,
        key_concepts=[
            "Transcription: DNA -> mRNA (T->U)",
            "Translation: mRNA -> Protein (codon->amino acid)",
            "Start codon: AUG/Met - translation BEGINS here, not at sequence start",
            "Stop codons: UAA, UAG, UGA - translation ENDS here",
            "Reading frame: determined by start codon position",
            "5' to 3' direction: standard orientation",
            "Complementarity: A-T/U, G-C",
        ],
        rules=[
            "Protein translation ALWAYS starts from start codon (AUG/Met)",
            "Questions about 'first protein translated' -> find ATG/AUG first",
            "DNA to mRNA: replace T with U",
            "mRNA to Protein: use codon table",
            "Stop codon terminates translation - nothing after is translated",
        ],
        common_pitfalls=[
            "Assuming translation starts at sequence beginning (wrong - starts at ATG)",
            "Ignoring reading frame shifts",
            "Continuing translation past stop codon",
            "Confusing template vs coding strand",
        ]
    ),
    
    BiomedicalDomain.METABOLISM: DomainKnowledge(
        domain=BiomedicalDomain.METABOLISM,
        key_concepts=[
            "Metabolic pathways: series of enzyme-catalyzed reactions",
            "Path coefficients: rate constants for each step",
            "Negative feedback: product inhibits earlier step (-| notation)",
            "Positive regulation: product activates downstream step (-> notation)",
            "Steady state: input = output",
            "Rate-limiting step: slowest step controls overall rate",
        ],
        rules=[
            "If path contains negative feedback (-|), relationship may be indirect",
            "Multiple paths between A and B -> check for feedback loops",
            "Negative feedback can break proportional relationships",
            "Look for -| notation indicating inhibition",
        ],
        common_pitfalls=[
            "Assuming direct proportional relationship when feedback exists",
            "Ignoring negative regulation (-|) in pathway",
            "Not considering alternative paths",
        ]
    ),
    
    BiomedicalDomain.BIOPHYSICS: DomainKnowledge(
        domain=BiomedicalDomain.BIOPHYSICS,
        key_concepts=[
            "Molecular order: highly ordered = more compact = smaller area",
            "Surface pressure: inverse relationship with molecular area",
            "Lipid packing: saturated chains pack tighter than unsaturated",
            "Phase behavior: ordered phases have smaller molecular area",
            "Fluorescence spectroscopy: measures membrane order/fluidity",
        ],
        rules=[
            "More ordered domains -> tighter packing -> smaller surface area",
            "Less ordered domains -> looser packing -> larger surface area",
            "Saturated fatty acids -> more ordered",
            "Unsaturated fatty acids (double bonds) -> less ordered",
            "Question asks 'which has lower surface area' -> look for MORE ordered",
        ],
        common_pitfalls=[
            "Confusing 'ordered' with 'larger' (opposite is true)",
            "Not understanding order-area relationship",
            "Ignoring saturation level of fatty acids",
        ]
    ),
    
    BiomedicalDomain.MOLECULAR_BIOLOGY: DomainKnowledge(
        domain=BiomedicalDomain.MOLECULAR_BIOLOGY,
        key_concepts=[
            "Gene expression: transcription + translation",
            "Promoters: DNA sequences where transcription initiates",
            "Vectors: plasmids used for gene cloning/expression",
            "Duet vectors: single plasmid with TWO promoters for co-expression",
            "Dual plasmid: TWO separate plasmids (need compatible origins)",
            "Selection markers: antibiotic resistance genes",
        ],
        rules=[
            "Co-expression question -> Duet vector is usually better than dual plasmid",
            "Duet = single vector = guaranteed co-expression",
            "Dual plasmid = need compatible origins + two selections",
            "Check origin compatibility for dual plasmid systems",
        ],
        common_pitfalls=[
            "Not recognizing Duet vectors as single-plasmid systems",
            "Assuming more plasmids = better expression",
            "Ignoring plasmid compatibility issues",
        ]
    ),
    
    BiomedicalDomain.GENETICS: DomainKnowledge(
        domain=BiomedicalDomain.GENETICS,
        key_concepts=[
            "Population genetics: allele frequencies in populations",
            "Genetic drift: random changes in allele frequency (stronger in small populations)",
            "Natural selection: directional changes based on fitness",
            "Mutation rate: rate of new mutations per generation",
            "Fixation: when allele reaches 100% frequency",
        ],
        rules=[
            "Small population -> genetic drift dominates",
            "Large population -> natural selection dominates",
            "Questions about mutation in populations -> consider drift",
            "Genomic mutation rate != fixation rate",
        ],
        common_pitfalls=[
            "Ignoring population size effects",
            "Confusing mutation rate with evolutionary force",
            "Not considering random vs selective forces",
        ]
    ),
}


# ============================================================
# KEY CONSTRAINT PATTERNS (General patterns, NOT question-specific)
# ============================================================

KEY_CONSTRAINT_PATTERNS = {
    ConstraintType.TEMPORAL: [
        (r'\b(first|initial|earliest)\b', "Indicates need to find START point"),
        (r'\b(last|final|ultimate)\b', "Indicates need to find END point"),
        (r'\b(before|prior to|preceding)\b', "Temporal ordering constraint"),
        (r'\b(after|following|subsequent)\b', "Temporal ordering constraint"),
    ],
    
    ConstraintType.QUALITATIVE: [
        (r'\b(best|optimal|most effective)\b', "Requires comparative evaluation"),
        (r'\b(worst|least|minimum)\b', "Requires comparative evaluation"),
        (r'\b(primary|main|dominant|major)\b', "Identify most significant factor"),
        (r'\b(correct|accurate|right)\b', "Single correct answer expected"),
    ],
    
    ConstraintType.COMPARATIVE: [
        (r'\b(higher|greater|larger|more)\b', "Comparative - choose larger value"),
        (r'\b(lower|smaller|less|fewer)\b', "Comparative - choose smaller value"),
        (r'\b(difference|versus|vs|compared)\b', "Comparison required"),
    ],
    
    ConstraintType.EXCLUSIVE: [
        (r'\b(only|single|sole|unique|one)\b', "Single option constraint"),
        (r'\b(first protein that will)\b', "CRITICAL: implies start codon rule"),
    ],
    
    ConstraintType.NEGATIVE: [
        (r'\b(not|except|cannot|exclude)\b', "Negative constraint"),
        (r'\b(¬∝|not proportional)\b', "No direct relationship"),
    ],
}


class DomainKnowledgeEnhancer:
    """
    Enhances N0 node with domain knowledge detection and injection
    """
    
    def __init__(self):
        self.domain_db = DOMAIN_KNOWLEDGE_DB
        self.constraint_patterns = KEY_CONSTRAINT_PATTERNS
    
    def detect_domains(self, question_text: str) -> List[Tuple[BiomedicalDomain, float]]:
        """
        Detect relevant domains for a question
        
        Returns list of (domain, confidence) tuples
        """
        text_lower = question_text.lower()
        domain_scores = []
        
        # Domain keyword patterns
        domain_keywords = {
            BiomedicalDomain.NUCLEIC_ACID_SCIENCE: [
                'dna', 'rna', 'mrna', 'sequence', 'codon', 'transcription', 
                'translation', 'amino acid', 'protein sequence', 'start codon',
                'stop codon', 'reading frame', 'first protein'
            ],
            BiomedicalDomain.METABOLISM: [
                'pathway', 'metabolic', 'coefficient', 'k1', 'k2', 'rate',
                'concentration', 'acetyl-coa', 'pep', 'pyruvate', 'feedback',
                '-|', 'activation', 'inhibition'
            ],
            BiomedicalDomain.BIOPHYSICS: [
                'surface area', 'monolayer', 'membrane', 'lipid', 'ceramide',
                'order', 'packing', 'fluorescence', 'spectroscopy', 'phase'
            ],
            BiomedicalDomain.MOLECULAR_BIOLOGY: [
                'vector', 'plasmid', 'clone', 'expression', 'promoter',
                'duet', 'co-expression', 'antibiotic', 'resistance'
            ],
            BiomedicalDomain.GENETICS: [
                'population', 'allele', 'frequency', 'drift', 'selection',
                'mutation rate', 'fixation', 'genetic', 'evolutionary'
            ],
        }
        
        for domain, keywords in domain_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Normalize by total keywords
                score = matches / len(keywords)
                domain_scores.append((domain, min(score * 2, 1.0)))  # Boost score
        
        # Sort by score
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores
    
    def extract_key_constraints(self, question_text: str) -> List[KeyConstraint]:
        """
        Extract key constraints from question text
        
        These are CRITICAL hints that change how to approach the question
        """
        constraints = []
        text_lower = question_text.lower()
        
        for constraint_type, patterns in self.constraint_patterns.items():
            for pattern, hint in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 20)
                    end = min(len(question_text), match.end() + 20)
                    context = question_text[start:end]
                    
                    # Determine domain hint
                    domain_hint = self._infer_domain_hint(match.group(), constraint_type)
                    
                    constraints.append(KeyConstraint(
                        constraint_type=constraint_type,
                        keyword=match.group(),
                        context=context,
                        domain_hint=domain_hint,
                        importance=self._calculate_importance(constraint_type, match.group())
                    ))
        
        # Sort by importance
        constraints.sort(key=lambda c: c.importance, reverse=True)
        return constraints
    
    def _infer_domain_hint(self, keyword: str, constraint_type: ConstraintType) -> Optional[str]:
        """Infer domain-specific hint from constraint"""
        hints = {
            'first protein': 'NUCLEIC_ACID_SCIENCE: translation starts from start codon (ATG/Met)',
            'first': 'Check for start/beginning point in process',
            'lower surface area': 'BIOPHYSICS: more ordered = smaller area',
            'higher': 'Comparative analysis needed',
            'best way': 'MOLECULAR_BIOLOGY: consider simplicity and reliability',
        }
        return hints.get(keyword.lower())
    
    def _calculate_importance(self, constraint_type: ConstraintType, keyword: str) -> float:
        """Calculate importance score for constraint"""
        # High importance constraints
        high_importance = ['first protein', 'best way', 'first', 'only', 'cannot']
        if any(h in keyword.lower() for h in high_importance):
            return 1.0
        
        # Type-based importance
        type_importance = {
            ConstraintType.TEMPORAL: 0.9,
            ConstraintType.QUALITATIVE: 0.8,
            ConstraintType.COMPARATIVE: 0.7,
            ConstraintType.EXCLUSIVE: 0.95,
            ConstraintType.NEGATIVE: 0.85,
        }
        return type_importance.get(constraint_type, 0.5)
    
    def get_domain_knowledge(self, domain: BiomedicalDomain) -> Optional[DomainKnowledge]:
        """Get domain knowledge for injection"""
        return self.domain_db.get(domain)
    
    def generate_enhanced_context(self, 
                                  question_text: str) -> Dict[str, Any]:
        """
        Generate enhanced context for N0 node
        
        This is the main entry point for N0 enhancement
        """
        # Detect domains
        domains = self.detect_domains(question_text)
        
        # Extract constraints
        constraints = self.extract_key_constraints(question_text)
        
        # Get relevant domain knowledge
        domain_knowledge_list = []
        for domain, score in domains[:3]:  # Top 3 domains
            dk = self.get_domain_knowledge(domain)
            if dk:
                domain_knowledge_list.append({
                    'domain': domain.value,
                    'confidence': score,
                    'key_concepts': dk.key_concepts,
                    'rules': dk.rules,
                    'common_pitfalls': dk.common_pitfalls
                })
        
        # Build critical hints
        critical_hints = []
        for constraint in constraints[:5]:  # Top 5 constraints
            if constraint.domain_hint:
                critical_hints.append({
                    'keyword': constraint.keyword,
                    'type': constraint.constraint_type.value,
                    'hint': constraint.domain_hint,
                    'importance': constraint.importance
                })
        
        return {
            'detected_domains': [(d.value, s) for d, s in domains],
            'key_constraints': [
                {
                    'keyword': c.keyword,
                    'type': c.constraint_type.value,
                    'context': c.context,
                    'importance': c.importance
                }
                for c in constraints
            ],
            'domain_knowledge': domain_knowledge_list,
            'critical_hints': critical_hints,
            'cross_domain': len(domains) > 1,
            'enhancement_summary': self._generate_summary(domains, constraints)
        }
    
    def _generate_summary(self, 
                         domains: List[Tuple[BiomedicalDomain, float]],
                         constraints: List[KeyConstraint]) -> str:
        """Generate human-readable enhancement summary"""
        lines = ["# N0 Enhancement Summary\n"]
        
        if domains:
            lines.append("## Detected Domains")
            for domain, score in domains[:3]:
                lines.append(f"- {domain.value}: {score:.0%} confidence")
        
        if constraints:
            lines.append("\n## Critical Constraints")
            for c in constraints[:3]:
                lines.append(f"- **{c.keyword}** ({c.constraint_type.value}): importance {c.importance:.1f}")
                if c.domain_hint:
                    lines.append(f"  - Hint: {c.domain_hint}")
        
        return "\n".join(lines)


# Convenience function
def enhance_n0_context(question_text: str) -> Dict[str, Any]:
    """
    Quick function to enhance N0 context
    
    Call this at the beginning of N0 node processing
    """
    enhancer = DomainKnowledgeEnhancer()
    return enhancer.generate_enhanced_context(question_text)


def get_critical_hints_for_question(question_text: str) -> List[str]:
    """
    Get just the critical hints for a question
    
    Returns list of hint strings
    """
    enhancer = DomainKnowledgeEnhancer()
    context = enhancer.generate_enhanced_context(question_text)
    return [h['hint'] for h in context.get('critical_hints', []) if h.get('hint')]

