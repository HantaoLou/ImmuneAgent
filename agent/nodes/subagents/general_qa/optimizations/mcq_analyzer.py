"""
MCQ Option Analyzer - P0 Priority Optimization

Provides deep analysis of multiple choice options to:
1. Parse option structure and identify key entities
2. Recognize vector/plasmid types and their characteristics
3. Compare options against domain knowledge
4. Generate exclusion/match recommendations
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class OptionMatchStatus(Enum):
    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    EXCLUDE = "exclude"
    NEED_MORE_INFO = "need_more_info"


class VectorType(Enum):
    SINGLE_VECTOR = "single_vector"
    DUET_VECTOR = "duet_vector"  # Dual-promoter single vector
    DUAL_PLASMID = "dual_plasmid"
    UNKNOWN = "unknown"


@dataclass
class EntityInfo:
    """Information about an entity in an option"""
    name: str
    entity_type: str  # e.g., "vector", "promoter", "antibiotic", "gene"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionAnalysis:
    """Complete analysis of a single option"""
    option_id: str
    option_text: str
    entities: List[EntityInfo] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (entity1, relation, entity2)
    vector_type: VectorType = VectorType.UNKNOWN
    structure_type: str = ""  # e.g., "dual_plasmid", "single_duet"
    keywords: Set[str] = field(default_factory=set)
    match_status: OptionMatchStatus = OptionMatchStatus.NEED_MORE_INFO
    match_reason: str = ""
    confidence: float = 0.0


# Known vector characteristics database
VECTOR_DATABASE = {
    # Duet vectors (dual-promoter single vectors)
    "pCDFDuet-1": {
        "type": "duet_vector",
        "promoters": 2,
        "origins": ["CDF"],
        "antibiotics": ["spectinomycin"],
        "copy_number": "medium",
        "features": ["MCS1", "MCS2", "His-tag", "S-tag"],
        "advantages": ["no_plasmid_incompatibility", "guaranteed_coexpression", "single_selection"],
        "description": "Duet vector with two independent MCS for co-expression"
    },
    "pETDuet-1": {
        "type": "duet_vector",
        "promoters": 2,
        "origins": ["pBR322"],
        "antibiotics": ["ampicillin"],
        "copy_number": "high",
        "features": ["MCS1", "MCS2", "His-tag", "S-tag"],
        "advantages": ["no_plasmid_incompatibility", "guaranteed_coexpression"],
        "description": "Duet vector for T7-based co-expression"
    },
    "pACYCDuet-1": {
        "type": "duet_vector",
        "promoters": 2,
        "origins": ["p15A"],
        "antibiotics": ["chloramphenicol"],
        "copy_number": "low",
        "features": ["MCS1", "MCS2"],
        "advantages": ["compatible_with_pET_vectors"],
        "description": "Low-copy Duet vector"
    },
    
    # Standard expression vectors
    "pET-28a(+)": {
        "type": "single_vector",
        "promoters": 1,
        "origins": ["pBR322"],
        "antibiotics": ["kanamycin"],
        "copy_number": "high",
        "features": ["T7_promoter", "His-tag", "thrombin_site"],
        "description": "Standard T7 expression vector"
    },
    "pET-15b": {
        "type": "single_vector",
        "promoters": 1,
        "origins": ["pBR322"],
        "antibiotics": ["ampicillin"],
        "copy_number": "high",
        "features": ["T7_promoter", "His-tag"],
        "description": "T7 expression vector with N-terminal His-tag"
    },
    "pCDF-1b": {
        "type": "single_vector",
        "promoters": 1,
        "origins": ["CDF"],
        "antibiotics": ["spectinomycin"],
        "copy_number": "medium",
        "features": ["T7_promoter"],
        "description": "Single-promoter CDF origin vector"
    },
    
    # Other vectors
    "pGEX-T4-1": {
        "type": "single_vector",
        "promoters": 1,
        "origins": ["pBR322"],
        "antibiotics": ["ampicillin"],
        "copy_number": "high",
        "features": ["GST_tag", "tac_promoter"],
        "description": "GST fusion vector"
    },
    "pASK-IBA3": {
        "type": "single_vector",
        "promoters": 1,
        "origins": ["pBR322"],
        "antibiotics": ["chloramphenicol"],
        "copy_number": "high",
        "features": ["tet_promoter", "Strep-tag"],
        "description": "Tet-regulated expression vector"
    },
    "pGEM-T": {
        "type": "cloning_vector",
        "promoters": 0,
        "origins": ["pBR322"],
        "antibiotics": ["ampicillin"],
        "copy_number": "high",
        "features": ["T7_SP6_promoters", "lacZ"],
        "description": "TA cloning vector"
    }
}

# Antibiotic compatibility matrix
ANTIBIOTIC_COMPATABILITY = {
    # Multiple antibiotics can be used together if they target different mechanisms
    "compatible_groups": [
        {"ampicillin", "spectinomycin"},  # Cell wall + protein synthesis
        {"kanamycin", "chloramphenicol"},  # Protein synthesis (different targets)
        {"spectinomycin", "kanamycin"},   # Both protein synthesis but different mechanisms
    ],
    "incompatible": [
        # Same mechanism antibiotics shouldn't be used together
        {"ampicillin"},  # Beta-lactam alone is fine
    ]
}

# Origin of replication compatibility
ORIGIN_COMPATIBILITY = {
    "pBR322": ["p15A", "CDF", "pSC101"],
    "p15A": ["pBR322", "CDF", "pSC101"],
    "CDF": ["pBR322", "p15A", "pSC101"],
    "pSC101": ["pBR322", "p15A", "CDF"],
}


class MCQOptionAnalyzer:
    """
    Deep analyzer for multiple choice options
    """
    
    def __init__(self, domain_knowledge: Optional[Dict] = None):
        self.domain_knowledge = domain_knowledge or {}
        self.vector_db = VECTOR_DATABASE
        
    def analyze_all_options(self, options: Dict[str, str], 
                           question_context: Optional[str] = None) -> Dict[str, OptionAnalysis]:
        """
        Analyze all options in a multiple choice question
        
        Args:
            options: Dict mapping option_id to option_text
            question_context: Optional context from the question
            
        Returns:
            Dict mapping option_id to OptionAnalysis
        """
        analyses = {}
        
        for option_id, option_text in options.items():
            analyses[option_id] = self.analyze_single_option(option_id, option_text, question_context)
        
        # Cross-validate options against each other
        self._cross_validate_options(analyses)
        
        return analyses
    
    def analyze_single_option(self, option_id: str, option_text: str,
                             question_context: Optional[str] = None) -> OptionAnalysis:
        """
        Analyze a single option in depth
        """
        analysis = OptionAnalysis(
            option_id=option_id,
            option_text=option_text
        )
        
        # Step 1: Extract entities
        analysis.entities = self._extract_entities(option_text)
        
        # Step 2: Extract relations
        analysis.relations = self._extract_relations(option_text, analysis.entities)
        
        # Step 3: Determine vector type
        analysis.vector_type = self._determine_vector_type(option_text, analysis.entities)
        
        # Step 4: Determine structure type
        analysis.structure_type = self._determine_structure_type(option_text, analysis)
        
        # Step 5: Extract keywords
        analysis.keywords = self._extract_keywords(option_text)
        
        # Step 6: Initial match status
        analysis.match_status = OptionMatchStatus.NEED_MORE_INFO
        
        return analysis
    
    def _extract_entities(self, text: str) -> List[EntityInfo]:
        """Extract named entities from option text"""
        entities = []
        
        # Extract vector names
        for vector_name in self.vector_db.keys():
            if vector_name.lower() in text.lower():
                entity = EntityInfo(
                    name=vector_name,
                    entity_type="vector",
                    attributes=self.vector_db[vector_name].copy()
                )
                entities.append(entity)
        
        # Extract antibiotics
        antibiotic_patterns = [
            (r'\b(ampicillin| Amp)\b', 'ampicillin'),
            (r'\b(kanamycin|Kan)\b', 'kanamycin'),
            (r'\b(spectinomycin|Spec|Spect)\b', 'spectinomycin'),
            (r'\b(chloramphenicol|Cam|Chlor)\b', 'chloramphenicol'),
            (r'\b(tetracycline|Tet)\b', 'tetracycline'),
        ]
        
        for pattern, abx_name in antibiotic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(EntityInfo(
                    name=abx_name,
                    entity_type="antibiotic"
                ))
        
        # Extract promoters
        promoter_patterns = [
            (r'\b(T7)\s*(?:promoter)?\b', 'T7_promoter'),
            (r'\b(T7\s*lac)\b', 'T7lac_promoter'),
            (r'\b(araBAD|arabinose)\b', 'arabinose_promoter'),
            (r'\b(lac)\b', 'lac_promoter'),
            (r'\b(tac)\b', 'tac_promoter'),
        ]
        
        for pattern, promoter_name in promoter_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(EntityInfo(
                    name=promoter_name,
                    entity_type="promoter"
                ))
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[EntityInfo]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities"""
        relations = []
        
        # Pattern for "X with Y" relations
        with_pattern = r'(\w+(?:-\w+)*\d*(?:\([+-]\))?)\s+with\s+(\w+(?:-\w+)*\d*(?:\([+-]\))?)'
        matches = re.findall(with_pattern, text, re.IGNORECASE)
        for m in matches:
            relations.append((m[0], "with", m[1]))
        
        # Pattern for "X and Y" relations
        and_pattern = r'(\w+(?:-\w+)*\d*(?:\([+-]\))?)\s+and\s+(\w+(?:-\w+)*\d*(?:\([+-]\))?)'
        matches = re.findall(and_pattern, text, re.IGNORECASE)
        for m in matches:
            relations.append((m[0], "and", m[1]))
        
        return relations
    
    def _determine_vector_type(self, text: str, entities: List[EntityInfo]) -> VectorType:
        """Determine the type of vector system in the option"""
        text_lower = text.lower()
        
        # Check for Duet vectors
        for entity in entities:
            if entity.entity_type == "vector":
                if self.vector_db.get(entity.name, {}).get("type") == "duet_vector":
                    return VectorType.DUET_VECTOR
        
        # Check for dual plasmid systems
        vector_count = len([e for e in entities if e.entity_type == "vector"])
        if vector_count >= 2:
            return VectorType.DUAL_PLASMID
        
        # Check for single vector
        if vector_count == 1:
            return VectorType.SINGLE_VECTOR
        
        # Check text patterns
        if "duet" in text_lower:
            return VectorType.DUET_VECTOR
        if " and " in text_lower and "resistance" in text_lower:
            # Multiple items with resistance markers likely dual plasmid
            return VectorType.DUAL_PLASMID
        
        return VectorType.UNKNOWN
    
    def _determine_structure_type(self, text: str, analysis: OptionAnalysis) -> str:
        """Determine the structural configuration described in the option"""
        if analysis.vector_type == VectorType.DUET_VECTOR:
            # Single Duet vector can express two genes
            return "single_duet_dual_expression"
        elif analysis.vector_type == VectorType.DUAL_PLASMID:
            # Two separate vectors
            return "dual_plasmid_system"
        elif analysis.vector_type == VectorType.SINGLE_VECTOR:
            return "single_vector_single_expression"
        
        # Analyze by entity count
        vector_entities = [e for e in analysis.entities if e.entity_type == "vector"]
        if len(vector_entities) == 1:
            # Check if it's a Duet
            if "Duet" in vector_entities[0].name:
                return "single_duet_dual_expression"
            return "single_vector_single_expression"
        
        return "unknown_structure"
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords"""
        keywords = set()
        
        # Technical terms
        tech_terms = [
            "co-expression", "expression", "vector", "plasmid", "promoter",
            "antibiotic", "resistance", "origin", "copy", "inducible",
            "constitutive", "fusion", "tag", "His", "GST", "MBP"
        ]
        
        text_lower = text.lower()
        for term in tech_terms:
            if term.lower() in text_lower:
                keywords.add(term)
        
        return keywords
    
    def _cross_validate_options(self, analyses: Dict[str, OptionAnalysis]):
        """Cross-validate options to identify better/worse choices"""
        # Group options by structure type
        by_structure = {}
        for opt_id, analysis in analyses.items():
            struct = analysis.structure_type
            if struct not in by_structure:
                by_structure[struct] = []
            by_structure[struct].append(opt_id)
        
        # For co-expression questions, Duet vectors are typically better than dual plasmid
        # Check if we have both types
        if "single_duet_dual_expression" in by_structure and "dual_plasmid_system" in by_structure:
            # Duet options are likely better for co-expression
            for opt_id in by_structure["single_duet_dual_expression"]:
                analyses[opt_id].match_status = OptionMatchStatus.MATCH
                analyses[opt_id].match_reason = "Duet vector provides guaranteed co-expression without plasmid compatibility issues"
                analyses[opt_id].confidence = 0.8
            
            # Dual plasmid options need more info
            for opt_id in by_structure["dual_plasmid_system"]:
                analyses[opt_id].match_status = OptionMatchStatus.PARTIAL_MATCH
                analyses[opt_id].match_reason = "Dual plasmid system works but requires compatibility verification"
                analyses[opt_id].confidence = 0.5
    
    def evaluate_for_coexpression(self, analyses: Dict[str, OptionAnalysis]) -> Dict[str, float]:
        """
        Evaluate options specifically for co-expression suitability
        
        Returns dict mapping option_id to suitability score (0-1)
        """
        scores = {}
        
        for opt_id, analysis in analyses.items():
            score = 0.0
            
            # Duet vectors are best for co-expression
            if analysis.vector_type == VectorType.DUET_VECTOR:
                score += 0.4
            
            # Single vector with dual promoters is good
            if "single_duet" in analysis.structure_type:
                score += 0.3
            
            # Check for compatible origins in dual plasmid
            if analysis.vector_type == VectorType.DUAL_PLASMID:
                vectors = [e for e in analysis.entities if e.entity_type == "vector"]
                if len(vectors) >= 2:
                    origins = [self.vector_db.get(v.name, {}).get("origins", []) for v in vectors]
                    # Check compatibility
                    if self._check_origin_compatibility(origins):
                        score += 0.2
                    else:
                        score -= 0.3  # Penalty for incompatible origins
            
            # Check antibiotic compatibility
            antibiotics = [e.name for e in analysis.entities if e.entity_type == "antibiotic"]
            if len(set(antibiotics)) >= 2:
                score += 0.1  # Multiple selection markers
            
            scores[opt_id] = max(0, min(1, score))
        
        return scores
    
    def _check_origin_compatibility(self, origins_list: List[List[str]]) -> bool:
        """Check if multiple origins are compatible"""
        if len(origins_list) < 2:
            return True
        
        for i, origins1 in enumerate(origins_list):
            for origins2 in origins_list[i+1:]:
                for o1 in origins1:
                    for o2 in origins2:
                        if o1 == o2:
                            return False  # Same origin is incompatible
                        if o2 not in ORIGIN_COMPATIBILITY.get(o1, []):
                            return False  # Not in compatibility list
        
        return True
    
    def generate_analysis_report(self, analyses: Dict[str, OptionAnalysis]) -> str:
        """Generate a human-readable analysis report"""
        lines = ["# MCQ Option Analysis Report\n"]
        
        lines.append("| Option | Vector Type | Structure | Entities | Match Status | Confidence |")
        lines.append("|--------|-------------|-----------|----------|--------------|------------|")
        
        for opt_id, analysis in analyses.items():
            entities_str = ", ".join([e.name for e in analysis.entities[:3]])
            if len(analysis.entities) > 3:
                entities_str += f" (+{len(analysis.entities)-3})"
            
            lines.append(f"| {opt_id} | {analysis.vector_type.value} | {analysis.structure_type} | {entities_str} | {analysis.match_status.value} | {analysis.confidence:.2f} |")
        
        lines.append("\n## Detailed Analysis\n")
        
        for opt_id, analysis in analyses.items():
            lines.append(f"\n### Option {opt_id}")
            lines.append(f"- **Text**: {analysis.option_text}")
            lines.append(f"- **Vector Type**: {analysis.vector_type.value}")
            lines.append(f"- **Structure**: {analysis.structure_type}")
            lines.append(f"- **Match Status**: {analysis.match_status.value}")
            if analysis.match_reason:
                lines.append(f"- **Reason**: {analysis.match_reason}")
            if analysis.entities:
                lines.append(f"- **Entities**: {', '.join([f'{e.name} ({e.entity_type})' for e in analysis.entities])}")
        
        return "\n".join(lines)


# Convenience function
def analyze_option_semantics(option_text: str) -> Dict[str, Any]:
    """
    Quick analysis function for a single option
    
    Returns structured analysis result
    """
    analyzer = MCQOptionAnalyzer()
    analysis = analyzer.analyze_single_option("temp", option_text)
    
    return {
        "entities": [{"name": e.name, "type": e.entity_type, "attributes": e.attributes} 
                     for e in analysis.entities],
        "relations": analysis.relations,
        "vector_type": analysis.vector_type.value,
        "structure_type": analysis.structure_type,
        "keywords": list(analysis.keywords),
        "is_duet_vector": analysis.vector_type == VectorType.DUET_VECTOR,
        "is_dual_plasmid": analysis.vector_type == VectorType.DUAL_PLASMID
    }
