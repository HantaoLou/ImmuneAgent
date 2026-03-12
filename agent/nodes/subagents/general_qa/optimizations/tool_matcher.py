"""
Tool-Question Matcher - P0 Priority Optimization

Pre-evaluates tool relevance before calling them:
1. Extract keywords from question
2. Match against tool capabilities
3. Rank tools by relevance
4. Skip irrelevant tools to save time
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum


class QuestionDomain(Enum):
    """Question domain categories"""
    MOLECULAR_CLONING = "molecular_cloning"
    PROTEIN_EXPRESSION = "protein_expression"
    GENETICS = "genetics"
    POPULATION_GENETICS = "population_genetics"
    IMMUNOLOGY = "immunology"
    PHARMACOLOGY = "pharmacology"
    CLINICAL_MEDICINE = "clinical_medicine"
    BIOINFORMATICS = "bioinformatics"
    STRUCTURAL_BIOLOGY = "structural_biology"
    CELL_BIOLOGY = "cell_biology"
    UNKNOWN = "unknown"


class ToolCategory(Enum):
    """Tool capability categories"""
    GENE_DISEASE = "gene_disease"          # Disease-gene associations
    PROTEIN_INFO = "protein_info"          # Protein information
    PROTEIN_INTERACTION = "protein_interaction"  # PPI networks
    SEQUENCE = "sequence"                  # Sequence analysis
    PATHWAY = "pathway"                    # Pathway analysis
    EXPRESSION = "expression"              # Expression data
    VARIANT = "variant"                    # Genetic variants
    ONTOLOGY = "ontology"                  # GO terms, ontologies
    LITERATURE = "literature"              # Paper search
    VECTOR = "vector"                      # Vector/plasmid info
    GENERAL = "general"                    # General purpose


@dataclass
class ToolInfo:
    """Information about a tool's capabilities"""
    name: str
    category: ToolCategory
    keywords: Set[str]
    domains: Set[QuestionDomain]
    description: str = ""
    
    def matches_keywords(self, text: str, threshold: float = 0.3) -> float:
        """Calculate keyword match score"""
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        
        if not self.keywords:
            return 0.0
        
        matching = self.keywords & text_words
        return len(matching) / len(self.keywords)


# Tool capability definitions
TOOL_DEFINITIONS: Dict[str, ToolInfo] = {
    # Gene-Disease tools
    "query_disgenet": ToolInfo(
        name="query_disgenet",
        category=ToolCategory.GENE_DISEASE,
        keywords={"disease", "gene", "association", "variant", "snp", "mutation", "phenotype", "morbid"},
        domains={QuestionDomain.GENETICS, QuestionDomain.CLINICAL_MEDICINE},
        description="Query disease-gene associations from DisGeNET"
    ),
    "query_omim": ToolInfo(
        name="query_omim",
        category=ToolCategory.GENE_DISEASE,
        keywords={"disease", "gene", "inheritance", "genetic", "disorder", "syndrome", "phenotype"},
        domains={QuestionDomain.GENETICS, QuestionDomain.CLINICAL_MEDICINE},
        description="Query OMIM database for genetic disorders"
    ),
    
    # Protein tools
    "query_proteinatlas": ToolInfo(
        name="query_proteinatlas",
        category=ToolCategory.PROTEIN_INFO,
        keywords={"protein", "expression", "tissue", "cell", "localization", "antibody", "atlas"},
        domains={QuestionDomain.CELL_BIOLOGY, QuestionDomain.PROTEIN_EXPRESSION},
        description="Query Human Protein Atlas for protein information"
    ),
    "query_uniprot": ToolInfo(
        name="query_uniprot",
        category=ToolCategory.PROTEIN_INFO,
        keywords={"protein", "sequence", "function", "domain", "uniprot", "annotation"},
        domains={QuestionDomain.PROTEIN_EXPRESSION, QuestionDomain.STRUCTURAL_BIOLOGY},
        description="Query UniProt for protein sequences and annotations"
    ),
    
    # PPI tools
    "query_ppi": ToolInfo(
        name="query_ppi",
        category=ToolCategory.PROTEIN_INTERACTION,
        keywords={"interaction", "binding", "complex", "partner", "ppi", "network", "association"},
        domains={QuestionDomain.PROTEIN_EXPRESSION, QuestionDomain.CELL_BIOLOGY},
        description="Query protein-protein interaction databases"
    ),
    "query_string": ToolInfo(
        name="query_string",
        category=ToolCategory.PROTEIN_INTERACTION,
        keywords={"interaction", "string", "network", "pathway", "functional", "association"},
        domains={QuestionDomain.PROTEIN_EXPRESSION, QuestionDomain.BIOINFORMATICS},
        description="Query STRING database for protein interactions"
    ),
    
    # Ontology tools
    "query_go_term": ToolInfo(
        name="query_go_term",
        category=ToolCategory.ONTOLOGY,
        keywords={"go", "ontology", "function", "process", "component", "annotation", "biological"},
        domains={QuestionDomain.BIOINFORMATICS, QuestionDomain.CELL_BIOLOGY},
        description="Query Gene Ontology terms"
    ),
    "query_go_hierarchy": ToolInfo(
        name="query_go_hierarchy",
        category=ToolCategory.ONTOLOGY,
        keywords={"hierarchy", "parent", "child", "ancestor", "descendant", "go", "ontology"},
        domains={QuestionDomain.BIOINFORMATICS},
        description="Query GO hierarchy relationships"
    ),
    
    # Knowledge graph
    "query_knowledge_graph": ToolInfo(
        name="query_knowledge_graph",
        category=ToolCategory.GENERAL,
        keywords={"entity", "relation", "knowledge", "graph", "triple", "association"},
        domains={QuestionDomain.GENETICS, QuestionDomain.CLINICAL_MEDICINE, QuestionDomain.BIOINFORMATICS},
        description="Query general knowledge graph"
    ),
}

# Domain keyword patterns for automatic domain detection
DOMAIN_KEYWORDS = {
    QuestionDomain.MOLECULAR_CLONING: [
        "vector", "plasmid", "clone", "cloneing", "restriction", "ligation",
        "insert", "transform", "competent", "antibiotic", "resistance",
        "promoter", "origin", "copy", "duet", "pet", "pcdf"
    ],
    QuestionDomain.PROTEIN_EXPRESSION: [
        "express", "expression", "protein", "purify", "purification",
        "tag", "his-tag", "gst", "mbp", "fusion", "soluble", "inclusion",
        "chaperone", "fold", "folding", "e.coli", "bl21"
    ],
    QuestionDomain.POPULATION_GENETICS: [
        "population", "allele", "frequency", "drift", "selection",
        "mutation rate", "fixation", "polymorphism", "genetic variation",
        "hardy-weinberg", "founder", "bottleneck"
    ],
    QuestionDomain.GENETICS: [
        "gene", "genetic", "mutation", "variant", "snp", "genotype",
        "phenotype", "inheritance", "dominant", "recessive", "locus"
    ],
    QuestionDomain.IMMUNOLOGY: [
        "immune", "antibody", "antigen", "t cell", "b cell", "cytokine",
        "inflammation", "autoimmune", "vaccine", "mhc", "receptor"
    ],
    QuestionDomain.PHARMACOLOGY: [
        "drug", "dose", "pharmacokinet", "pharmacodynam", "metabolism",
        "adverse", "toxicity", "therapeutic", "bud", "stability", "usp"
    ],
    QuestionDomain.CLINICAL_MEDICINE: [
        "patient", "clinical", "diagnosis", "treatment", "therapy",
        "symptom", "disease", "prognosis", "efficacy", "safety"
    ],
    QuestionDomain.BIOINFORMATICS: [
        "sequence", "alignment", "blast", "annotation", "pipeline",
        "omics", "transcriptome", "genome", "bioinformatics"
    ],
    QuestionDomain.STRUCTURAL_BIOLOGY: [
        "structure", "crystal", "nmr", "cryo-em", "folding", "domain",
        "secondary", "tertiary", "quaternary", "pdb"
    ],
    QuestionDomain.CELL_BIOLOGY: [
        "cell", "membrane", "organelle", "cytoplasm", "nucleus",
        "mitochondria", "endoplasmic", "golgi", "vesicle"
    ],
}


class ToolQuestionMatcher:
    """
    Matches tools to questions based on domain and keyword relevance
    """
    
    def __init__(self, tool_definitions: Optional[Dict[str, ToolInfo]] = None):
        self.tool_definitions = tool_definitions or TOOL_DEFINITIONS
        self.domain_keywords = DOMAIN_KEYWORDS
        
    def detect_question_domains(self, question_text: str) -> List[Tuple[QuestionDomain, float]]:
        """
        Detect which domains a question belongs to
        
        Returns list of (domain, confidence) tuples sorted by confidence
        """
        text_lower = question_text.lower()
        domain_scores = []
        
        for domain, keywords in self.domain_keywords.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                # Score is normalized by total keywords in domain
                score = matches / len(keywords)
                domain_scores.append((domain, score))
        
        # Sort by score descending
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        return domain_scores
    
    def evaluate_tool_relevance(self, tool_name: str, question_text: str,
                                detected_domains: Optional[List[QuestionDomain]] = None) -> float:
        """
        Evaluate how relevant a tool is for a given question
        
        Returns relevance score between 0 and 1
        """
        if tool_name not in self.tool_definitions:
            return 0.0
        
        tool = self.tool_definitions[tool_name]
        
        # Calculate keyword match score
        keyword_score = tool.matches_keywords(question_text)
        
        # Calculate domain match score
        if detected_domains is None:
            detected_domains = [d for d, _ in self.detect_question_domains(question_text)]
        
        domain_score = 0.0
        if detected_domains:
            matching_domains = tool.domains & set(detected_domains)
            domain_score = len(matching_domains) / len(detected_domains)
        
        # Combined score (weighted average)
        combined_score = 0.4 * keyword_score + 0.6 * domain_score
        
        return combined_score
    
    def rank_tools_by_relevance(self, available_tools: List[str], 
                                question_text: str) -> List[Tuple[str, float]]:
        """
        Rank all available tools by their relevance to the question
        
        Returns list of (tool_name, relevance_score) sorted by score
        """
        detected_domains = [d for d, _ in self.detect_question_domains(question_text)]
        
        tool_scores = []
        for tool_name in available_tools:
            score = self.evaluate_tool_relevance(tool_name, question_text, detected_domains)
            tool_scores.append((tool_name, score))
        
        # Sort by score descending
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores
    
    def select_relevant_tools(self, available_tools: List[str],
                              question_text: str,
                              threshold: float = 0.1,
                              max_tools: int = 10) -> List[str]:
        """
        Select only the most relevant tools for a question
        
        Args:
            available_tools: List of available tool names
            question_text: The question to match against
            threshold: Minimum relevance score to include tool
            max_tools: Maximum number of tools to return
            
        Returns:
            List of tool names sorted by relevance
        """
        ranked = self.rank_tools_by_relevance(available_tools, question_text)
        
        # Filter by threshold and limit count
        selected = [name for name, score in ranked if score >= threshold]
        
        if len(selected) > max_tools:
            selected = selected[:max_tools]
        
        return selected
    
    def get_tool_selection_report(self, available_tools: List[str],
                                   question_text: str) -> str:
        """Generate a human-readable tool selection report"""
        detected_domains = self.detect_question_domains(question_text)
        ranked_tools = self.rank_tools_by_relevance(available_tools, question_text)
        
        lines = ["# Tool Selection Report\n"]
        
        lines.append("## Detected Domains")
        for domain, score in detected_domains[:5]:
            lines.append(f"- {domain.value}: {score:.2f}")
        
        lines.append("\n## Tool Relevance Ranking")
        lines.append("| Tool | Category | Score | Selected |")
        lines.append("|------|----------|-------|----------|")
        
        for tool_name, score in ranked_tools:
            tool = self.tool_definitions.get(tool_name)
            category = tool.category.value if tool else "unknown"
            selected = "[OK]" if score >= 0.1 else ""
            lines.append(f"| {tool_name} | {category} | {score:.3f} | {selected} |")
        
        return "\n".join(lines)
    
    def should_skip_tool(self, tool_name: str, question_text: str) -> Tuple[bool, str]:
        """
        Determine if a tool should be skipped for a given question
        
        Returns (should_skip, reason)
        """
        if tool_name not in self.tool_definitions:
            return True, "Tool not in definitions"
        
        tool = self.tool_definitions[tool_name]
        detected_domains = [d for d, _ in self.detect_question_domains(question_text)]
        
        # Check if tool's domains overlap with question domains
        domain_overlap = tool.domains & set(detected_domains)
        
        if not domain_overlap:
            return True, f"Tool domains {tool.domains} don't match question domains {set(detected_domains)}"
        
        # Check keyword relevance
        keyword_score = tool.matches_keywords(question_text)
        if keyword_score < 0.05:
            return True, f"Low keyword relevance ({keyword_score:.3f})"
        
        return False, "Tool appears relevant"


# Convenience functions

def pre_evaluate_tool_relevance(question_text: str, 
                                available_tools: List[str],
                                threshold: float = 0.1) -> List[str]:
    """
    Quick function to get relevant tools for a question
    
    Args:
        question_text: The question to analyze
        available_tools: List of available tool names
        threshold: Minimum relevance score
        
    Returns:
        List of relevant tool names
    """
    matcher = ToolQuestionMatcher()
    return matcher.select_relevant_tools(available_tools, question_text, threshold)


def get_question_domains(question_text: str) -> List[str]:
    """
    Quick function to detect question domains
    
    Returns list of domain names as strings
    """
    matcher = ToolQuestionMatcher()
    domains = matcher.detect_question_domains(question_text)
    return [d.value for d, _ in domains]


def filter_tools_by_domain(available_tools: List[str], 
                           target_domains: List[str]) -> List[str]:
    """
    Filter tools that are relevant for target domains
    
    Args:
        available_tools: List of available tool names
        target_domains: List of domain names to match
        
    Returns:
        List of tools relevant to the target domains
    """
    matcher = ToolQuestionMatcher()
    
    # Convert string domain names to enum
    domain_enums = set()
    for d in target_domains:
        try:
            domain_enums.add(QuestionDomain(d))
        except ValueError:
            pass
    
    relevant_tools = []
    for tool_name in available_tools:
        if tool_name in matcher.tool_definitions:
            tool = matcher.tool_definitions[tool_name]
            if tool.domains & domain_enums:
                relevant_tools.append(tool_name)
    
    return relevant_tools
