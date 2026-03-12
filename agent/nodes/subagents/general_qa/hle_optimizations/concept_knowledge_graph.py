"""
Concept Knowledge Graph for HLE

Provides a knowledge graph for biology concepts with:
- Concept definitions and relationships
- Contrast relationships for easily confused concepts
- Quick lookup for concept clarification
- Integration with reasoning templates

Key Features:
- ConceptRelation: Types of relationships between concepts
- ConceptContrast: Pairs of commonly confused concepts
- ConceptKnowledgeGraph: Main graph structure with queries
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class ConceptRelation(Enum):
    """Types of relationships between concepts"""
    IS_A = "is_a"                   # Inheritance (mRNA IS_A RNA)
    PART_OF = "part_of"            # Composition (promoter PART_OF gene)
    REGULATES = "regulates"         # Regulatory (transcription_factor REGULATES gene)
    CONVERTS_TO = "converts_to"    # Metabolic (glucose CONVERTS_TO pyruvate)
    PRECEDES = "precedes"          # Sequential (transcription PRECEDES translation)
    CONTRADICTS = "contradicts"    # Mutually exclusive
    CONFUSED_WITH = "confused_with"  # Commonly confused
    EXAMPLE_OF = "example_of"      # Instance (p53 EXAMPLE_OF tumor_suppressor)
    CAUSES = "causes"              # Causal (mutation CAUSES disease)


@dataclass
class ConceptNode:
    """A node in the concept knowledge graph"""
    name: str
    definition: str
    aliases: List[str] = field(default_factory=list)
    related_concepts: Dict[str, ConceptRelation] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    common_errors: List[str] = field(default_factory=list)
    
    def add_relation(self, other_concept: str, relation: ConceptRelation):
        """Add a relationship to another concept"""
        self.related_concepts[other_concept] = relation
    
    def get_related_by_type(self, relation_type: ConceptRelation) -> List[str]:
        """Get all concepts related by a specific type"""
        return [
            concept for concept, rel in self.related_concepts.items()
            if rel == relation_type
        ]


@dataclass
class ConceptContrast:
    """
    A contrast between two commonly confused concepts.
    
    This is crucial for avoiding conceptual errors in HLE.
    """
    concept_a: str
    concept_b: str
    key_difference: str
    formula_a: Optional[str] = None
    formula_b: Optional[str] = None
    example_a: Optional[str] = None
    example_b: Optional[str] = None
    mnemonic: Optional[str] = None
    trick_indicator: Optional[str] = None  # Hint that this might be tested
    
    def get_comparison_text(self) -> str:
        """Generate comparison text for prompts"""
        text = f"""
## {self.concept_a} vs {self.concept_b}

**Key Difference:** {self.key_difference}
"""
        if self.formula_a and self.formula_b:
            text += f"""
**{self.concept_a}:** {self.formula_a}
**{self.concept_b}:** {self.formula_b}
"""
        if self.example_a and self.example_b:
            text += f"""
**Example:**
- {self.concept_a}: {self.example_a}
- {self.concept_b}: {self.example_b}
"""
        if self.mnemonic:
            text += f"\n**Memory Aid:** {self.mnemonic}\n"
        
        if self.trick_indicator:
            text += f"\n**[WARN]️ Watch for:** {self.trick_indicator}\n"
        
        return text


class ConceptKnowledgeGraph:
    """
    Knowledge graph for biology concepts.
    
    Provides quick lookup and traversal for concept relationships,
    with special support for commonly confused concept pairs.
    """
    
    def __init__(self):
        self.concepts: Dict[str, ConceptNode] = {}
        self.contrasts: List[ConceptContrast] = []
        self._initialize_default_concepts()
    
    def _initialize_default_concepts(self):
        """Initialize with common biology concepts"""
        
        # Genetics concepts
        self._add_genetics_concepts()
        
        # Molecular biology concepts
        self._add_molecular_biology_concepts()
        
        # Biochemistry concepts
        self._add_biochemistry_concepts()
        
        # Clinical concepts
        self._add_clinical_concepts()
        
        # Add contrasts
        self._add_default_contrasts()
    
    def _add_genetics_concepts(self):
        """Add genetics-related concepts"""
        
        # Heritability concepts
        self.add_concept(ConceptNode(
            name="H²",
            definition="Broad-sense heritability: proportion of phenotypic variance due to ALL genetic factors",
            aliases=["broad-sense heritability", "H-squared"],
            properties={
                "formula": "H² = Vg / Vp",
                "components": ["additive variance (Va)", "dominance variance (Vd)", "epistatic variance (Vi)"],
                "range": "[0, 1]",
                "use": "Explains how much genetics contributes to phenotype"
            },
            examples=["If H² = 0.8, 80% of phenotypic variation is due to genetic differences"],
            common_errors=[
                "Confusing with narrow-sense heritability (h²)",
                "Using for breeding prediction (use h² instead)"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="h²",
            definition="Narrow-sense heritability: proportion of phenotypic variance due to ADDITIVE genetic effects only",
            aliases=["narrow-sense heritability", "h-squared"],
            properties={
                "formula": "h² = Va / Vp",
                "components": ["additive variance (Va) only"],
                "range": "[0, H²]",
                "use": "Predicts response to selection in breeding"
            },
            examples=["If h² = 0.3, selection will be 30% effective"],
            common_errors=[
                "Confusing with broad-sense heritability (H²)",
                "Not realizing h² ≤ H² always"
            ]
        ))
        
        # Genetic drift vs selection
        self.add_concept(ConceptNode(
            name="genetic_drift",
            definition="Random changes in allele frequencies due to sampling error, strongest in small populations",
            aliases=["drift", "random drift"],
            properties={
                "direction": "random",
                "strength_factor": "population size (stronger in small populations)",
                "effect_on_neutral_alleles": "can lead to fixation or loss"
            },
            examples=[
                "Neutral allele reaching 100% frequency in small isolated population",
                "Founder effect in island populations"
            ],
            common_errors=[
                "Attributing random changes to selection",
                "Not recognizing drift in small populations"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="natural_selection",
            definition="Differential survival and reproduction based on heritable traits",
            aliases=["selection", "positive selection"],
            properties={
                "direction": "toward higher fitness",
                "strength_factor": "selection coefficient",
                "effect_on_alleles": "increases frequency of beneficial alleles"
            },
            examples=[
                "Antibiotic resistance in bacteria",
                "Peppered moth coloration changes"
            ],
            common_errors=[
                "Confusing with genetic drift",
                "Assuming all changes are adaptive"
            ]
        ))
        
        # Linkage concepts
        self.add_concept(ConceptNode(
            name="genetic_linkage",
            definition="Tendency of genes on the same chromosome to be inherited together",
            aliases=["linkage"],
            properties={
                "cause": "physical proximity on chromosome",
                "measure": "recombination frequency (cM)",
                "effect": "deviation from independent assortment"
            },
            examples=[
                "Genes 5 cM apart show 5% recombination",
                "Linked genes in same haplotype"
            ],
            common_errors=[
                "Assuming all genes assort independently",
                "Not accounting for linkage in dihybrid crosses"
            ]
        ))
        
        # Population genetics concepts
        self.add_concept(ConceptNode(
            name="Fst",
            definition="Fixation index: proportion of genetic variance due to differences between populations",
            aliases=["fixation index", "F_ST", "F-statistics"],
            properties={
                "formula": "Fst = (Ht - Hs) / Ht",
                "range": "[0, 1]",
                "interpretation": {
                    "0": "no differentiation (panmixia)",
                    "0.05-0.15": "moderate differentiation",
                    "0.15-0.25": "great differentiation",
                    ">0.25": "very great differentiation"
                }
            },
            examples=[
                "Fst = 0.15 between mainland and island populations",
                "High Fst indicates limited gene flow"
            ],
            common_errors=[
                "Confusing with genetic distance (Dxy)",
                "Interpreting as absolute measure of divergence"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="Dxy",
            definition="Average number of nucleotide differences between sequences from two populations",
            aliases=["genetic distance", "nucleotide divergence"],
            properties={
                "formula": "Dxy = Σ(xy) / L",
                "units": "substitutions per site",
                "use": "Measures absolute divergence between populations/species"
            },
            examples=[
                "Dxy = 0.02 means 2% sequence divergence",
                "Higher Dxy indicates longer separation time"
            ],
            common_errors=[
                "Confusing with Fst (relative vs absolute)",
                "Not accounting for ancestral polymorphism"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="hybrid_zone",
            definition="Region where two distinct populations or species meet and interbreed",
            aliases=["cline zone", "contact zone"],
            properties={
                "characteristics": ["cline width", "cline shape", "selection pressure"],
                "fate": ["fusion", "stability", "reinforcement", "extinction of hybrids"],
                "key_parameters": ["dispersal distance", "selection strength", "hybrid fitness"]
            },
            examples=[
                "European crow hybrid zone (hooded vs carrion)",
                "Bombina toad hybrid zone"
            ],
            common_errors=[
                "Assuming hybrid zones always lead to fusion",
                "Ignoring selection against hybrids"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="cline",
            definition="Gradient in a trait or allele frequency across a geographic area",
            aliases=["geographic gradient", "frequency gradient"],
            properties={
                "width": "related to dispersal distance and selection strength",
                "shape": ["step", "sigmoid", "complex"],
                "formation_cause": "balance between gene flow and selection"
            },
            examples=[
                "Temperature adaptation cline in Drosophila",
                "Melanism cline in peppered moths"
            ],
            common_errors=[
                "Assuming clines always indicate selection",
                "Ignoring historical demography"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="gene_flow",
            definition="Transfer of genetic material between populations through migration",
            aliases=["migration", "gene migration"],
            properties={
                "effect_on_divergence": "reduces differentiation (homogenizing)",
                "rate_parameter": "m (migration rate per generation)",
                "balance": "counteracts divergence from drift and selection"
            },
            examples=[
                "Human gene flow between continents",
                "Pollen-mediated gene flow in plants"
            ],
            common_errors=[
                "Assuming gene flow prevents all divergence",
                "Not considering asymmetric gene flow"
            ]
        ))
    
    def _add_molecular_biology_concepts(self):
        """Add molecular biology concepts"""
        
        # trp operon
        self.add_concept(ConceptNode(
            name="trp_operon",
            definition="Bacterial operon for tryptophan biosynthesis, regulated by repression and attenuation",
            aliases=["tryptophan operon"],
            properties={
                "components": ["promoter", "operator", "leader (trpL)", "structural genes (trpE, trpD, trpC, trpB, trpA)"],
                "regulation_levels": 2,
                "repressor": "TrpR (activated by tryptophan)",
                "attenuation": "leader peptide stalling mechanism"
            },
            examples=[
                "High Trp: repression + attenuation both active",
                "Low Trp: attenuation allows transcription"
            ],
            common_errors=[
                "Only considering TrpR repression",
                "Missing the RNA-level attenuation mechanism"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="attenuation",
            definition="Transcriptional regulation via premature termination, based on translation of leader peptide",
            aliases=["transcriptional attenuation"],
            properties={
                "mechanism": "ribosome stalling affects RNA secondary structure",
                "key_structure": "3-4 stem loop (terminator) vs 2-3 stem loop (anti-terminator)",
                "sensing": "aminoacyl-tRNA availability"
            },
            examples=[
                "trp operon attenuation",
                "his operon attenuation"
            ],
            common_errors=[
                "Confusing with repression",
                "Not understanding the leader peptide role"
            ]
        ))
        
        # Regulation concepts
        self.add_concept(ConceptNode(
            name="allosteric_regulation",
            definition="Modulation of enzyme activity by binding of effector at site distinct from active site",
            aliases=["allostery", "allosteric control"],
            properties={
                "binding_site": "allosteric site (different from active site)",
                "reversibility": "reversible",
                "cooperativity": "often exhibits positive cooperativity"
            },
            examples=[
                "ATP inhibition of phosphofructokinase",
                "Oxygen binding to hemoglobin"
            ],
            common_errors=[
                "Confusing with covalent modification",
                "Thinking effector binds at active site"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="feedback_inhibition",
            definition="Inhibition of an early enzyme in a pathway by the end product",
            aliases=["end-product inhibition"],
            properties={
                "direction": "backward from product to early enzyme",
                "purpose": "homeostasis, prevent overproduction",
                "mechanism": "usually allosteric"
            },
            examples=[
                "CTP inhibition of ATCase in pyrimidine synthesis",
                "Cholesterol inhibition of HMG-CoA reductase"
            ],
            common_errors=[
                "Confusing with transcriptional repression",
                "Not recognizing as distinct from competitive inhibition"
            ]
        ))
    
    def _add_biochemistry_concepts(self):
        """Add biochemistry concepts"""
        
        self.add_concept(ConceptNode(
            name="enzyme_kinetics",
            definition="Study of reaction rates catalyzed by enzymes",
            aliases=["Michaelis-Menten kinetics"],
            properties={
                "key_parameters": ["Km (Michaelis constant)", "Vmax (maximum velocity)", "kcat (turnover number)"],
                "Km_meaning": "substrate concentration at half Vmax (affinity measure)",
                "Vmax_meaning": "maximum rate when all enzyme bound"
            },
            examples=[
                "Km = 10 µM means low substrate affinity",
                "Vmax increases with enzyme concentration"
            ],
            common_errors=[
                "Confusing Km with Kd",
                "Not recognizing competitive vs non-competitive inhibition effects"
            ]
        ))
        
        self.add_concept(ConceptNode(
            name="competitive_inhibition",
            definition="Inhibitor binds at active site, competes with substrate",
            aliases=["competitive"],
            properties={
                "binding_site": "active site",
                "effect_on_Km": "increases",
                "effect_on_Vmax": "no change",
                "reversibility": "reversible by increasing substrate"
            },
            examples=["Malonate inhibition of succinate dehydrogenase"],
            common_errors=["Confusing with non-competitive inhibition"]
        ))
        
        self.add_concept(ConceptNode(
            name="noncompetitive_inhibition",
            definition="Inhibitor binds at allosteric site, reduces enzyme activity regardless of substrate",
            aliases=["non-competitive", "allosteric inhibition"],
            properties={
                "binding_site": "allosteric site",
                "effect_on_Km": "no change",
                "effect_on_Vmax": "decreases",
                "reversibility": "cannot be overcome by more substrate"
            },
            examples=["Heavy metal inhibition of enzymes"],
            common_errors=["Confusing with competitive inhibition"]
        ))
    
    def _add_clinical_concepts(self):
        """Add clinical/medical concepts"""
        
        self.add_concept(ConceptNode(
            name="sensitivity",
            definition="True positive rate: proportion of actual positives correctly identified",
            aliases=["true positive rate", "recall"],
            properties={
                "formula": "TP / (TP + FN)",
                "use": "screening tests (minimize false negatives)",
                "interpretation": "high sensitivity = few false negatives"
            },
            examples=[
                "Test with 99% sensitivity catches 99% of diseased individuals",
                "SNOUT: Sensitive test when Negative rules OUT"
            ],
            common_errors=["Confusing with specificity", "Using for confirmation"]
        ))
        
        self.add_concept(ConceptNode(
            name="specificity",
            definition="True negative rate: proportion of actual negatives correctly identified",
            aliases=["true negative rate"],
            properties={
                "formula": "TN / (TN + FP)",
                "use": "confirmatory tests (minimize false positives)",
                "interpretation": "high specificity = few false positives"
            },
            examples=[
                "Test with 99% specificity correctly excludes 99% of healthy individuals",
                "SPIN: Specific test when Positive rules IN"
            ],
            common_errors=["Confusing with sensitivity", "Using for screening"]
        ))
    
    def _add_default_contrasts(self):
        """Add commonly confused concept pairs"""
        
        # Heritability contrast
        self.add_contrast(ConceptContrast(
            concept_a="H²",
            concept_b="h²",
            key_difference="H² includes ALL genetic variance (additive + dominance + epistatic), h² includes ONLY additive variance",
            formula_a="H² = Vg/Vp = (Va + Vd + Vi) / Vp",
            formula_b="h² = Va/Vp",
            example_a="H² = 0.8 means 80% of phenotypic variance is genetic (all types)",
            example_b="h² = 0.4 means 40% of phenotypic variance is additive genetic",
            mnemonic="H=Huge (broad), h=handful (narrow/additive only)",
            trick_indicator="Questions about 'response to selection' or 'breeding effectiveness' → use h²"
        ))
        
        # Drift vs Selection contrast
        self.add_contrast(ConceptContrast(
            concept_a="genetic_drift",
            concept_b="natural_selection",
            key_difference="Drift is RANDOM (no fitness effect), selection is DIRECTIONAL (based on fitness)",
            example_a="Neutral allele fixation in small population",
            example_b="Beneficial allele increasing in frequency",
            mnemonic="Drift=Dice (random), Selection=Survival (fitness-based)",
            trick_indicator="Small population + neutral allele → think drift first"
        ))
        
        # Competitive vs Non-competitive inhibition
        self.add_contrast(ConceptContrast(
            concept_a="competitive_inhibition",
            concept_b="noncompetitive_inhibition",
            key_difference="Competitive: binds active site, increases Km, Vmax unchanged. Non-competitive: binds elsewhere, Km unchanged, decreases Vmax",
            formula_a="Effect: Km↑, Vmax unchanged",
            formula_b="Effect: Km unchanged, Vmax↓",
            example_a="Drug competing with substrate at active site",
            example_b="Heavy metal binding to allosteric site",
            mnemonic="Competitive = Competes for same site; Non-competitive = Not competing",
            trick_indicator="Lineweaver-Burk plot: competitive lines intersect at Y-axis"
        ))
        
        # Sensitivity vs Specificity
        self.add_contrast(ConceptContrast(
            concept_a="sensitivity",
            concept_b="specificity",
            key_difference="Sensitivity = catching sick people (TP rate). Specificity = excluding healthy people (TN rate)",
            formula_a="Sensitivity = TP/(TP+FN)",
            formula_b="Specificity = TN/(TN+FP)",
            example_a="High sensitivity test for screening (few false negatives)",
            example_b="High specificity test for confirmation (few false positives)",
            mnemonic="SNOUT (Sensitive Negative rules OUT), SPIN (Specific Positive rules IN)",
            trick_indicator="Screening → sensitivity; Confirmation → specificity"
        ))
        
        # Attenuation vs Repression
        self.add_contrast(ConceptContrast(
            concept_a="repression",
            concept_b="attenuation",
            key_difference="Repression prevents transcription initiation. Attenuation causes premature termination during transcription",
            example_a="TrpR repressor blocks RNA polymerase binding",
            example_b="Leader peptide stalling allows anti-terminator formation",
            mnemonic="Repression = door closed; Attenuation = start walking but stop early",
            trick_indicator="trp operon questions often test understanding of BOTH mechanisms"
        ))
        
        # Fst vs Dxy
        self.add_contrast(ConceptContrast(
            concept_a="Fst",
            concept_b="Dxy",
            key_difference="Fst measures RELATIVE differentiation (proportion of variance between populations). Dxy measures ABSOLUTE divergence (actual sequence differences)",
            formula_a="Fst = (Ht - Hs) / Ht (relative, 0-1)",
            formula_b="Dxy = Σ(xy) / L (absolute, substitutions/site)",
            example_a="Fst = 0.15 means 15% of variation is between populations",
            example_b="Dxy = 0.02 means 2% sequence divergence",
            mnemonic="Fst = Fraction (relative), Dxy = Distance (absolute)",
            trick_indicator="Population structure → Fst; Speciation time → Dxy"
        ))
        
        # Gene flow vs Genetic drift
        self.add_contrast(ConceptContrast(
            concept_a="gene_flow",
            concept_b="genetic_drift",
            key_difference="Gene flow HOMOGENIZES populations (reduces differences). Drift DIFFERENTIATES populations randomly (increases differences)",
            example_a="Migration between populations reduces Fst",
            example_b="Small isolated population diverges due to random drift",
            mnemonic="Flow = Fusion (brings together); Drift = Divergence (pushes apart)",
            trick_indicator="High Fst suggests limited gene flow + strong drift"
        ))
        
        # Hybrid zone width vs Selection strength
        self.add_contrast(ConceptContrast(
            concept_a="cline_width",
            concept_b="selection_strength",
            key_difference="Stronger selection → narrower cline. Higher dispersal → wider cline. Width ∝ √(dispersal²/selection)",
            example_a="Wide cline = weak selection or high dispersal",
            example_b="Narrow cline = strong selection or low dispersal",
            mnemonic="Strong selection squeezes the zone narrow",
            trick_indicator="Questions about cline width need both dispersal and selection info"
        ))
        
        # Coiled-coil oligomeric states
        self.add_contrast(ConceptContrast(
            concept_a="coiled_coil_dimer",
            concept_b="coiled_coil_trimer",
            key_difference="a/d position residues determine oligomeric state: Small residues (Ala, Asn) favor dimers; Large residues (Leu, Ile) favor trimers/tetramers",
            example_a="a-position = Ala-rich → dimer",
            example_b="a-position = Leu-rich → trimer/tetramer",
            mnemonic="A = Ala = Always 2 (dimer); L = Leu = Large number (3+)",
            trick_indicator="Check a and d positions for oligomeric state prediction"
        ))
        
        # Linear phase issues
        self.add_contrast(ConceptContrast(
            concept_a="substrate_depletion",
            concept_b="product_inhibition",
            key_difference="Substrate depletion: nonlinear because [S] decreases → DECREASE enzyme. Product inhibition: nonlinear because [P] increases → REMOVE product",
            example_a="Decrease enzyme to extend linear phase",
            example_b="Remove product or decrease enzyme for product inhibition",
            mnemonic="Substrate Short = Slow enzyme; Product Piles = Purge product",
            trick_indicator="Linear phase disappears → diagnose cause before suggesting solution"
        ))
    
    def add_concept(self, concept: ConceptNode):
        """Add a concept to the graph"""
        self.concepts[concept.name.lower()] = concept
        
        # Also add by aliases
        for alias in concept.aliases:
            self.concepts[alias.lower()] = concept
    
    def add_contrast(self, contrast: ConceptContrast):
        """Add a concept contrast"""
        self.contrasts.append(contrast)
    
    def get_concept(self, name: str) -> Optional[ConceptNode]:
        """Get a concept by name or alias"""
        return self.concepts.get(name.lower())
    
    def get_contrast(self, concept_a: str, concept_b: str) -> Optional[ConceptContrast]:
        """Get contrast between two concepts"""
        a_lower = concept_a.lower()
        b_lower = concept_b.lower()
        
        for contrast in self.contrasts:
            if (contrast.concept_a.lower() == a_lower and contrast.concept_b.lower() == b_lower) or \
               (contrast.concept_a.lower() == b_lower and contrast.concept_b.lower() == a_lower):
                return contrast
        
        return None
    
    def get_related_concepts(
        self, 
        concept_name: str, 
        relation_type: Optional[ConceptRelation] = None
    ) -> List[str]:
        """Get concepts related to a given concept"""
        concept = self.get_concept(concept_name)
        if not concept:
            return []
        
        if relation_type:
            return concept.get_related_by_type(relation_type)
        
        return list(concept.related_concepts.keys())
    
    def find_contrasts_for_concept(self, concept_name: str) -> List[ConceptContrast]:
        """Find all contrasts involving a concept"""
        name_lower = concept_name.lower()
        
        return [
            c for c in self.contrasts
            if c.concept_a.lower() == name_lower or c.concept_b.lower() == name_lower
        ]
    
    def explain_concept(self, name: str) -> str:
        """Get a comprehensive explanation of a concept"""
        concept = self.get_concept(name)
        if not concept:
            return f"Concept '{name}' not found in knowledge graph."
        
        explanation = f"## {concept.name}\n\n"
        explanation += f"**Definition:** {concept.definition}\n\n"
        
        if concept.properties:
            explanation += "**Properties:**\n"
            for key, value in concept.properties.items():
                if isinstance(value, list):
                    explanation += f"- {key}: {', '.join(value)}\n"
                else:
                    explanation += f"- {key}: {value}\n"
            explanation += "\n"
        
        if concept.examples:
            explanation += "**Examples:**\n"
            for example in concept.examples:
                explanation += f"- {example}\n"
            explanation += "\n"
        
        if concept.common_errors:
            explanation += "**[WARN]️ Common Errors:**\n"
            for error in concept.common_errors:
                explanation += f"- {error}\n"
            explanation += "\n"
        
        # Add contrasts
        contrasts = self.find_contrasts_for_concept(name)
        if contrasts:
            explanation += "**Related Contrasts:**\n"
            for contrast in contrasts:
                other = contrast.concept_b if contrast.concept_a.lower() == name.lower() else contrast.concept_a
                explanation += f"- vs {other}: {contrast.key_difference}\n"
        
        return explanation
    
    def check_for_confusion(
        self, 
        text: str, 
        concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Check if text might contain conceptual confusion"""
        warnings = []
        
        for concept in concepts:
            contrasts = self.find_contrasts_for_concept(concept)
            for contrast in contrasts:
                other = contrast.concept_b if contrast.concept_a.lower() == concept.lower() else contrast.concept_a
                
                # Check if both concepts appear in text
                if concept.lower() in text.lower() and other.lower() in text.lower():
                    # Potential confusion point
                    warnings.append({
                        "type": "potential_confusion",
                        "concepts": [concept, other],
                        "difference": contrast.key_difference,
                        "suggestion": f"Verify that {concept} and {other} are being used correctly"
                    })
        
        return warnings
    
    def get_concept_context_for_question(
        self, 
        question: str
    ) -> Dict[str, Any]:
        """Get relevant concept context for a question"""
        context = {
            "identified_concepts": [],
            "relevant_contrasts": [],
            "concept_definitions": [],
            "warnings": []
        }
        
        question_lower = question.lower()
        
        # Find mentioned concepts
        for name, concept in self.concepts.items():
            if name in question_lower:
                context["identified_concepts"].append(concept.name)
                
                # Add definition
                context["concept_definitions"].append({
                    "name": concept.name,
                    "definition": concept.definition
                })
                
                # Find contrasts
                contrasts = self.find_contrasts_for_concept(name)
                for contrast in contrasts:
                    if contrast not in context["relevant_contrasts"]:
                        context["relevant_contrasts"].append(contrast)
        
        # Check for potential confusion
        context["warnings"] = self.check_for_confusion(
            question, 
            context["identified_concepts"]
        )
        
        return context

