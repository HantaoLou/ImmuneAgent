"""
Domain-Specific Reasoning Templates for HLE

These templates provide structured reasoning frameworks for high-frequency
error domains identified in the HLE analysis:
- Genetics (inheritance, Hardy-Weinberg, genetic drift)
- Molecular Biology (operons, regulation, central dogma)
- Clinical Diagnosis (differential diagnosis, guidelines)

Key Features:
- Step-by-step reasoning frameworks
- Common pitfalls and how to avoid them
- Concept clarification checklists
- Validation checkpoints
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class ReasoningStepType(Enum):
    """Types of reasoning steps"""
    IDENTIFICATION = "identification"
    CLASSIFICATION = "classification"
    CALCULATION = "calculation"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"


@dataclass
class ReasoningStep:
    """A single step in a reasoning template"""
    step_number: int
    step_type: ReasoningStepType
    instruction: str
    key_questions: List[str] = field(default_factory=list)
    common_errors: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    output_format: str = ""


@dataclass
class ReasoningTemplate:
    """Complete reasoning template for a domain"""
    name: str
    description: str
    steps: List[ReasoningStep]
    key_concepts: Dict[str, str]
    common_pitfalls: List[Dict[str, str]]
    validation_checklist: List[str]
    example_workflow: Optional[List[str]] = None


@dataclass
class PitfallWarning:
    """Warning about a common pitfall"""
    pitfall_name: str
    description: str
    how_to_avoid: str
    example_of_error: str
    example_of_correct: str


class CommonPitfallsRegistry:
    """Registry of common pitfalls across all domains"""
    
    GENETICS_PITFALLS = [
        {
            "name": "H² vs h² Confusion",
            "description": "Confusing broad-sense (H²) and narrow-sense (h²) heritability",
            "how_to_avoid": "H² = Vg/Vp includes ALL genetic variance. h² = Va/Vp includes ONLY additive variance. Remember: h² ≤ H² always.",
            "example_of_error": "H² = 0.8 means breeding selection will be 80% effective",
            "example_of_correct": "h² = 0.4 means breeding selection will be 40% effective (H² = 0.8 just means genetics explains 80% of phenotypic variance)"
        },
        {
            "name": "Drift vs Selection",
            "description": "Misattributing allele frequency changes to wrong mechanism",
            "how_to_avoid": "Drift = random, stronger in small populations. Selection = directional, based on fitness differences. Ask: Is the change random or fitness-related?",
            "example_of_error": "Small population's neutral allele fixation shows positive selection",
            "example_of_correct": "Small population's neutral allele fixation is due to genetic drift (no fitness effect)"
        },
        {
            "name": "Incomplete Dominance vs Codominance",
            "description": "Confusing blending inheritance with simultaneous expression",
            "how_to_avoid": "Incomplete dominance = INTERMEDIATE phenotype (pink flowers). Codominance = BOTH expressed (AB blood type shows both A and B antigens).",
            "example_of_error": "ABO blood type is incomplete dominance",
            "example_of_correct": "ABO blood type is codominance (A and B antigens both present)"
        },
        {
            "name": "Linkage vs Independent Assortment",
            "description": "Assuming all genes assort independently",
            "how_to_avoid": "Check if genes are on the same chromosome. Linked genes deviate from expected Mendelian ratios. Look for 'recombination frequency' hints.",
            "example_of_error": "Genes 5 cM apart will show 9:3:3:1 ratio in dihybrid cross",
            "example_of_correct": "Genes 5 cM apart are linked and will show parental bias in offspring"
        }
    ]
    
    MOLECULAR_BIOLOGY_PITFALLS = [
        {
            "name": "trp Operon Attenuation Mechanism",
            "description": "Missing the RNA-level attenuation control",
            "how_to_avoid": "trp operon has DUAL control: repression (TrpR repressor) AND attenuation (transcription termination). Attenuation responds to Trp-tRNA charging levels.",
            "example_of_error": "trp operon is only controlled by TrpR repressor",
            "example_of_correct": "trp operon has dual control: TrpR repression AND attenuation via leader peptide stalling"
        },
        {
            "name": "Transcription vs Translation Regulation",
            "description": "Confusing where regulation occurs",
            "how_to_avoid": "Ask: Is the regulation changing mRNA levels (transcriptional) or protein production rate (translational)? Look for mRNA stability, ribosome binding, etc.",
            "example_of_error": "miRNA increases protein expression by activating translation",
            "example_of_correct": "miRNA typically DECREASES expression by mRNA degradation or translational repression"
        },
        {
            "name": "Allosteric vs Covalent Regulation",
            "description": "Confusing reversible binding with covalent modification",
            "how_to_avoid": "Allosteric = reversible binding of effector molecule. Covalent = phosphorylation, acetylation, etc. Look for 'kinase', 'phosphatase' keywords.",
            "example_of_error": "ATP allosterically activates phosphofructokinase permanently",
            "example_of_correct": "ATP allosterically inhibits phosphofructokinase (reversible binding at allosteric site)"
        }
    ]
    
    CLINICAL_PITFALLS = [
        {
            "name": "Zebra vs Horse",
            "description": "Focusing on rare conditions before common ones",
            "how_to_avoid": "'When you hear hoofbeats, think horses, not zebras.' Always consider common conditions first. Use prevalence data.",
            "example_of_error": "First differential for fever in Oklahoma is Lyme disease",
            "example_of_correct": "First differentials for fever in Oklahoma should include common regional diseases (Ehrlichia, Rickettsia)"
        },
        {
            "name": "Sensitivity vs Specificity Confusion",
            "description": "Misunderstanding test characteristics",
            "how_to_avoid": "Sensitivity = true positive rate (how many sick people are caught). Specificity = true negative rate (how many healthy people are correctly excluded).",
            "example_of_error": "A test with 99% sensitivity is good for ruling out disease",
            "example_of_correct": "High sensitivity is good for SCREENING (few false negatives). High specificity is good for CONFIRMATION (few false positives)"
        },
        {
            "name": "Incomplete Differential Diagnosis",
            "description": "Stopping too early in generating differentials",
            "how_to_avoid": "Generate at least 3-5 differentials before narrowing. Consider: infectious, inflammatory, neoplastic, metabolic, toxic causes.",
            "example_of_error": "The only differential for this presentation is condition X",
            "example_of_correct": "Differentials include X (most likely given epidemiology), Y (supported by lab findings), Z (less likely but serious if missed)"
        }
    ]
    
    @classmethod
    def get_pitfalls_for_domain(cls, domain: str) -> List[Dict[str, str]]:
        """Get common pitfalls for a specific domain"""
        # 安全检查: 确保 domain 不为 None
        if domain is None:
            # Return all for general queries
            return (
                cls.GENETICS_PITFALLS + 
                cls.MOLECULAR_BIOLOGY_PITFALLS + 
                cls.CLINICAL_PITFALLS
            )
        
        domain_lower = domain.lower()
        
        if "genetic" in domain_lower:
            return cls.GENETICS_PITFALLS
        elif "molecular" in domain_lower or "biochem" in domain_lower:
            return cls.MOLECULAR_BIOLOGY_PITFALLS
        elif "clinical" in domain_lower or "medicine" in domain_lower:
            return cls.CLINICAL_PITFALLS
        
        # Return all for general queries
        return (
            cls.GENETICS_PITFALLS + 
            cls.MOLECULAR_BIOLOGY_PITFALLS + 
            cls.CLINICAL_PITFALLS
        )
    
    @classmethod
    def check_for_pitfall(
        cls, 
        reasoning: str, 
        domain: str
    ) -> List[PitfallWarning]:
        """Check if reasoning shows signs of common pitfalls"""
        warnings = []
        pitfalls = cls.get_pitfalls_for_domain(domain)
        
        # 安全检查: 确保 reasoning 不为 None
        if reasoning is None:
            reasoning = ""
        reasoning_lower = reasoning.lower()
        
        for pitfall in pitfalls:
            # Simple keyword matching (can be enhanced with NLP)
            pitfall_keywords = pitfall["name"].lower().split()
            if any(kw in reasoning_lower for kw in pitfall_keywords):
                # Check if the reasoning might contain the error
                example_error = pitfall.get("example_of_error", "") or ""
                error_keywords = example_error.lower().split()[:5]
                if error_keywords and any(kw in reasoning_lower for kw in error_keywords):
                    warnings.append(PitfallWarning(
                        pitfall_name=pitfall["name"],
                        description=pitfall.get("description", ""),
                        how_to_avoid=pitfall.get("how_to_avoid", ""),
                        example_of_error=example_error,
                        example_of_correct=pitfall.get("example_of_correct", "")
                    ))
        
        return warnings


class GeneticsReasoningTemplate(ReasoningTemplate):
    """Reasoning template for genetics problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the genetic pattern",
                key_questions=[
                    "Is this Mendelian or non-Mendelian inheritance?",
                    "Is it autosomal or sex-linked?",
                    "Is it dominant or recessive?"
                ],
                common_errors=[
                    "Assuming all traits follow simple Mendelian ratios",
                    "Ignoring sex-linkage when males and females have different patterns",
                    "Confusing incomplete dominance with codominance"
                ],
                validation_criteria=[
                    "Pattern consistent with proposed inheritance mode",
                    "Punnett square supports observed ratios"
                ],
                output_format="Inheritance mode: [autosomal/sex-linked] [dominant/recessive] [Mendelian/non-Mendelian]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.CLASSIFICATION,
                instruction="Classify the cross type and expected ratios",
                key_questions=[
                    "What is the parental (P) generation genotype?",
                    "What cross is being performed (monohybrid, dihybrid, test cross)?",
                    "What are the expected F1 and F2 ratios?"
                ],
                common_errors=[
                    "Forgetting that F1 of two homozygotes are all heterozygous",
                    "Using wrong ratios for test crosses (1:1 for heterozygote × homozygote recessive)"
                ],
                validation_criteria=[
                    "Expected ratios match known cross patterns",
                    "F1 ratios consistent with P generation"
                ],
                output_format="Cross type: [type], Expected F1: [ratio], Expected F2: [ratio]"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.CALCULATION,
                instruction="Perform genetic calculations",
                key_questions=[
                    "What is the probability calculation needed?",
                    "Are Hardy-Weinberg calculations needed?",
                    "Is recombination frequency relevant?"
                ],
                common_errors=[
                    "Confusing H² (broad-sense) with h² (narrow-sense) heritability",
                    "Forgetting that Hardy-Weinberg frequencies (p², 2pq, q²) sum to 1",
                    "Not converting recombination frequency to map units (1% = 1 cM)"
                ],
                validation_criteria=[
                    "Probabilities sum to 1 (or expected total)",
                    "Units are correct",
                    "H² ≥ h² always"
                ],
                output_format="Calculation: [formula] = [result]"
            ),
            ReasoningStep(
                step_number=4,
                step_type=ReasoningStepType.VALIDATION,
                instruction="Check for special cases and edge conditions",
                key_questions=[
                    "Is there gene interaction (epistasis)?",
                    "Are the genes linked?",
                    "Is penetrance or expressivity relevant?"
                ],
                common_errors=[
                    "Assuming all genes are unlinked",
                    "Ignoring epistatic interactions",
                    "Not accounting for variable penetrance"
                ],
                validation_criteria=[
                    "All special cases identified",
                    "Final answer accounts for all factors"
                ],
                output_format="Special considerations: [list]"
            ),
            ReasoningStep(
                step_number=5,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Synthesize final answer",
                key_questions=[
                    "Does the answer directly address the question?",
                    "Is the reasoning chain complete?",
                    "Are units and format correct?"
                ],
                common_errors=[
                    "Answering a different question than asked",
                    "Providing incomplete reasoning"
                ],
                validation_criteria=[
                    "Answer is in required format",
                    "All steps logically connected"
                ],
                output_format="Final answer: [answer]"
            )
        ]
        
        key_concepts = {
            "H²": "Broad-sense heritability = Vg/Vp (all genetic variance)",
            "h²": "Narrow-sense heritability = Va/Vp (additive variance only)",
            "Hardy-Weinberg": "p² + 2pq + q² = 1 for allele frequencies",
            "Linkage": "Genes on same chromosome, recombination frequency = map distance",
            "Epistasis": "One gene masks expression of another",
            "Penetrance": "Proportion of individuals with genotype who show phenotype",
            "Expressivity": "Degree to which a phenotype is expressed"
        }
        
        super().__init__(
            name="Genetics Reasoning",
            description="Structured reasoning for genetics problems",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=CommonPitfallsRegistry.GENETICS_PITFALLS,
            validation_checklist=[
                "Inheritance pattern identified correctly",
                "Cross type classified",
                "Calculations verified",
                "Special cases considered",
                "Answer format matches requirement"
            ],
            example_workflow=[
                "1. Pattern: Autosomal recessive (skips generations, equal M/F)",
                "2. Cross: aa × Aa → 50% affected expected",
                "3. No linkage detected (independent assortment)",
                "4. Answer: 0.5 or 50%"
            ]
        )
    
    def generate_prompt(self, question: str) -> str:
        """Generate a structured prompt for this template"""
        prompt = f"""## Genetics Problem Solving Framework

### Question
{question}

### Step-by-Step Reasoning

**Step 1: Identify Genetic Pattern**
- Is this Mendelian or non-Mendelian?
- Autosomal or sex-linked?
- Dominant or recessive?

**Step 2: Classify Cross Type**
- What are the parental genotypes?
- What cross is being performed?
- What are expected ratios?

**Step 3: Perform Calculations**
- Apply appropriate formulas
- Show work clearly
- Verify probabilities sum correctly

**Step 4: Check Special Cases**
- Gene interactions?
- Linkage?
- Penetrance/expressivity?

**Step 5: Final Answer**
- Format as required
- Verify against question

### Key Concepts to Remember
- H² (broad-sense) includes ALL genetic variance: Vg/Vp
- h² (narrow-sense) includes ONLY additive variance: Va/Vp
- Hardy-Weinberg: p² + 2pq + q² = 1
- Linked genes: recombination frequency = map distance (cM)

### Common Pitfalls to Avoid
"""
        for pitfall in self.common_pitfalls[:3]:
            prompt += f"\n- **{pitfall['name']}**: {pitfall['how_to_avoid']}\n"
        
        return prompt


class MolecularBiologyTemplate(ReasoningTemplate):
    """Reasoning template for molecular biology problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the biological system and level",
                key_questions=[
                    "What system is involved (gene regulation, metabolism, signaling)?",
                    "What is the hierarchical level (DNA, RNA, protein, cellular)?",
                    "Is this a normal or disrupted process?"
                ],
                common_errors=[
                    "Confusing transcriptional and translational regulation",
                    "Missing multi-level regulation (e.g., trp operon has dual control)"
                ],
                validation_criteria=[
                    "System correctly identified",
                    "All relevant components listed"
                ],
                output_format="System: [name], Level: [DNA/RNA/protein/cellular]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Build the causal chain",
                key_questions=[
                    "What is the input signal/stimulus?",
                    "What are the intermediate steps?",
                    "What is the output/response?"
                ],
                common_errors=[
                    "Missing intermediate steps",
                    "Reversing cause and effect"
                ],
                validation_criteria=[
                    "Chain is logically connected",
                    "Each step leads to the next"
                ],
                output_format="Signal → [intermediates] → Response"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.ABDUCTION,
                instruction="Apply forward or backward reasoning",
                key_questions=[
                    "If forward: What happens if component X is mutated?",
                    "If backward: What mutation would cause phenotype Y?"
                ],
                common_errors=[
                    "Not considering partial loss-of-function vs null",
                    "Forgetting about compensatory mechanisms"
                ],
                validation_criteria=[
                    "Prediction matches known biology",
                    "Alternative explanations considered"
                ],
                output_format="Prediction: If [condition], then [result]"
            ),
            ReasoningStep(
                step_number=4,
                step_type=ReasoningStepType.VALIDATION,
                instruction="Check for special mechanisms",
                key_questions=[
                    "Is attenuation involved (trp-like)?",
                    "Is there feedback inhibition?",
                    "Is there allosteric regulation?"
                ],
                common_errors=[
                    "Only considering one level of regulation",
                    "Missing RNA-level mechanisms"
                ],
                validation_criteria=[
                    "All regulation levels considered",
                    "Special mechanisms identified if present"
                ],
                output_format="Regulation: [primary] + [secondary if any]"
            ),
            ReasoningStep(
                step_number=5,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Synthesize final answer",
                key_questions=[
                    "Does the answer explain all observations?",
                    "Is the mechanism complete?"
                ],
                common_errors=[
                    "Incomplete explanation",
                    "Missing key mechanism"
                ],
                validation_criteria=[
                    "Answer addresses all parts of question",
                    "Mechanism is complete"
                ],
                output_format="Final answer: [answer]"
            )
        ]
        
        key_concepts = {
            "Central Dogma": "DNA → RNA → Protein (with exceptions like retroviruses)",
            "Attenuation": "Transcription termination control via leader peptide",
            "Allosteric Regulation": "Effector binding at site distinct from active site",
            "Feedback Inhibition": "End product inhibits early step in pathway",
            "Cooperativity": "Binding of one ligand affects binding of subsequent ligands"
        }
        
        # trp operon specific template
        trp_template = """
## trp Operon Attenuation Mechanism

### Four-Region Structure
- Region 1: Contains 2 Trp codons (stalling site)
- Region 2: Can pair with Region 1 or 3
- Region 3: Can pair with Region 2 or 4
- Region 4: Pairs with Region 3 to form terminator

### High Tryptophan Conditions
1. Ribosome translates Region 1 smoothly (Trp-tRNA abundant)
2. Region 2 covered by ribosome
3. Region 3-4 pair → terminator stem-loop forms
4. Transcription terminates early

### Low Tryptophan Conditions
1. Ribosome stalls at Trp codons in Region 1
2. Region 2 available to pair with Region 3
3. Region 2-3 pair → anti-terminator forms
4. Region 4 cannot pair, transcription continues

### Key Mutations
- Region 1 mutations → may affect stalling
- Region 4 mutations (U-rich → GC-rich) → destabilize terminator → constitutive expression
"""
        
        super().__init__(
            name="Molecular Biology Reasoning",
            description="Structured reasoning for molecular biology problems",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=CommonPitfallsRegistry.MOLECULAR_BIOLOGY_PITFALLS,
            validation_checklist=[
                "System and level identified",
                "Causal chain complete",
                "Special mechanisms considered",
                "Answer explains all observations"
            ],
            example_workflow=[
                "1. System: trp operon regulation",
                "2. Chain: High Trp → repression + attenuation → low expression",
                "3. Forward: Region 4 mutation → no termination → constitutive",
                "4. Answer: U-rich to GC-rich in Region 4 prevents terminator formation"
            ]
        )
        
        self.special_templates = {
            "trp_operon": trp_template
        }
    
    def get_special_template(self, topic: str) -> Optional[str]:
        """Get specialized template for specific topics"""
        return self.special_templates.get(topic.lower())


class ClinicalDiagnosisTemplate(ReasoningTemplate):
    """Reasoning template for clinical diagnosis problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Extract key clinical information",
                key_questions=[
                    "What is the chief complaint?",
                    "What are key history elements (PMH, medications, exposures)?",
                    "What are key physical exam findings?",
                    "What are key lab/imaging results?"
                ],
                common_errors=[
                    "Missing important history elements",
                    "Overlooking subtle physical findings"
                ],
                validation_criteria=[
                    "All clinical data extracted",
                    "Key positive and negative findings noted"
                ],
                output_format="Chief complaint: [CC], Key findings: [list]"
            ),
            ReasoningStep(
                step_type=ReasoningStepType.ABDUCTION,
                step_number=2,
                instruction="Generate differential diagnosis",
                key_questions=[
                    "What are the most common causes of this presentation?",
                    "What serious conditions must not be missed?",
                    "What is the epidemiological context (age, geography, season)?"
                ],
                common_errors=[
                    "Jumping to rare diagnosis before common ones (zebra vs horse)",
                    "Not considering epidemiological context"
                ],
                validation_criteria=[
                    "At least 3-5 differentials generated",
                    "Differentials ranked by likelihood"
                ],
                output_format="Differential diagnosis: 1. [most likely] 2. [second] 3. [third]..."
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Evaluate each differential",
                key_questions=[
                    "What findings support this diagnosis?",
                    "What findings argue against this diagnosis?",
                    "What additional tests would help?"
                ],
                common_errors=[
                    "Confirmation bias (only seeking supporting evidence)",
                    "Not considering test characteristics (sensitivity/specificity)"
                ],
                validation_criteria=[
                    "Each differential systematically evaluated",
                    "Supporting and refuting evidence listed"
                ],
                output_format="For [diagnosis]: Supporting: [list], Against: [list]"
            ),
            ReasoningStep(
                step_number=4,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Select most likely diagnosis",
                key_questions=[
                    "Which diagnosis best explains all findings?",
                    "Are there findings that strongly point to one diagnosis?",
                    "Is there a unifying diagnosis or are there multiple conditions?"
                ],
                common_errors=[
                    "Ignoring inconsistent findings",
                    "Forcing single diagnosis when multiple conditions present"
                ],
                validation_criteria=[
                    "Selected diagnosis explains most findings",
                    "Inconsistencies acknowledged and explained"
                ],
                output_format="Most likely diagnosis: [diagnosis] because [reasoning]"
            ),
            ReasoningStep(
                step_number=5,
                step_type=ReasoningStepType.VALIDATION,
                instruction="Verify against clinical guidelines",
                key_questions=[
                    "What do clinical guidelines recommend?",
                    "Are there diagnostic criteria that must be met?",
                    "Is specialist consultation indicated?"
                ],
                common_errors=[
                    "Not following established guidelines",
                    "Missing required criteria for diagnosis"
                ],
                validation_criteria=[
                    "Consistent with current guidelines",
                    "Diagnostic criteria met (if applicable)"
                ],
                output_format="Guideline compliance: [met/not met], Criteria: [list]"
            )
        ]
        
        key_concepts = {
            "Sensitivity": "True positive rate - how many sick people test positive",
            "Specificity": "True negative rate - how many healthy people test negative",
            "PPV": "Positive predictive value - probability of disease given positive test",
            "NPV": "Negative predictive value - probability of no disease given negative test",
            "Likelihood Ratio": "How much a test result changes disease probability"
        }
        
        super().__init__(
            name="Clinical Diagnosis Reasoning",
            description="Structured reasoning for clinical diagnosis",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=CommonPitfallsRegistry.CLINICAL_PITFALLS,
            validation_checklist=[
                "All clinical information extracted",
                "Differential diagnosis generated",
                "Each differential evaluated",
                "Final diagnosis justified",
                "Guidelines considered"
            ],
            example_workflow=[
                "1. CC: Fever + rash, PMH: Oklahoma resident, hiking exposure",
                "2. DDx: 1. Ehrlichiosis 2. RMSF 3. Lyme (less likely in Oklahoma)",
                "3. Ehrlichiosis: + for Oklahoma, + for fever, + for exposure; Lyme: - for geography",
                "4. Most likely: Ehrlichiosis (geography + clinical picture)",
                "5. Plan: Doxycycline empiric treatment per guidelines"
            ]
        )
    
    def generate_diagnostic_prompt(
        self,
        chief_complaint: str,
        history: Dict[str, Any],
        physical_exam: Dict[str, Any],
        labs: Dict[str, Any]
    ) -> str:
        """Generate a structured diagnostic prompt"""
        
        prompt = f"""## Clinical Reasoning Framework

### Case Summary
**Chief Complaint:** {chief_complaint}

**History of Present Illness:**
{self._format_dict(history)}

**Physical Examination:**
{self._format_dict(physical_exam)}

**Laboratory/Imaging:**
{self._format_dict(labs)}

### Diagnostic Approach

**Step 1: Information Synthesis**
List all positive and pertinent negative findings.

**Step 2: Differential Diagnosis Generation**
Start with most common, include "must not miss" diagnoses.

**Step 3: Differential Evaluation**
For each diagnosis:
- Supporting features (+)
- Refuting features (-)
- Additional tests needed

**Step 4: Final Diagnosis**
Select most likely diagnosis with justification.

**Step 5: Management Plan**
Include immediate treatment and follow-up.

### Key Reminders
- "When you hear hoofbeats, think horses, not zebras"
- Consider epidemiology (geography, season, exposure)
- Know test characteristics (sensitivity vs specificity)
- Don't delay treatment for confirmatory tests if condition is serious
"""
        return prompt
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for display"""
        if not d:
            return "Not provided"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())


# Template factory
def get_template_for_domain(domain: str) -> Optional[ReasoningTemplate]:
    """Get the appropriate reasoning template for a domain"""
    # 安全检查: 确保 domain 不为 None
    if domain is None:
        return None
    
    domain_lower = domain.lower()
    
    if "genetic" in domain_lower or "population" in domain_lower or "evolution" in domain_lower:
        return GeneticsReasoningTemplate()
    elif "molecular" in domain_lower or "biochem" in domain_lower:
        return MolecularBiologyTemplate()
    elif "clinical" in domain_lower or "medicine" in domain_lower or "diagnos" in domain_lower:
        return ClinicalDiagnosisTemplate()
    elif "protein" in domain_lower or "structur" in domain_lower or "biophys" in domain_lower:
        return ProteinStructureTemplate()
    elif "enzyme" in domain_lower or "kinetic" in domain_lower or "metabol" in domain_lower:
        return EnzymeKineticsTemplate()
    elif "microbiol" in domain_lower or "bacter" in domain_lower or "virul" in domain_lower:
        return MicrobiologyTemplate()
    elif "bioinformat" in domain_lower or "sequenc" in domain_lower or "genom" in domain_lower:
        return BioinformaticsTemplate()
    
    return None


# ==================== Additional Domain Templates ====================

class ProteinStructureTemplate(ReasoningTemplate):
    """Reasoning template for protein structure and biophysics problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the structural element and analysis type",
                key_questions=[
                    "What structural element is involved (coiled-coil, helix, sheet, domain)?",
                    "What type of analysis is needed (prediction, stability, interaction)?",
                    "What experimental data is available (DLS, SEC-MALS, CD, etc.)?"
                ],
                common_errors=[
                    "Ignoring the heptad repeat pattern in coiled-coils",
                    "Misinterpreting hydrodynamic radius from DLS"
                ],
                validation_criteria=[
                    "Structural element correctly identified",
                    "Analysis type matched to question"
                ],
                output_format="Structure: [type], Analysis: [type]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Apply structure prediction rules",
                key_questions=[
                    "For coiled-coils: What are the a and d position residues?",
                    "For stability: What factors affect folding (temperature, mutations)?",
                    "For interactions: What interfaces are involved?"
                ],
                common_errors=[
                    "Not analyzing a/d positions in coiled-coil heptad repeats",
                    "Assuming all hydrophobic cores are equivalent"
                ],
                validation_criteria=[
                    "Heptad positions correctly assigned (a-g)",
                    "Core residues analyzed for knobs-into-holes packing"
                ],
                output_format="Prediction: [oligomeric state/structure]"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.ABDUCTION,
                instruction="Interpret experimental data",
                key_questions=[
                    "DLS: What does Rh tell us about aggregation state?",
                    "SEC-MALS: What is the absolute molecular weight?",
                    "CD: What is the secondary structure content?"
                ],
                common_errors=[
                    "Confusing hydrodynamic radius with molecular weight",
                    "Not accounting for protein shape in DLS interpretation"
                ],
                validation_criteria=[
                    "Data correctly interpreted",
                    "Conclusions supported by data"
                ],
                output_format="Data interpretation: [finding]"
            ),
            ReasoningStep(
                step_number=4,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Synthesize structural conclusion",
                key_questions=[
                    "Does the structure explain the function?",
                    "Are there mutations that would affect the structure?",
                    "What are the stability determinants?"
                ],
                common_errors=[
                    "Not connecting structure to function",
                    "Ignoring stability implications"
                ],
                validation_criteria=[
                    "Conclusion logically follows from analysis",
                    "All relevant factors considered"
                ],
                output_format="Final answer: [structural conclusion]"
            )
        ]
        
        key_concepts = {
            "Coiled-Coil Heptad Repeat": "7-residue pattern (a-b-c-d-e-f-g) where a,d are hydrophobic core",
            "Knobs-into-Holes": "Core packing that determines oligomeric state",
            "DLS Rh": "Hydrodynamic radius - increases with aggregation",
            "SEC-MALS": "Size exclusion + light scattering for absolute Mw",
            "PDI": "Polydispersity index - measures sample homogeneity"
        }
        
        super().__init__(
            name="Protein Structure Reasoning",
            description="Structured reasoning for protein structure problems",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=[
                {
                    "name": "Coiled-Coil Oligomer Prediction",
                    "description": "Predicting wrong oligomeric state from sequence",
                    "how_to_avoid": "Analyze a and d positions: small residues (Ala, Asn) favor dimers; large residues (Leu, Ile) favor trimers/tetramers; charged residues can specify higher oligomers",
                    "example_of_error": "Any hydrophobic a/d means dimer",
                    "example_of_correct": "a,d positions determine oligomer: Ala-rich = dimer, Leu-rich = trimer/tetramer"
                }
            ],
            validation_checklist=[
                "Structural element identified",
                "Heptad pattern analyzed (for coiled-coils)",
                "Experimental data interpreted correctly",
                "Structure-function connection made"
            ]
        )


class EnzymeKineticsTemplate(ReasoningTemplate):
    """Reasoning template for enzyme kinetics and metabolism problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the kinetic problem type",
                key_questions=[
                    "Is this about reaction rates, linear phase, or metabolite tracking?",
                    "What are the enzyme and substrate?",
                    "What experimental conditions are described?"
                ],
                common_errors=[
                    "Not recognizing substrate depletion as cause of nonlinearity",
                    "Confusing initial rate with steady-state rate"
                ],
                validation_criteria=[
                    "Problem type identified",
                    "Key components listed"
                ],
                output_format="Problem type: [rate/linear/metabolism]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Analyze the kinetic mechanism",
                key_questions=[
                    "For linear phase issues: Is substrate being depleted?",
                    "For metabolism: What pathway is involved?",
                    "What are the rate-limiting steps?"
                ],
                common_errors=[
                    "Assuming higher enzyme concentration always helps",
                    "Not tracing isotope through metabolic steps"
                ],
                validation_criteria=[
                    "Mechanism correctly identified",
                    "Key factors enumerated"
                ],
                output_format="Mechanism: [description]"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.CALCULATION,
                instruction="Perform kinetic calculations",
                key_questions=[
                    "What kinetic parameters are needed (Km, Vmax, kcat)?",
                    "For isotope tracing: Which carbons are labeled?",
                    "What formulas apply?"
                ],
                common_errors=[
                    "Not tracking individual carbon atoms",
                    "Misapplying Michaelis-Menten assumptions"
                ],
                validation_criteria=[
                    "Correct formulas used",
                    "Units verified"
                ],
                output_format="Calculation: [result]"
            ),
            ReasoningStep(
                step_number=4,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Provide solution based on kinetics",
                key_questions=[
                    "What change would improve the situation?",
                    "For isotope questions: Where does the label end up?",
                    "What is the expected outcome?"
                ],
                common_errors=[
                    "Suggesting changes that worsen the problem",
                    "Not considering alternative explanations"
                ],
                validation_criteria=[
                    "Solution addresses the problem",
                    "Prediction is biologically reasonable"
                ],
                output_format="Solution: [answer]"
            )
        ]
        
        key_concepts = {
            "Linear Phase Loss": "Substrate depletion → decrease enzyme concentration",
            "Product Inhibition": "Product accumulation → decrease enzyme or remove product",
            "Isotope Tracking": "Follow labeled carbon through metabolic steps",
            "Glycolysis": "Glucose → 2 Pyruvate (no CO2 release)",
            "PDH Reaction": "Pyruvate → Acetyl-CoA + CO2 (1st CO2 from glucose C1)"
        }
        
        super().__init__(
            name="Enzyme Kinetics Reasoning",
            description="Structured reasoning for enzyme kinetics and metabolism",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=[
                {
                    "name": "Linear Phase Disappearance",
                    "description": "Not recognizing why product vs time curve is nonlinear",
                    "how_to_avoid": "Nonlinear product curve = substrate running out. Solution: LESS enzyme (slower consumption) or MORE substrate",
                    "example_of_error": "Increase enzyme to get more product",
                    "example_of_correct": "Decrease enzyme concentration to extend linear phase by slowing substrate consumption"
                },
                {
                    "name": "Isotope Carbon Tracking",
                    "description": "Losing track of which carbon is released when",
                    "how_to_avoid": "Glycolysis: no CO2. PDH: C1 → CO2. TCA cycle: C2,C3,C4 released over multiple turns",
                    "example_of_error": "1-13C-glucose releases 13CO2 during glycolysis",
                    "example_of_correct": "1-13C-glucose releases 13CO2 at PDH step (first CO2), not during glycolysis"
                }
            ],
            validation_checklist=[
                "Problem type identified",
                "Mechanism analyzed",
                "Calculations verified",
                "Solution addresses problem"
            ]
        )


class MicrobiologyTemplate(ReasoningTemplate):
    """Reasoning template for microbiology problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the microorganism and question type",
                key_questions=[
                    "What organism is involved (bacteria, virus, fungus)?",
                    "Is this about virulence, metabolism, or identification?",
                    "What growth conditions are described?"
                ],
                common_errors=[
                    "Not considering growth conditions",
                    "Confusing similar organisms"
                ],
                validation_criteria=[
                    "Organism identified",
                    "Question type classified"
                ],
                output_format="Organism: [name], Question type: [type]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Apply microbiological knowledge",
                key_questions=[
                    "What are the key characteristics of this organism?",
                    "What virulence factors are relevant?",
                    "What metabolic pathways are involved?"
                ],
                common_errors=[
                    "Not considering virulence factor interactions",
                    "Missing regulatory mechanisms"
                ],
                validation_criteria=[
                    "Characteristics correctly stated",
                    "Relevant factors identified"
                ],
                output_format="Characteristics: [list]"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Synthesize microbiological conclusion",
                key_questions=[
                    "What explains the observed phenotype?",
                    "How do virulence factors interact?",
                    "What would mutation/disruption cause?"
                ],
                common_errors=[
                    "Incomplete virulence factor analysis",
                    "Not considering host-pathogen interactions"
                ],
                validation_criteria=[
                    "Explanation complete",
                    "All factors considered"
                ],
                output_format="Conclusion: [answer]"
            )
        ]
        
        key_concepts = {
            "Virulence Factors": "Pathogen components that cause disease",
            "Quorum Sensing": "Population-density dependent gene regulation",
            "Biofilm": "Surface-attached community with altered phenotype",
            "Motility Types": "Swimming (single), swarming (surface), twitching (type IV pili)"
        }
        
        super().__init__(
            name="Microbiology Reasoning",
            description="Structured reasoning for microbiology problems",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=[
                {
                    "name": "Virulence Factor Interactions",
                    "description": "Not considering how multiple virulence factors work together",
                    "how_to_avoid": "Map virulence factors to their targets. Check if factors affect same or different host pathways",
                    "example_of_error": "Each virulence factor works independently",
                    "example_of_correct": "Virulence factors may synergize, antagonize, or target overlapping host pathways"
                }
            ],
            validation_checklist=[
                "Organism identified",
                "Characteristics applied",
                "Interactions considered"
            ]
        )


class BioinformaticsTemplate(ReasoningTemplate):
    """Reasoning template for bioinformatics problems"""
    
    def __init__(self):
        steps = [
            ReasoningStep(
                step_number=1,
                step_type=ReasoningStepType.IDENTIFICATION,
                instruction="Identify the bioinformatics task",
                key_questions=[
                    "What type of analysis (sequence, structure, expression, phylogeny)?",
                    "What tools or algorithms are appropriate?",
                    "What is the input data format?"
                ],
                common_errors=[
                    "Choosing wrong algorithm for data type",
                    "Not considering sequence orientation/reading frame"
                ],
                validation_criteria=[
                    "Task type identified",
                    "Appropriate tools selected"
                ],
                output_format="Task: [type], Tools: [list]"
            ),
            ReasoningStep(
                step_number=2,
                step_type=ReasoningStepType.DEDUCTION,
                instruction="Apply bioinformatics methods",
                key_questions=[
                    "For sequences: What is the correct reading frame?",
                    "For clustering: What distance metric is appropriate?",
                    "For annotation: What databases to search?"
                ],
                common_errors=[
                    "Using wrong reading frame",
                    "Inappropriate distance metric for data type"
                ],
                validation_criteria=[
                    "Method correctly applied",
                    "Parameters appropriate"
                ],
                output_format="Method: [description]"
            ),
            ReasoningStep(
                step_number=3,
                step_type=ReasoningStepType.SYNTHESIS,
                instruction="Interpret bioinformatics results",
                key_questions=[
                    "What do the scores/metrics mean?",
                    "Is the result statistically significant?",
                    "What is the biological interpretation?"
                ],
                common_errors=[
                    "Over-interpreting weak signals",
                    "Not considering multiple testing"
                ],
                validation_criteria=[
                    "Results correctly interpreted",
                    "Biological meaning extracted"
                ],
                output_format="Interpretation: [finding]"
            )
        ]
        
        key_concepts = {
            "Reading Frame": "Codon starting position (+1, +2, +3, -1, -2, -3)",
            "ORF": "Open reading frame - potential protein-coding region",
            "E-value": "Expected number of random hits - lower is better",
            "Clustering": "Grouping similar sequences/genes based on distance metric"
        }
        
        super().__init__(
            name="Bioinformatics Reasoning",
            description="Structured reasoning for bioinformatics problems",
            steps=steps,
            key_concepts=key_concepts,
            common_pitfalls=[
                {
                    "name": "Reading Frame Selection",
                    "description": "Translating in wrong reading frame",
                    "how_to_avoid": "Start at ATG (Met), check for stop codons, verify against known protein length",
                    "example_of_error": "Any ATG is a start codon",
                    "example_of_correct": "Valid start codon: ATG near 5' end, followed by ORF without premature stops"
                }
            ],
            validation_checklist=[
                "Task identified",
                "Method applied correctly",
                "Results interpreted"
            ]
        )


