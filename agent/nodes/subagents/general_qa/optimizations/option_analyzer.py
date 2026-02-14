"""
P0: Multiple Choice Option Analyzer

Provides deep analysis of multiple choice options including:
- Semantic structure parsing
- Entity extraction (vectors, plasmids, etc.)
- Vector type identification (Duet, single, dual)
- Option exclusion based on constraints
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class OptionStructureType(Enum):
    """Structure types for options"""
    SINGLE_VECTOR = "single_vector"       # 单载体
    DUAL_PLASMID = "dual_plasmid"         # 双质粒系统
    DUET_VECTOR = "duet_vector"           # Duet双启动子载体
    TRIPLE_SYSTEM = "triple_system"       # 三元系统
    UNKNOWN = "unknown"


class MatchStatus(Enum):
    """Match status for options"""
    MATCH = "match"
    PARTIAL_MATCH = "partial_match"
    EXCLUDE = "exclude"
    NEED_MORE_INFO = "need_more_info"


@dataclass
class OptionAnalysis:
    """Analysis result for a single option"""
    option_id: str
    option_text: str
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    structure_type: OptionStructureType = OptionStructureType.UNKNOWN
    keywords: List[str] = field(default_factory=list)
    
    # 特殊实体识别
    vector_names: List[str] = field(default_factory=list)
    resistance_markers: List[str] = field(default_factory=list)
    promoters: List[str] = field(default_factory=list)
    
    # 匹配状态
    match_status: MatchStatus = MatchStatus.NEED_MORE_INFO
    match_reason: str = ""
    
    # 技术特征
    technical_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCQAnalysisResult:
    """Complete MCQ analysis result"""
    question_text: str
    options: Dict[str, OptionAnalysis]
    excluded_options: List[Tuple[str, str, str]]  # (option_id, reason, constraint)
    remaining_options: List[str]
    recommended_answer: Optional[str] = None
    confidence: float = 0.0


# 常见载体知识库
VECTOR_KNOWLEDGE = {
    "pCDFDuet-1": {
        "type": "duet",
        "origin": "CDF",
        "copy_number": "low",
        "resistance": "spectinomycin",
        "promoters": ["T7", "T7lac"],
        "features": ["Dual MCS", "Single vector for co-expression"],
        "advantages": ["No plasmid incompatibility", "Guaranteed co-transfer", "Simple selection"],
        "use_cases": ["Co-expression of chaperone + target"]
    },
    "pET-28a(+)": {
        "type": "expression",
        "origin": "pBR322",
        "copy_number": "high",
        "resistance": "kanamycin",
        "promoters": ["T7", "T7lac"],
        "features": ["N-terminal His-tag", "T7 tag"],
        "advantages": ["High expression", "Easy purification"],
        "use_cases": ["High-level protein expression"]
    },
    "pCDF-1b": {
        "type": "expression",
        "origin": "CDF",
        "copy_number": "low",
        "resistance": "spectinomycin",
        "promoters": ["T7"],
        "features": ["Single MCS"],
        "advantages": ["Compatible with ColE1 plasmids"],
        "use_cases": ["Dual plasmid co-expression"]
    },
    "pGEX-T4-1": {
        "type": "fusion",
        "origin": "pBR322",
        "copy_number": "medium",
        "resistance": "ampicillin",
        "promoters": ["tac"],
        "features": ["GST fusion tag"],
        "advantages": ["Solubility enhancement"],
        "use_cases": ["Fusion protein expression"]
    },
    "pET-15b": {
        "type": "expression",
        "origin": "pBR322",
        "copy_number": "high",
        "resistance": "ampicillin",
        "promoters": ["T7"],
        "features": ["N-terminal His-tag", "Thrombin site"],
        "advantages": ["Easy purification"],
        "use_cases": ["His-tagged protein expression"]
    },
    "pASK-IBA3": {
        "type": "expression",
        "origin": "pACYC",
        "copy_number": "low",
        "resistance": "chloramphenicol",
        "promoters": ["tet"],
        "features": ["Strep-tag II"],
        "advantages": ["Tight regulation"],
        "use_cases": ["Tightly controlled expression"]
    },
    "pGEM-T": {
        "type": "cloning",
        "origin": "pUC",
        "copy_number": "high",
        "resistance": "ampicillin",
        "promoters": ["T7", "SP6"],
        "features": ["TA cloning"],
        "advantages": ["Easy cloning"],
        "use_cases": ["PCR product cloning"]
    }
}

# 载体兼容性矩阵
PLASMID_COMPATIBILITY = {
    "pBR322": ["CDF", "pACYC", "pSC101"],
    "CDF": ["pBR322", "ColE1", "pACYC"],
    "pACYC": ["pBR322", "CDF", "pSC101"],
    "pSC101": ["pBR322", "CDF", "pACYC"],
    "ColE1": ["CDF", "pACYC", "pSC101"]
}


class MCQOptionAnalyzer:
    """Deep analyzer for multiple choice options"""
    
    def __init__(self, vector_knowledge: Dict = None):
        self.vector_knowledge = vector_knowledge or VECTOR_KNOWLEDGE
        self.plasmid_compatibility = PLASMID_COMPATIBILITY
    
    def analyze_question(self, question_text: str, options: Dict[str, str]) -> MCQAnalysisResult:
        """
        Analyze all options for a multiple choice question
        
        Args:
            question_text: The question text
            options: Dict mapping option_id to option_text
        
        Returns:
            MCQAnalysisResult with complete analysis
        """
        # 分析每个选项
        analyzed_options = {}
        for opt_id, opt_text in options.items():
            analyzed_options[opt_id] = self._analyze_single_option(opt_id, opt_text)
        
        # 根据问题要求筛选选项
        excluded = []
        remaining = []
        
        for opt_id, analysis in analyzed_options.items():
            if analysis.match_status == MatchStatus.EXCLUDE:
                excluded.append((opt_id, analysis.match_reason, ""))
            else:
                remaining.append(opt_id)
        
        # 推荐最佳答案
        recommended = self._recommend_best_option(question_text, analyzed_options, remaining)
        
        return MCQAnalysisResult(
            question_text=question_text,
            options=analyzed_options,
            excluded_options=excluded,
            remaining_options=remaining,
            recommended_answer=recommended,
            confidence=self._calculate_confidence(analyzed_options, recommended)
        )
    
    def _analyze_single_option(self, option_id: str, option_text: str) -> OptionAnalysis:
        """Analyze a single option"""
        analysis = OptionAnalysis(
            option_id=option_id,
            option_text=option_text
        )
        
        # 1. 提取实体
        analysis.entities = self._extract_entities(option_text)
        
        # 2. 识别结构类型
        analysis.structure_type = self._identify_structure_type(option_text, analysis.entities)
        
        # 3. 提取载体名称
        analysis.vector_names = self._extract_vector_names(option_text)
        
        # 4. 提取抗性标记
        analysis.resistance_markers = self._extract_resistance_markers(option_text)
        
        # 5. 提取技术特征
        analysis.technical_features = self._extract_technical_features(
            option_text, 
            analysis.vector_names
        )
        
        # 6. 评估匹配状态
        analysis.match_status, analysis.match_reason = self._evaluate_match_status(analysis)
        
        return analysis
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from option text"""
        entities = []
        
        # 提取载体名称
        vector_pattern = r'p[A-Z][A-Za-z0-9\-]+(?:\([+-]\))?'
        entities.extend(re.findall(vector_pattern, text))
        
        # 提取抗性基因
        resistance_pattern = r'(kanamycin|ampicillin|spectinomycin|chloramphenicol|tetracycline)\s+resistance'
        entities.extend(re.findall(resistance_pattern, text, re.IGNORECASE))
        
        return list(set(entities))
    
    def _identify_structure_type(self, text: str, entities: List[str]) -> OptionStructureType:
        """Identify the structure type of the option"""
        text_lower = text.lower()
        
        # 检查Duet载体
        if "duet" in text_lower:
            return OptionStructureType.DUET_VECTOR
        
        # 检查双质粒系统
        if " and " in text_lower:
            # 检查是否有两个载体名称
            vector_count = len([e for e in entities if e.startswith('p')])
            if vector_count >= 2:
                return OptionStructureType.DUAL_PLASMID
        
        # 检查三元系统
        if text.count(" and ") >= 2:
            return OptionStructureType.TRIPLE_SYSTEM
        
        # 检查单载体
        if len([e for e in entities if e.startswith('p')]) == 1:
            return OptionStructureType.SINGLE_VECTOR
        
        return OptionStructureType.UNKNOWN
    
    def _extract_vector_names(self, text: str) -> List[str]:
        """Extract vector names from text"""
        pattern = r'p[A-Z][A-Za-z0-9\-]+(?:\([+-]\))?'
        return re.findall(pattern, text)
    
    def _extract_resistance_markers(self, text: str) -> List[str]:
        """Extract resistance markers from text"""
        markers = []
        resistance_map = {
            "kanamycin": "kan",
            "ampicillin": "amp",
            "spectinomycin": "spec",
            "chloramphenicol": "cam",
            "tetracycline": "tet"
        }
        
        for full_name, abbrev in resistance_map.items():
            if full_name.lower() in text.lower():
                markers.append(full_name)
        
        return markers
    
    def _extract_technical_features(self, text: str, vector_names: List[str]) -> Dict[str, Any]:
        """Extract technical features from option"""
        features = {
            "vectors_info": [],
            "plasmid_origins": [],
            "is_compatible": True,
            "copy_numbers": []
        }
        
        for vector in vector_names:
            if vector in self.vector_knowledge:
                info = self.vector_knowledge[vector]
                features["vectors_info"].append(info)
                features["plasmid_origins"].append(info.get("origin", "unknown"))
                features["copy_numbers"].append(info.get("copy_number", "unknown"))
        
        # 检查载体兼容性
        if len(features["plasmid_origins"]) >= 2:
            origins = features["plasmid_origins"]
            for i, origin1 in enumerate(origins):
                for origin2 in origins[i+1:]:
                    if origin1 == origin2:
                        features["is_compatible"] = False
                        break
                    # 检查兼容性矩阵
                    if origin1 in self.plasmid_compatibility:
                        if origin2 not in self.plasmid_compatibility[origin1]:
                            features["is_compatible"] = False
                            break
        
        return features
    
    def _evaluate_match_status(self, analysis: OptionAnalysis) -> Tuple[MatchStatus, str]:
        """Evaluate the match status of an option"""
        # 检查载体兼容性
        if not analysis.technical_features.get("is_compatible", True):
            return MatchStatus.EXCLUDE, "Incompatible plasmid origins"
        
        # Duet载体优先级高
        if analysis.structure_type == OptionStructureType.DUET_VECTOR:
            return MatchStatus.MATCH, "Duet vector provides optimal co-expression in single plasmid"
        
        # 双质粒系统次之
        if analysis.structure_type == OptionStructureType.DUAL_PLASMID:
            if analysis.technical_features.get("is_compatible"):
                return MatchStatus.PARTIAL_MATCH, "Compatible dual plasmid system"
            else:
                return MatchStatus.EXCLUDE, "Incompatible dual plasmid system"
        
        return MatchStatus.NEED_MORE_INFO, "Requires further evaluation"
    
    def _recommend_best_option(self, question_text: str, 
                               analyzed_options: Dict[str, OptionAnalysis],
                               remaining: List[str]) -> Optional[str]:
        """Recommend the best option based on analysis"""
        if not remaining:
            return None
        
        # 评分每个选项
        scores = {}
        for opt_id in remaining:
            analysis = analyzed_options[opt_id]
            score = 0
            
            # Duet载体加分
            if analysis.structure_type == OptionStructureType.DUET_VECTOR:
                score += 100
            
            # 兼容性加分
            if analysis.technical_features.get("is_compatible"):
                score += 50
            
            # 简洁性加分（载体数量少）
            vector_count = len(analysis.vector_names)
            score += max(0, 30 - vector_count * 10)
            
            scores[opt_id] = score
        
        # 返回最高分选项
        if scores:
            return max(scores.keys(), key=lambda x: scores[x])
        return None
    
    def _calculate_confidence(self, analyzed_options: Dict[str, OptionAnalysis],
                              recommended: Optional[str]) -> float:
        """Calculate confidence score for recommendation"""
        if not recommended:
            return 0.0
        
        rec_analysis = analyzed_options.get(recommended)
        if not rec_analysis:
            return 0.0
        
        base_confidence = 0.5
        
        # Duet载体高置信度
        if rec_analysis.structure_type == OptionStructureType.DUET_VECTOR:
            base_confidence += 0.3
        
        # 兼容性高置信度
        if rec_analysis.technical_features.get("is_compatible"):
            base_confidence += 0.1
        
        # 有知识库支持
        if rec_analysis.vector_names:
            known_vectors = [v for v in rec_analysis.vector_names if v in self.vector_knowledge]
            if known_vectors:
                base_confidence += 0.1
        
        return min(1.0, base_confidence)


def analyze_option_semantics(option_text: str) -> Dict[str, Any]:
    """
    Analyze the semantic structure of an option
    
    Returns:
        Dict with entities, relations, structure type, and keywords
    """
    analysis = {
        "entities": [],
        "relations": [],
        "structure": None,
        "keywords": []
    }
    
    # 识别载体类型
    if "Duet" in option_text:
        analysis["structure"] = "duet_vector"
    elif " and " in option_text and "resistance" in option_text:
        # 检查是双质粒还是单一Duet载体
        if "Duet" not in option_text:
            analysis["structure"] = "dual_plasmid"
        else:
            analysis["structure"] = "duet_vector"
    elif len(re.findall(r'p[A-Z][A-Za-z0-9\-]+', option_text)) == 1:
        analysis["structure"] = "single_vector"
    
    # 提取实体
    analysis["entities"] = re.findall(r'p[A-Z][A-Za-z0-9\-]+(?:\([+-]\))?', option_text)
    
    # 提取关键词
    keywords = []
    if "co-expression" in option_text.lower():
        keywords.append("co-expression")
    if "chaperone" in option_text.lower():
        keywords.append("chaperone")
    analysis["keywords"] = keywords
    
    return analysis


def exclude_options_by_constraints(options: Dict[str, str], 
                                   constraints: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Exclude options based on constraints
    
    Args:
        options: Dict mapping option_id to option_text
        constraints: List of constraint strings
    
    Returns:
        Tuple of (remaining_options, excluded_options)
    """
    excluded = []
    remaining = []
    
    for opt_id, opt_text in options.items():
        should_exclude = False
        exclude_reason = ""
        
        for constraint in constraints:
            # 检查约束违反
            if violates_constraint(opt_text, constraint):
                should_exclude = True
                exclude_reason = constraint
                break
        
        if should_exclude:
            excluded.append((opt_id, exclude_reason))
        else:
            remaining.append((opt_id, opt_text))
    
    return remaining, excluded


def violates_constraint(option_text: str, constraint: str) -> bool:
    """Check if an option violates a constraint"""
    option_lower = option_text.lower()
    constraint_lower = constraint.lower()
    
    # 提取约束中的关键词
    constraint_keywords = set(re.findall(r'\b\w+\b', constraint_lower))
    constraint_keywords.discard('not')
    constraint_keywords.discard('cannot')
    constraint_keywords.discard('except')
    
    # 检查否定约束
    if any(neg in constraint_lower for neg in ["cannot", "not", "except", "exclude"]):
        # 如果选项包含约束关键词，则违反
        for keyword in constraint_keywords:
            if keyword in option_lower and len(keyword) > 3:
                return True
    
    return False


def identify_duet_vector_option(options: Dict[str, str]) -> Optional[str]:
    """
    Identify if any option is a Duet vector option
    
    Duet vectors are preferred for co-expression because:
    - Single plasmid = no compatibility issues
    - Dual promoters = controlled co-expression
    - Simple selection = one antibiotic
    """
    for opt_id, opt_text in options.items():
        if "Duet" in opt_text:
            # 确认是单独的Duet载体，而非双质粒系统
            vectors = re.findall(r'p[A-Z][A-Za-z0-9\-]+', opt_text)
            if len(vectors) == 1:  # 只有一个Duet载体
                return opt_id
    
    return None

