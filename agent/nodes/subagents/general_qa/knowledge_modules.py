"""
知识模块系统

模块化的领域知识封装，便于更新和扩展。
"""

from typing import Dict, List, Any, Optional
from .enums import Domain


class KnowledgeModule:
    """知识模块基类"""
    
    def __init__(self, domain: Domain, name: str):
        self.domain = domain
        self.name = name
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据问题和关键信息获取知识上下文
        
        Args:
            question: 用户问题
            key_info: 提取的关键信息
            
        Returns:
            包含知识上下文的字典
        """
        return {
            "domain": self.domain.value,
            "module_name": self.name,
            "relevant_concepts": [],
            "key_facts": [],
            "related_theories": [],
            "experimental_methods": [],
            "common_mistakes": []
        }
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        """
        判断该知识模块是否与问题相关
        
        Args:
            question: 用户问题
            key_info: 提取的关键信息
            
        Returns:
            是否相关
        """
        return False


class BiologyModule(KnowledgeModule):
    """生物学知识模块"""
    
    def __init__(self):
        super().__init__(Domain.BIOLOGY, "biology")
        # 生物学关键词
        self.keywords = [
            "细胞", "基因", "蛋白质", "DNA", "RNA", "转录", "翻译",
            "代谢", "酶", "细胞膜", "细胞核", "线粒体", "叶绿体",
            "细胞分裂", "有丝分裂", "减数分裂", "遗传", "突变"
        ]
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.keywords)
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        context = super().get_context(question, key_info)
        context.update({
            "relevant_concepts": [
                "细胞结构与功能",
                "基因表达调控",
                "蛋白质合成",
                "细胞周期",
                "遗传规律"
            ],
            "key_facts": [
                "中心法则：DNA -> RNA -> 蛋白质",
                "细胞是生命的基本单位",
                "基因是遗传信息的基本单位"
            ],
            "related_theories": [
                "细胞学说",
                "进化论",
                "遗传学定律"
            ],
            "experimental_methods": [
                "PCR",
                "Western Blot",
                "流式细胞术",
                "显微镜观察"
            ],
            "common_mistakes": [
                "混淆转录和翻译",
                "误解基因表达调控机制",
                "忽略实验条件对结果的影响"
            ]
        })
        return context


class ImmunologyModule(KnowledgeModule):
    """免疫学知识模块"""
    
    def __init__(self):
        super().__init__(Domain.IMMUNOLOGY, "immunology")
        self.keywords = [
            "免疫", "抗体", "抗原", "B细胞", "T细胞", "NK细胞",
            "补体", "细胞因子", "免疫应答", "免疫记忆", "疫苗",
            "自身免疫", "过敏", "免疫缺陷", "MHC", "HLA"
        ]
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.keywords)
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        context = super().get_context(question, key_info)
        context.update({
            "relevant_concepts": [
                "适应性免疫与固有免疫",
                "抗体结构与功能",
                "T细胞与B细胞的分化",
                "免疫记忆机制",
                "免疫耐受"
            ],
            "key_facts": [
                "抗体由B细胞产生，具有特异性",
                "T细胞介导细胞免疫，B细胞介导体液免疫",
                "MHC分子参与抗原呈递"
            ],
            "related_theories": [
                "克隆选择理论",
                "危险信号理论",
                "免疫网络理论"
            ],
            "experimental_methods": [
                "ELISA",
                "流式细胞术",
                "免疫组化",
                "免疫印迹"
            ],
            "common_mistakes": [
                "混淆抗体和抗原",
                "误解免疫记忆的机制",
                "忽略免疫系统的复杂性"
            ]
        })
        return context


class ChemistryModule(KnowledgeModule):
    """化学知识模块"""
    
    def __init__(self):
        super().__init__(Domain.CHEMISTRY, "chemistry")
        self.keywords = [
            "化学键", "分子", "原子", "反应", "催化剂", "平衡",
            "pH", "酸", "碱", "氧化", "还原", "有机", "无机",
            "官能团", "反应机理", "立体化学"
        ]
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.keywords)
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        context = super().get_context(question, key_info)
        context.update({
            "relevant_concepts": [
                "化学键理论",
                "反应动力学",
                "化学平衡",
                "酸碱理论",
                "有机反应机理"
            ],
            "key_facts": [
                "化学键决定分子性质",
                "反应速率受温度和催化剂影响",
                "化学平衡遵循勒夏特列原理"
            ],
            "related_theories": [
                "价键理论",
                "分子轨道理论",
                "过渡态理论"
            ],
            "experimental_methods": [
                "光谱分析",
                "色谱法",
                "质谱",
                "核磁共振"
            ],
            "common_mistakes": [
                "混淆反应速率和平衡常数",
                "误解pH和酸碱性的关系",
                "忽略立体化学因素"
            ]
        })
        return context


class MolecularBiologyModule(KnowledgeModule):
    """分子生物学知识模块"""
    
    def __init__(self):
        super().__init__(Domain.MOLECULAR_BIOLOGY, "molecular_biology")
        self.keywords = [
            "PCR", "克隆", "质粒", "限制性内切酶", "连接酶",
            "转录", "翻译", "启动子", "增强子", "沉默子",
            "基因敲除", "基因过表达", "RNA干扰", "CRISPR"
        ]
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.keywords)
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        context = super().get_context(question, key_info)
        context.update({
            "relevant_concepts": [
                "基因工程技术",
                "基因表达调控",
                "表观遗传学",
                "基因编辑技术",
                "蛋白质工程"
            ],
            "key_facts": [
                "PCR用于DNA扩增",
                "限制性内切酶用于DNA切割",
                "CRISPR用于基因编辑"
            ],
            "related_theories": [
                "中心法则",
                "基因表达调控理论",
                "表观遗传学理论"
            ],
            "experimental_methods": [
                "PCR",
                "分子克隆",
                "基因敲除",
                "CRISPR-Cas9"
            ],
            "common_mistakes": [
                "混淆PCR和RT-PCR",
                "误解基因编辑的脱靶效应",
                "忽略实验对照的重要性"
            ]
        })
        return context


class GeneralModule(KnowledgeModule):
    """通用知识模块（默认模块）"""
    
    def __init__(self):
        super().__init__(Domain.GENERAL, "general")
    
    def is_relevant(self, question: str, key_info: Dict[str, Any]) -> bool:
        # 通用模块总是相关的（作为后备）
        return True
    
    def get_context(self, question: str, key_info: Dict[str, Any]) -> Dict[str, Any]:
        context = super().get_context(question, key_info)
        context.update({
            "relevant_concepts": ["通用科学知识"],
            "key_facts": ["需要根据具体问题提供相关信息"],
            "related_theories": [],
            "experimental_methods": [],
            "common_mistakes": []
        })
        return context


# 知识模块注册表
KNOWLEDGE_MODULES: Dict[Domain, List[KnowledgeModule]] = {
    Domain.BIOLOGY: [BiologyModule()],
    Domain.IMMUNOLOGY: [ImmunologyModule()],
    Domain.CHEMISTRY: [ChemistryModule()],
    Domain.MOLECULAR_BIOLOGY: [MolecularBiologyModule()],
    Domain.GENERAL: [GeneralModule()],
}


def get_relevant_modules(question: str, domain: Domain, key_info: Dict[str, Any]) -> List[KnowledgeModule]:
    """
    根据问题和领域获取相关的知识模块
    
    Args:
        question: 用户问题
        domain: 问题领域
        key_info: 提取的关键信息
        
    Returns:
        相关的知识模块列表
    """
    relevant_modules = []
    
    # 获取指定领域的模块
    domain_modules = KNOWLEDGE_MODULES.get(domain, [])
    for module in domain_modules:
        if module.is_relevant(question, key_info):
            relevant_modules.append(module)
    
    # 如果没有找到相关模块，使用通用模块
    if not relevant_modules:
        general_modules = KNOWLEDGE_MODULES.get(Domain.GENERAL, [])
        relevant_modules.extend(general_modules)
    
    return relevant_modules

