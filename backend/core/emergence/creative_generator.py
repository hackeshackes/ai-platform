"""
Creative Generator - 创意生成器
创造性问题解决、新策略发现、跨模态创新、艺术创作
"""

import random
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np


class CreativeMode(Enum):
    """创意模式"""
    DIVERGENT = "divergent"          # 发散思维
    CONVERGENT = "convergent"       # 收敛思维
    LATERAL = "lateral"             # 横向思维
    ANALOGICAL = "analogical"       # 类比思维
    ASSOCIATIVE = "associative"      # 联想思维
    ABSTRACT = "abstract"           # 抽象思维


class InnovationType(Enum):
    """创新类型"""
    INCREMENTAL = "incremental"     # 渐进式创新
    RADICAL = "radical"             # 根本性创新
    DISRUPTIVE = "disruptive"       # 颠覆式创新
    ARCHITECTURAL = "architectural" # 架构式创新


@dataclass
class CreativeSolution:
    """创意解决方案"""
    solution_id: str
    description: str
    innovation_type: InnovationType
    confidence: float
    novelty_score: float
    applicability: float
    steps: List[str]
    key_insights: List[str]
    analogies: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IdeaChain:
    """创意链"""
    chain_id: str
    ideas: List[Dict]
    chain_length: int
    coherence: float
    creativity_score: float


class CreativeGenerator:
    """
    创意生成器
    实现创造性问题解决、新策略发现、跨模态创新和艺术创作
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.idea_history: List[CreativeSolution] = []
        self.concept_graph: Dict[str, Set[str]] = defaultdict(set)
        self.analogy_cache: Dict[str, List[Dict]] = {}
        self.mode = CreativeMode.DIVERGENT
        
    def _default_config(self) -> Dict:
        return {
            'max_solutions': 10,
            'novelty_threshold': 0.6,
            'creativity_weight': 0.4,
            'analogy_depth': 3,
            'chain_length': 5,
            'temperature': 0.8,
            'max_iterations': 100
        }
    
    def solve(self, problem: Dict, constraints: Optional[Dict] = None) -> List[CreativeSolution]:
        """
        创造性解决问题
        
        Args:
            problem: 问题描述
            constraints: 约束条件
            
        Returns:
            创意解决方案列表
        """
        constraints = constraints or {}
        
        # 识别问题类型
        problem_type = self._classify_problem(problem)
        
        # 选择创意模式
        mode = self._select_mode(problem_type)
        
        # 生成初始创意集
        initial_ideas = self._generate_initial_ideas(problem, constraints)
        
        # 应用创意模式进行演化
        evolved_ideas = self._evolve_ideas(initial_ideas, mode, constraints)
        
        # 跨模态创新
        cross_modal_ideas = self._cross_modal_innovation(evolved_ideas, problem)
        
        # 评估和筛选
        solutions = self._evaluate_solutions(cross_modal_ideas, constraints)
        
        # 记录到历史
        self.idea_history.extend(solutions)
        
        # 更新概念图
        for solution in solutions:
            self._update_concept_graph(solution)
        
        return solutions[:self.config['max_solutions']]
    
    def _classify_problem(self, problem: Dict) -> str:
        """分类问题类型"""
        problem_text = str(problem).lower()
        
        if 'optimize' in problem_text or 'improve' in problem_text:
            return 'optimization'
        elif 'create' in problem_text or 'design' in problem_text:
            return 'design'
        elif 'solve' in problem_text or 'fix' in problem_text:
            return 'problem_solving'
        elif 'discover' in problem_text or 'find' in problem_text:
            return 'discovery'
        else:
            return 'general'
    
    def _select_mode(self, problem_type: str) -> CreativeMode:
        """选择创意模式"""
        mode_map = {
            'optimization': CreativeMode.CONVERGENT,
            'design': CreativeMode.LATERAL,
            'problem_solving': CreativeMode.ANALOGICAL,
            'discovery': CreativeMode.ABSTRACT,
            'general': CreativeMode.DIVERGENT
        }
        return mode_map.get(problem_type, CreativeMode.DIVERGENT)
    
    def _generate_initial_ideas(self, problem: Dict, constraints: Dict) -> List[Dict]:
        """生成初始创意"""
        ideas = []
        
        # 基于问题分解生成
        sub_problems = self._decompose_problem(problem)
        
        for i, sub in enumerate(sub_problems):
            idea = {
                'id': f"idea_{i}",
                'sub_problem': sub,
                'approach': self._generate_approach(sub),
                'components': [],
                'raw_novelty': random.uniform(0.5, 0.9)
            }
            ideas.append(idea)
        
        # 生成组合方案
        for i in range(len(ideas)):
            for j in range(i + 1, len(ideas)):
                combo = {
                    'id': f"combo_{i}_{j}",
                    'sub_problem': [ideas[i]['sub_problem'], ideas[j]['sub_problem']],
                    'approach': f"{ideas[i]['approach']} + {ideas[j]['approach']}",
                    'components': [ideas[i], ideas[j]],
                    'raw_novelty': (ideas[i]['raw_novelty'] + ideas[j]['raw_novelty']) / 2 + 0.1
                }
                ideas.append(combo)
        
        return ideas
    
    def _decompose_problem(self, problem: Dict) -> List[str]:
        """分解问题"""
        problem_str = str(problem)
        
        # 简单分解
        parts = problem_str.split(',')
        if len(parts) < 2:
            parts = problem_str.split('.')
            
        return [p.strip() for p in parts if p.strip()]
    
    def _generate_approach(self, sub_problem: str) -> str:
        """生成方法"""
        approaches = [
            "分解为更小问题",
            "逆向思考",
            "寻找类比",
            "约束放宽",
            "随机组合",
            "层次抽象",
            "多角度分析",
            "约束反转"
        ]
        return random.choice(approaches)
    
    def _evolve_ideas(self, ideas: List[Dict], mode: CreativeMode, 
                     constraints: Dict) -> List[Dict]:
        """演化创意"""
        evolved = ideas.copy()
        
        for iteration in range(self.config['max_iterations']):
            # 确保有足够的父代
            if len(evolved) < 2:
                break
                
            parent1, parent2 = random.sample(evolved, 2)
            child = self._crossover(parent1, parent2, mode)
            child = self._mutate(child, mode, constraints)
            fitness = self._calculate_fitness(child, constraints)
            child['fitness'] = fitness
            
            if fitness > 0.5 or random.random() < self.config['temperature']:
                evolved.append(child)
        
        return evolved
    
    def _crossover(self, parent1: Dict, parent2: Dict, mode: CreativeMode) -> Dict:
        """交叉操作"""
        child = {
            'id': f"child_{random.randint(0, 9999)}",
            'approach': '',
            'raw_novelty': (parent1['raw_novelty'] + parent2['raw_novelty']) / 2
        }
        
        if mode == CreativeMode.LATERAL:
            child['approach'] = random.choice([
                parent1['approach'].split('+')[0] if '+' in parent1['approach'] else parent1['approach'],
                parent2['approach'].split('+')[0] if '+' in parent2['approach'] else parent2['approach']
            ])
        else:
            child['approach'] = f"{parent1['approach']} & {parent2['approach']}"
        
        return child
    
    def _mutate(self, idea: Dict, mode: CreativeMode, constraints: Dict) -> Dict:
        """变异操作"""
        mutation_strength = 1 - self.config['temperature']
        
        if random.random() < mutation_strength:
            new_approach = self._generate_approach(str(idea.get('sub_problem', '')))
            if '+' in idea['approach']:
                idea['approach'] += f" + {new_approach}"
            else:
                idea['approach'] = f"{idea['approach']} + {new_approach}"
        
        idea['raw_novelty'] = min(1.0, idea['raw_novelty'] + random.uniform(-0.1, 0.1))
        return idea
    
    def _calculate_fitness(self, idea: Dict, constraints: Dict) -> float:
        """计算适应度"""
        novelty = idea.get('raw_novelty', 0.5)
        complexity = len(idea.get('approach', '')) / 100.0
        feasibility = 1.0 - min(complexity, 1.0)
        
        return novelty * 0.4 + feasibility * 0.6
    
    def _cross_modal_innovation(self, ideas: List[Dict], problem: Dict) -> List[Dict]:
        """跨模态创新"""
        cross_modal = []
        
        domains = ['视觉', '听觉', '语言', '逻辑', '空间', '运动']
        
        for idea in ideas:
            for domain in random.sample(domains, k=3):
                cross_idea = {
                    'id': f"cross_{idea['id']}_{domain}",
                    'original_idea': idea,
                    'domain': domain,
                    'cross_modal_approach': f"从{domain}角度重新思考: {idea['approach']}",
                    'raw_novelty': idea['raw_novelty'] + 0.1
                }
                cross_modal.append(cross_idea)
        
        return ideas + cross_modal
    
    def _evaluate_solutions(self, ideas: List[Dict], constraints: Dict) -> List[CreativeSolution]:
        """评估和筛选解决方案"""
        solutions = []
        
        for i, idea in enumerate(ideas):
            approach = idea.get('approach', f'方案{i+1}')
            novelty_score = idea.get('raw_novelty', 0.5)
            innovation_type = self._classify_innovation(novelty_score)
            confidence = idea.get('fitness', 0.5)
            applicability = self._calculate_applicability(idea, constraints)
            
            solution = CreativeSolution(
                solution_id=f"solution_{i}",
                description=f"方案{i+1}: {approach}",
                innovation_type=innovation_type,
                confidence=confidence,
                novelty_score=novelty_score,
                applicability=applicability,
                steps=self._generate_steps(idea),
                key_insights=self._extract_insights(idea),
                analogies=self._find_analogies(idea)
            )
            solutions.append(solution)
        
        # 按综合分数排序
        solutions.sort(key=lambda x: x.confidence * 0.5 + x.novelty_score * 0.3 + x.applicability * 0.2, 
                      reverse=True)
        
        return solutions
    
    def _classify_innovation(self, novelty_score: float) -> InnovationType:
        """分类创新类型"""
        if novelty_score > 0.8:
            return InnovationType.RADICAL
        elif novelty_score > 0.6:
            return InnovationType.DISRUPTIVE
        elif novelty_score > 0.4:
            return InnovationType.INCREMENTAL
        else:
            return InnovationType.ARCHITECTURAL
    
    def _calculate_applicability(self, idea: Dict, constraints: Dict) -> float:
        """计算适用性"""
        if not constraints:
            return 0.7
            
        applicability = 1.0
        for key, value in constraints.items():
            if key.lower() in str(idea).lower():
                applicability *= 0.9
                
        return min(applicability, 1.0)
    
    def _generate_steps(self, idea: Dict) -> List[str]:
        """生成执行步骤"""
        steps = [
            f"分析问题: {idea.get('sub_problem', '未知')}",
            f"应用方法: {idea.get('approach', '待定')}",
            "进行原型设计",
            "测试和迭代",
            "评估结果"
        ]
        return steps
    
    def _extract_insights(self, idea: Dict) -> List[str]:
        """提取关键洞察"""
        return [
            f"核心方法: {idea.get('approach', 'N/A')}",
            "需要的关键资源",
            "潜在的风险点"
        ]
    
    def _find_analogies(self, idea: Dict) -> List[str]:
        """寻找类比"""
        analogies = [
            "类似自然界中的自组织系统",
            "类似于社交网络的传播机制",
            "类似于免疫系统的适应能力"
        ]
        return random.sample(analogies, k=min(2, len(analogies)))
    
    def _update_concept_graph(self, solution: CreativeSolution):
        """更新概念图"""
        for insight in solution.key_insights:
            words = insight.split(':')[0] if ':' in insight else insight
            self.concept_graph[words].add(solution.solution_id)
    
    def discover_strategy(self, context: Dict) -> Dict:
        """发现新策略"""
        strategies = []
        
        # 分析历史策略
        historical_patterns = self._analyze_patterns()
        
        # 识别机会
        opportunities = self._identify_opportunities(context)
        
        for opp in opportunities:
            strategy = {
                'opportunity': opp,
                'approach': f"利用{opp}的新策略",
                'expected_impact': random.uniform(0.6, 0.95),
                'novelty': random.uniform(0.5, 0.9)
            }
            strategies.append(strategy)
        
        return {
            'strategies': strategies,
            'pattern_analysis': historical_patterns,
            'opportunities': opportunities
        }
    
    def _analyze_patterns(self) -> Dict:
        """分析模式"""
        return {
            'trend': 'increasing_complexity',
            'dominant_approach': 'combinatorial',
            'success_rate': 0.72
        }
    
    def _identify_opportunities(self, context: Dict) -> List[str]:
        """识别机会"""
        return [
            '自动化重复任务',
            '跨领域知识整合',
            '用户需求深层挖掘'
        ]
    
    def create_artistic_content(self, theme: str, style: str) -> Dict:
        """艺术创作"""
        return {
            'theme': theme,
            'style': style,
            'elements': [
                f"{theme}的视觉元素",
                f"{style}的表现手法",
                "创新组合"
            ],
            'creativity_score': random.uniform(0.7, 0.95),
            'novelty_score': random.uniform(0.6, 0.9)
        }
    
    def get_idea_history(self) -> List[Dict]:
        """获取创意历史"""
        return [
            {
                'id': sol.solution_id,
                'description': sol.description,
                'innovation_type': sol.innovation_type.value,
                'confidence': sol.confidence,
                'novelty': sol.novelty_score
            }
            for sol in self.idea_history
        ]
