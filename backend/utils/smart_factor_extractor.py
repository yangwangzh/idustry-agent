"""
智能因子提取器 - 从Tavily数据中智能提取公司因子
基于NLP技术从高质量报告中提取真实的业务指标
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SmartFactorExtractor:
    """智能因子提取器 - 从Tavily数据中提取真实业务指标"""
    
    def __init__(self):
        """初始化智能因子提取器"""
        self.feature_qubits = 4  # 特征量子比特数量
    
    def extract_factors_from_tavily_data(self, tavily_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从Tavily数据中智能提取因子信息 - 基于NLP的智能分析"""
        factors = []
        report_text = tavily_data.get('report', '')
        
        if not report_text:
            logger.warning("No report text found in Tavily data")
            return self._get_default_factors()
        
        try:
            # 1. 财务因子提取
            financial_factors = self._extract_financial_factors(report_text)
            factors.extend(financial_factors)
            
            # 2. 市场因子提取
            market_factors = self._extract_market_factors(report_text)
            factors.extend(market_factors)
            
            # 3. 竞争因子提取
            competitive_factors = self._extract_competitive_factors(report_text)
            factors.extend(competitive_factors)
            
            # 4. 增长因子提取
            growth_factors = self._extract_growth_factors(report_text)
            factors.extend(growth_factors)
            
            # 5. 技术因子提取
            tech_factors = self._extract_tech_factors(report_text)
            factors.extend(tech_factors)
            
            logger.info(f"✅ 成功提取 {len(factors)} 个智能因子")
            return factors
            
        except Exception as e:
            logger.error(f"智能因子提取失败: {e}")
            return self._get_default_factors()
    
    def _extract_financial_factors(self, report_text: str) -> List[Dict[str, Any]]:
        """提取财务相关因子"""
        factors = []
        
        # 营收规模
        revenue_patterns = [
            r'营收.*?(\d+(?:\.\d+)?)\s*(?:亿|万|千)?(?:美元|元|RMB)?',
            r'收入.*?(\d+(?:\.\d+)?)\s*(?:亿|万|千)?(?:美元|元|RMB)?',
            r'revenue.*?(\d+(?:\.\d+)?)\s*(?:billion|million|thousand)?',
            r'(\d+(?:\.\d+)?)\s*(?:亿|万|千)?(?:美元|元|RMB)?.*?营收',
            r'(\d+(?:\.\d+)?)\s*(?:billion|million|thousand).*?revenue'
        ]
        
        revenue_value = self._extract_numeric_value(report_text, revenue_patterns, "营收规模")
        if revenue_value > 0:
            factors.append({
                "name": "营收规模",
                "value": min(revenue_value / 100.0, 10.0),  # 标准化到0-10
                "weight": 0.25
            })
        
        # 利润率
        profit_patterns = [
            r'利润率.*?(\d+(?:\.\d+)?)%',
            r'毛利率.*?(\d+(?:\.\d+)?)%',
            r'净利率.*?(\d+(?:\.\d+)?)%',
            r'profit margin.*?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%.*?利润率'
        ]
        
        profit_value = self._extract_numeric_value(report_text, profit_patterns, "利润率")
        if profit_value > 0:
            factors.append({
                "name": "利润率",
                "value": min(profit_value / 10.0, 10.0),  # 标准化到0-10
                "weight": 0.25
            })
        
        # 增长率
        growth_patterns = [
            r'增长率.*?(\d+(?:\.\d+)?)%',
            r'增长.*?(\d+(?:\.\d+)?)%',
            r'growth.*?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%.*?增长'
        ]
        
        growth_value = self._extract_numeric_value(report_text, growth_patterns, "增长率")
        if growth_value > 0:
            factors.append({
                "name": "财务增长率",
                "value": min(growth_value / 5.0, 10.0),  # 标准化到0-10
                "weight": 0.2
            })
        
        # 现金流
        cash_flow_indicators = ['现金流', 'cash flow', '经营现金流', 'operating cash flow']
        cash_flow_score = self._calculate_text_importance(report_text, cash_flow_indicators)
        if cash_flow_score > 0:
            factors.append({
                "name": "现金流健康度",
                "value": cash_flow_score,
                "weight": 0.15
            })
        
        # 负债率
        debt_indicators = ['负债率', 'debt ratio', '资产负债率', 'debt-to-equity']
        debt_score = self._calculate_text_importance(report_text, debt_indicators)
        if debt_score > 0:
            # 转换为健康度（负债率越低越好）
            health_score = max(0, 10.0 - debt_score * 5)
            factors.append({
                "name": "财务健康度",
                "value": health_score,
                "weight": 0.15
            })
        
        return factors
    
    def _extract_market_factors(self, report_text: str) -> List[Dict[str, Any]]:
        """提取市场相关因子"""
        factors = []
        
        # 市场份额
        market_share_patterns = [
            r'市场份额.*?(\d+(?:\.\d+)?)%',
            r'market share.*?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%.*?市场份额',
            r'(\d+(?:\.\d+)?)%.*?market share'
        ]
        
        market_share_value = self._extract_numeric_value(report_text, market_share_patterns, "市场份额")
        if market_share_value > 0:
            factors.append({
                "name": "市场份额",
                "value": min(market_share_value / 2.0, 10.0),  # 标准化到0-10
                "weight": 0.3
            })
        
        # 市场地位
        market_position_indicators = [
            '市场领导者', 'market leader', '行业第一', 'leading position',
            '市场领先', 'market leading', '行业龙头', 'dominant position'
        ]
        market_position_score = self._calculate_text_importance(report_text, market_position_indicators)
        if market_position_score > 0:
            factors.append({
                "name": "市场地位",
                "value": market_position_score,
                "weight": 0.25
            })
        
        # 品牌影响力
        brand_indicators = ['品牌', 'brand', '品牌价值', 'brand value', '品牌知名度', 'brand awareness']
        brand_score = self._calculate_text_importance(report_text, brand_indicators)
        if brand_score > 0:
            factors.append({
                "name": "品牌影响力",
                "value": brand_score,
                "weight": 0.2
            })
        
        # 客户满意度
        customer_indicators = ['客户满意度', 'customer satisfaction', '用户满意度', 'user satisfaction']
        customer_score = self._calculate_text_importance(report_text, customer_indicators)
        if customer_score > 0:
            factors.append({
                "name": "客户满意度",
                "value": customer_score,
                "weight": 0.15
            })
        
        # 市场覆盖度
        coverage_indicators = ['全球', 'global', '国际化', 'international', '覆盖', 'coverage']
        coverage_score = self._calculate_text_importance(report_text, coverage_indicators)
        if coverage_score > 0:
            factors.append({
                "name": "市场覆盖度",
                "value": coverage_score,
                "weight": 0.1
            })
        
        return factors
    
    def _extract_competitive_factors(self, report_text: str) -> List[Dict[str, Any]]:
        """提取竞争相关因子"""
        factors = []
        
        # 竞争优势
        competitive_advantage_indicators = [
            '竞争优势', 'competitive advantage', '核心竞争力', 'core competency',
            '差异化', 'differentiation', '独特优势', 'unique advantage'
        ]
        competitive_advantage_score = self._calculate_text_importance(report_text, competitive_advantage_indicators)
        if competitive_advantage_score > 0:
            factors.append({
                "name": "竞争优势",
                "value": competitive_advantage_score,
                "weight": 0.3
            })
        
        # 技术优势
        tech_advantage_indicators = [
            '技术优势', 'technical advantage', '技术创新', 'technology innovation',
            '专利', 'patent', '知识产权', 'intellectual property'
        ]
        tech_advantage_score = self._calculate_text_importance(report_text, tech_advantage_indicators)
        if tech_advantage_score > 0:
            factors.append({
                "name": "技术优势",
                "value": tech_advantage_score,
                "weight": 0.25
            })
        
        # 成本优势
        cost_advantage_indicators = [
            '成本优势', 'cost advantage', '成本控制', 'cost control',
            '规模效应', 'economies of scale', '效率', 'efficiency'
        ]
        cost_advantage_score = self._calculate_text_importance(report_text, cost_advantage_indicators)
        if cost_advantage_score > 0:
            factors.append({
                "name": "成本优势",
                "value": cost_advantage_score,
                "weight": 0.2
            })
        
        # 渠道优势
        channel_advantage_indicators = [
            '渠道', 'channel', '分销网络', 'distribution network',
            '合作伙伴', 'partnership', '生态系统', 'ecosystem'
        ]
        channel_advantage_score = self._calculate_text_importance(report_text, channel_advantage_indicators)
        if channel_advantage_score > 0:
            factors.append({
                "name": "渠道优势",
                "value": channel_advantage_score,
                "weight": 0.15
            })
        
        # 人才优势
        talent_advantage_indicators = [
            '人才', 'talent', '团队', 'team', '专家', 'expert',
            '管理团队', 'management team', '领导力', 'leadership'
        ]
        talent_advantage_score = self._calculate_text_importance(report_text, talent_advantage_indicators)
        if talent_advantage_score > 0:
            factors.append({
                "name": "人才优势",
                "value": talent_advantage_score,
                "weight": 0.1
            })
        
        return factors
    
    def _extract_growth_factors(self, report_text: str) -> List[Dict[str, Any]]:
        """提取增长相关因子"""
        factors = []
        
        # 用户增长率
        user_growth_patterns = [
            r'用户增长.*?(\d+(?:\.\d+)?)%',
            r'用户增长率.*?(\d+(?:\.\d+)?)%',
            r'user growth.*?(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%.*?用户增长'
        ]
        
        user_growth_value = self._extract_numeric_value(report_text, user_growth_patterns, "用户增长率")
        if user_growth_value > 0:
            factors.append({
                "name": "用户增长率",
                "value": min(user_growth_value / 3.0, 10.0),  # 标准化到0-10
                "weight": 0.25
            })
        
        # 产品创新
        innovation_indicators = [
            '创新', 'innovation', '新产品', 'new product', '研发', 'R&D',
            '技术突破', 'breakthrough', '创新产品', 'innovative product'
        ]
        innovation_score = self._calculate_text_importance(report_text, innovation_indicators)
        if innovation_score > 0:
            factors.append({
                "name": "创新能力",
                "value": innovation_score,
                "weight": 0.25
            })
        
        # 国际化程度
        international_indicators = [
            '国际化', 'international', '全球', 'global', '海外', 'overseas',
            '国际市场', 'international market', '全球化', 'globalization'
        ]
        international_score = self._calculate_text_importance(report_text, international_indicators)
        if international_score > 0:
            factors.append({
                "name": "国际化程度",
                "value": international_score,
                "weight": 0.2
            })
        
        # 并购扩张
        expansion_indicators = [
            '并购', 'acquisition', '收购', 'merger', '扩张', 'expansion',
            '投资', 'investment', '战略投资', 'strategic investment'
        ]
        expansion_score = self._calculate_text_importance(report_text, expansion_indicators)
        if expansion_score > 0:
            factors.append({
                "name": "扩张能力",
                "value": expansion_score,
                "weight": 0.15
            })
        
        # 市场拓展
        market_expansion_indicators = [
            '市场拓展', 'market expansion', '新市场', 'new market',
            '业务拓展', 'business expansion', '多元化', 'diversification'
        ]
        market_expansion_score = self._calculate_text_importance(report_text, market_expansion_indicators)
        if market_expansion_score > 0:
            factors.append({
                "name": "市场拓展",
                "value": market_expansion_score,
                "weight": 0.15
            })
        
        return factors
    
    def _extract_tech_factors(self, report_text: str) -> List[Dict[str, Any]]:
        """提取技术相关因子"""
        factors = []
        
        # 技术实力
        tech_capability_indicators = [
            '技术实力', 'technical capability', '技术领先', 'technology leadership',
            '核心技术', 'core technology', '技术优势', 'technical advantage'
        ]
        tech_capability_score = self._calculate_text_importance(report_text, tech_capability_indicators)
        if tech_capability_score > 0:
            factors.append({
                "name": "技术实力",
                "value": tech_capability_score,
                "weight": 0.3
            })
        
        # AI技术
        ai_indicators = [
            '人工智能', 'AI', 'artificial intelligence', '机器学习', 'machine learning',
            '深度学习', 'deep learning', '算法', 'algorithm', '智能', 'intelligent'
        ]
        ai_score = self._calculate_text_importance(report_text, ai_indicators)
        if ai_score > 0:
            factors.append({
                "name": "AI技术",
                "value": ai_score,
                "weight": 0.25
            })
        
        # 数据能力
        data_indicators = [
            '数据', 'data', '大数据', 'big data', '数据分析', 'data analytics',
            '数据挖掘', 'data mining', '数据驱动', 'data-driven'
        ]
        data_score = self._calculate_text_importance(report_text, data_indicators)
        if data_score > 0:
            factors.append({
                "name": "数据能力",
                "value": data_score,
                "weight": 0.2
            })
        
        # 云计算
        cloud_indicators = [
            '云计算', 'cloud computing', '云服务', 'cloud service',
            '云平台', 'cloud platform', '云端', 'cloud'
        ]
        cloud_score = self._calculate_text_importance(report_text, cloud_indicators)
        if cloud_score > 0:
            factors.append({
                "name": "云计算",
                "value": cloud_score,
                "weight": 0.15
            })
        
        # 移动技术
        mobile_indicators = [
            '移动', 'mobile', '移动应用', 'mobile app', '移动平台', 'mobile platform',
            '移动互联网', 'mobile internet', '移动技术', 'mobile technology'
        ]
        mobile_score = self._calculate_text_importance(report_text, mobile_indicators)
        if mobile_score > 0:
            factors.append({
                "name": "移动技术",
                "value": mobile_score,
                "weight": 0.1
            })
        
        return factors
    
    def _extract_numeric_value(self, text: str, patterns: List[str], factor_name: str) -> float:
        """从文本中提取数值"""
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # 取第一个匹配的数值
                    value = float(matches[0])
                    logger.debug(f"提取到 {factor_name}: {value}")
                    return value
                except (ValueError, TypeError):
                    continue
        
        logger.debug(f"未找到 {factor_name} 的数值")
        return 0.0
    
    def _calculate_text_importance(self, text: str, indicators: List[str]) -> float:
        """计算文本中指标的重要性分数"""
        if not text or not indicators:
            return 0.0
        
        text_lower = text.lower()
        total_score = 0.0
        
        for indicator in indicators:
            indicator_lower = indicator.lower()
            # 计算出现次数
            count = text_lower.count(indicator_lower)
            if count > 0:
                # 根据出现次数和文本长度计算重要性
                importance = min(count * 2.0, 10.0)  # 最多10分
                total_score += importance
        
        # 标准化到0-10范围
        normalized_score = min(total_score / len(indicators), 10.0)
        return normalized_score
    
    def _get_default_factors(self) -> List[Dict[str, Any]]:
        """获取默认因子（当智能提取失败时使用）"""
        return [
            {
                "name": "信息丰富度",
                "value": 5.0,
                "weight": 0.2
            },
            {
                "name": "数据可信度", 
                "value": 5.0,
                "weight": 0.25
            },
            {
                "name": "财务健康度",
                "value": 5.0,
                "weight": 0.3
            },
            {
                "name": "市场活跃度",
                "value": 5.0,
                "weight": 0.25
            }
        ]
    
    def extract_features_from_factors(self, factors: List[Dict[str, Any]]) -> List[float]:
        """从因子数据中提取特征向量"""
        import numpy as np
        
        features = []

        # 提取主要特征
        for factor in factors:
            value = factor.get('value', 0.0)
            weight = factor.get('weight', 0.0)

            # 特征工程：结合值和权重
            weighted_value = value * weight
            features.append(weighted_value)

        # 填充到固定长度
        while len(features) < self.feature_qubits:
            features.append(0.0)

        # 标准化到 [0, 2π] 范围（适合角度编码）
        features = np.array(features[:self.feature_qubits])
        if np.max(np.abs(features)) > 0:
            features = (features - np.min(features)) / (np.max(features) - np.min(features)) * 2 * np.pi

        return features.tolist() 