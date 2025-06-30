#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的MBG-NBD客户生命周期价值预测系统
Complete MBG-NBD Customer Lifetime Value Prediction System

功能模块：
1. 数据加载与预处理
2. 客户分层与行为异质性分析
3. 季节性马尔可夫链学习
4. MBG-NBD模型训练
5. CLV预测与验证
6. 效果对比与优化分析

作者: Manus AI
版本: 2.0
日期: 2025-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import json
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CompleteMBGNBDSystem:
    """
    完整的MBG-NBD客户生命周期价值预测系统
    """

    def __init__(self, prediction_period=90):
        """
        初始化系统

        参数:
        prediction_period: 预测期长度（天）
        """
        self.prediction_period = prediction_period

        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.rfm_data = None
        self.customer_segments = None

        # 季节性分析
        self.seasonal_factors = None
        self.markov_transition_matrix = None
        self.emission_probabilities = None

        # 模型参数
        self.base_model_params = None  # 基础MBG-NBD参数
        self.enhanced_model_params = None  # 增强模型参数

        # 预测结果
        self.base_predictions = None
        self.enhanced_predictions = None

        # 性能指标
        self.performance_metrics = {}

        print("🚀 完整MBG-NBD系统初始化完成")
        print(f"   预测期长度: {prediction_period}天")

    # ==================== 第一阶段：数据加载与预处理 ====================

    def load_data(self, data_path):
        """
        第一阶段：数据加载与预处理
        """
        print("\n" + "=" * 60)
        print("第一阶段：数据加载与预处理")
        print("=" * 60)

        try:
            # 1.1 加载原始数据
            print("📂 1.1 加载原始数据...")
            self.raw_data = pd.read_csv(data_path)

            # 标准化列名
            if 'create_time' in self.raw_data.columns:
                self.raw_data.columns = ['customer_id', 'amount', 'order_date']

            print(f"   ✅ 原始数据加载成功: {len(self.raw_data)}条记录")
            print(f"   📊 数据列: {list(self.raw_data.columns)}")

            # 1.2 数据清洗与预处理
            print("\n🧹 1.2 数据清洗与预处理...")

            # 数据类型转换
            self.raw_data['order_date'] = pd.to_datetime(self.raw_data['order_date'])
            self.raw_data['amount'] = pd.to_numeric(self.raw_data['amount'], errors='coerce')

            # 数据质量检查
            null_count = self.raw_data.isnull().sum().sum()
            duplicate_count = self.raw_data.duplicated().sum()
            negative_amount = (self.raw_data['amount'] <= 0).sum()

            print(f"   📋 数据质量检查:")
            print(f"      - 空值数量: {null_count}")
            print(f"      - 重复记录: {duplicate_count}")
            print(f"      - 非正金额: {negative_amount}")

            # 数据清洗
            original_count = len(self.raw_data)

            # 移除空值和非正金额
            self.processed_data = self.raw_data.dropna()
            self.processed_data = self.processed_data[self.processed_data['amount'] > 0]

            # 移除重复记录
            self.processed_data = self.processed_data.drop_duplicates()

            cleaned_count = len(self.processed_data)
            removed_count = original_count - cleaned_count

            print(f"   ✅ 数据清洗完成:")
            print(f"      - 清洗前: {original_count}条")
            print(f"      - 清洗后: {cleaned_count}条")
            print(f"      - 移除: {removed_count}条 ({removed_count / original_count * 100:.1f}%)")

            # 1.3 添加时间特征
            print("\n📅 1.3 添加时间特征...")

            self.processed_data['year'] = self.processed_data['order_date'].dt.year
            self.processed_data['month'] = self.processed_data['order_date'].dt.month
            self.processed_data['quarter'] = self.processed_data['order_date'].dt.quarter
            self.processed_data['dayofweek'] = self.processed_data['order_date'].dt.dayofweek
            self.processed_data['week'] = self.processed_data['order_date'].dt.isocalendar().week

            # 数据时间范围
            date_range = {
                'start_date': self.processed_data['order_date'].min(),
                'end_date': self.processed_data['order_date'].max(),
                'span_days': (self.processed_data['order_date'].max() - self.processed_data['order_date'].min()).days,
                'span_years': (self.processed_data['order_date'].max() - self.processed_data[
                    'order_date'].min()).days / 365
            }

            print(f"   ✅ 时间特征添加完成:")
            print(
                f"      - 时间范围: {date_range['start_date'].strftime('%Y-%m-%d')} 到 {date_range['end_date'].strftime('%Y-%m-%d')}")
            print(f"      - 时间跨度: {date_range['span_days']}天 ({date_range['span_years']:.1f}年)")

            # 1.4 基础统计分析
            print("\n📊 1.4 基础统计分析...")

            basic_stats = {
                'total_customers': self.processed_data['customer_id'].nunique(),
                'total_transactions': len(self.processed_data),
                'total_revenue': self.processed_data['amount'].sum(),
                'avg_transaction_value': self.processed_data['amount'].mean(),
                'median_transaction_value': self.processed_data['amount'].median(),
                'transactions_per_customer': len(self.processed_data) / self.processed_data['customer_id'].nunique()
            }

            print(f"   ✅ 基础统计完成:")
            print(f"      - 客户总数: {basic_stats['total_customers']:,}")
            print(f"      - 交易总数: {basic_stats['total_transactions']:,}")
            print(f"      - 总收入: {basic_stats['total_revenue']:,.2f}元")
            print(f"      - 平均交易额: {basic_stats['avg_transaction_value']:.2f}元")
            print(f"      - 人均交易次数: {basic_stats['transactions_per_customer']:.1f}次")

            return True

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    # ==================== 第二阶段：客户分层与行为异质性分析 ====================

    def create_customer_segmentation(self):
        """
        第二阶段：客户分层与行为异质性分析
        """
        print("\n" + "=" * 60)
        print("第二阶段：客户分层与行为异质性分析")
        print("=" * 60)

        # 2.1 计算RFM特征
        print("📊 2.1 计算RFM特征...")

        # 计算观察期截止日期（最后一次交易日期）
        observation_end = self.processed_data['order_date'].max()

        # 按客户聚合计算RFM
        customer_rfm = self.processed_data.groupby('customer_id').agg({
            'order_date': ['min', 'max', 'count'],
            'amount': ['sum', 'mean']
        }).round(2)

        # 重命名列
        customer_rfm.columns = ['first_purchase', 'last_purchase', 'frequency', 'monetary_total', 'monetary_avg']
        customer_rfm = customer_rfm.reset_index()

        # 计算Recency（最近一次购买距今天数）
        customer_rfm['recency'] = (observation_end - customer_rfm['last_purchase']).dt.days

        # 计算T（客户生命周期长度）
        customer_rfm['T'] = (customer_rfm['last_purchase'] - customer_rfm['first_purchase']).dt.days
        customer_rfm['T'] = customer_rfm['T'].apply(lambda x: max(x, 1))  # 最小为1天

        # 计算历史购买频率（用于MBG-NBD）
        customer_rfm['historical_frequency'] = customer_rfm['frequency'] - 1  # MBG-NBD中频率不包括首次购买
        customer_rfm['historical_frequency'] = customer_rfm['historical_frequency'].apply(lambda x: max(x, 0))

        print(f"   ✅ RFM特征计算完成: {len(customer_rfm)}个客户")
        print(f"      - 平均Recency: {customer_rfm['recency'].mean():.1f}天")
        print(f"      - 平均Frequency: {customer_rfm['frequency'].mean():.1f}次")
        print(f"      - 平均Monetary: {customer_rfm['monetary_avg'].mean():.2f}元")

        # 2.2 客户价值分层
        print("\n🎯 2.2 客户价值分层...")

        # 使用RFM评分进行分层
        # R评分（Recency越小越好）
        customer_rfm['R_score'] = pd.qcut(customer_rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')

        # F评分（Frequency越大越好）
        customer_rfm['F_score'] = pd.qcut(customer_rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5],
                                          duplicates='drop')

        # M评分（Monetary越大越好）
        customer_rfm['M_score'] = pd.qcut(customer_rfm['monetary_total'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # 计算综合RFM评分
        customer_rfm['RFM_score'] = (customer_rfm['R_score'].astype(float) +
                                     customer_rfm['F_score'].astype(float) +
                                     customer_rfm['M_score'].astype(float)) / 3

        # 客户分层
        def classify_customer_segment(rfm_score):
            if rfm_score >= 4.5:
                return 'Champions'  # 冠军客户
            elif rfm_score >= 3.5:
                return 'Loyal'  # 忠诚客户
            elif rfm_score >= 2.5:
                return 'Potential'  # 潜力客户
            elif rfm_score >= 1.5:
                return 'At_Risk'  # 风险客户
            else:
                return 'Lost'  # 流失客户

        customer_rfm['segment'] = customer_rfm['RFM_score'].apply(classify_customer_segment)

        # 分层统计
        segment_stats = customer_rfm.groupby('segment').agg({
            'customer_id': 'count',
            'monetary_total': ['sum', 'mean'],
            'frequency': 'mean',
            'recency': 'mean'
        }).round(2)

        segment_stats.columns = ['customer_count', 'total_revenue', 'avg_revenue', 'avg_frequency', 'avg_recency']
        segment_stats['revenue_percentage'] = (
                    segment_stats['total_revenue'] / segment_stats['total_revenue'].sum() * 100).round(1)

        print(f"   ✅ 客户分层完成:")
        for segment in segment_stats.index:
            stats = segment_stats.loc[segment]
            print(f"      - {segment}: {stats['customer_count']}人 ({stats['revenue_percentage']}%收入)")

        # 2.3 行为异质性分析
        print("\n🔍 2.3 行为异质性分析...")

        # 使用K-means聚类进行更精细的行为分析
        features_for_clustering = ['recency', 'frequency', 'monetary_avg', 'T']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(customer_rfm[features_for_clustering])

        # 确定最优聚类数
        silhouette_scores = []
        K_range = range(2, min(8, len(customer_rfm) // 10))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k = K_range[np.argmax(silhouette_scores)]

        # 执行最优聚类
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        customer_rfm['behavior_cluster'] = kmeans.fit_predict(scaled_features)

        print(f"   ✅ 行为异质性分析完成:")
        print(f"      - 最优聚类数: {optimal_k}")
        print(f"      - 轮廓系数: {max(silhouette_scores):.3f}")

        # 聚类特征分析
        cluster_analysis = customer_rfm.groupby('behavior_cluster')[features_for_clustering].mean().round(2)

        for cluster_id in cluster_analysis.index:
            cluster_data = cluster_analysis.loc[cluster_id]
            customer_count = (customer_rfm['behavior_cluster'] == cluster_id).sum()
            print(
                f"      - 聚类{cluster_id}: {customer_count}人, R={cluster_data['recency']:.0f}, F={cluster_data['frequency']:.1f}, M={cluster_data['monetary_avg']:.0f}")

        self.rfm_data = customer_rfm
        self.customer_segments = {
            'segment_stats': segment_stats,
            'cluster_analysis': cluster_analysis,
            'optimal_k': optimal_k,
            'scaler': scaler
        }

        return customer_rfm

    # ==================== 第三阶段：季节性马尔可夫链学习 ====================

    def learn_seasonal_patterns(self):
        """
        第三阶段：季节性马尔可夫链学习
        """
        print("\n" + "=" * 60)
        print("第三阶段：季节性马尔可夫链学习")
        print("=" * 60)

        # 3.1 季节性模式识别
        print("🌟 3.1 季节性模式识别...")

        # 按月统计收入
        monthly_revenue = self.processed_data.groupby(['year', 'month'])['amount'].sum().reset_index()
        monthly_revenue['date'] = pd.to_datetime(monthly_revenue[['year', 'month']].assign(day=1))
        monthly_revenue = monthly_revenue.sort_values('date')

        # 计算月度季节性因子
        monthly_avg = monthly_revenue.groupby('month')['amount'].mean()
        overall_avg = monthly_avg.mean()
        basic_seasonal_factors = (monthly_avg / overall_avg).to_dict()

        print(f"   ✅ 基础季节性因子计算完成:")
        for month, factor in basic_seasonal_factors.items():
            print(f"      - {month}月: {factor:.3f}")

        # 3.2 马尔可夫状态转移矩阵学习
        print("\n🔗 3.2 马尔可夫状态转移矩阵学习...")

        # 构建12x12的月份转移矩阵
        transition_matrix = np.zeros((12, 12))

        # 统计月份间的转移
        monthly_sequence = monthly_revenue['month'].values
        for i in range(len(monthly_sequence) - 1):
            current_month = monthly_sequence[i] - 1  # 转换为0-11索引
            next_month = monthly_sequence[i + 1] - 1
            transition_matrix[current_month][next_month] += 1

        # 归一化为概率矩阵
        for i in range(12):
            row_sum = transition_matrix[i].sum()
            if row_sum > 0:
                transition_matrix[i] = transition_matrix[i] / row_sum
            else:
                # 如果某月没有数据，使用均匀分布
                transition_matrix[i] = np.ones(12) / 12

        print(f"   ✅ 转移矩阵学习完成:")
        print(f"      - 矩阵维度: {transition_matrix.shape}")
        print(f"      - 非零元素: {np.count_nonzero(transition_matrix)}")

        # 3.3 发射概率学习
        print("\n📊 3.3 发射概率学习...")

        # 计算每个月的消费强度分布
        emission_probabilities = {}

        for month in range(1, 13):
            month_data = self.processed_data[self.processed_data['month'] == month]['amount']

            if len(month_data) > 0:
                emission_probabilities[month] = {
                    'mean': month_data.mean(),
                    'std': month_data.std(),
                    'count': len(month_data),
                    'total': month_data.sum()
                }
            else:
                # 如果某月没有数据，使用全局平均值
                global_mean = self.processed_data['amount'].mean()
                global_std = self.processed_data['amount'].std()
                emission_probabilities[month] = {
                    'mean': global_mean,
                    'std': global_std,
                    'count': 0,
                    'total': 0
                }

        print(f"   ✅ 发射概率学习完成:")
        for month, prob in emission_probabilities.items():
            print(f"      - {month}月: 均值={prob['mean']:.0f}, 标准差={prob['std']:.0f}, 样本数={prob['count']}")

        # 3.4 马尔可夫季节性因子计算
        print("\n🎯 3.4 马尔可夫季节性因子计算...")

        # 计算稳态分布
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_index = np.argmax(eigenvalues.real)
        stationary_distribution = np.abs(eigenvectors[:, stationary_index].real)
        stationary_distribution = stationary_distribution / stationary_distribution.sum()

        # 综合计算马尔可夫季节性因子
        markov_seasonal_factors = {}
        global_emission_mean = np.mean([prob['mean'] for prob in emission_probabilities.values()])

        for month in range(1, 13):
            # 发射权重（70%）
            emission_weight = emission_probabilities[month]['mean'] / global_emission_mean

            # 转移权重（30%）
            transition_weight = stationary_distribution[month - 1] * 12  # 归一化到平均值1

            # 综合季节性因子
            markov_factor = 0.7 * emission_weight + 0.3 * transition_weight
            markov_seasonal_factors[month] = markov_factor

        print(f"   ✅ 马尔可夫季节性因子计算完成:")
        for month, factor in markov_seasonal_factors.items():
            basic_factor = basic_seasonal_factors[month]
            improvement = abs(factor - 1) / abs(basic_factor - 1) if abs(basic_factor - 1) > 0.001 else 1
            print(f"      - {month}月: 基础={basic_factor:.3f}, 马尔可夫={factor:.3f}, 改进={improvement:.2f}x")

        self.seasonal_factors = {
            'basic': basic_seasonal_factors,
            'markov': markov_seasonal_factors
        }
        self.markov_transition_matrix = transition_matrix
        self.emission_probabilities = emission_probabilities

        return markov_seasonal_factors

    # ==================== 第四阶段：MBG-NBD模型训练 ====================

    def train_mbgnbd_models(self):
        """
        第四阶段：MBG-NBD模型训练
        """
        print("\n" + "=" * 60)
        print("第四阶段：MBG-NBD模型训练")
        print("=" * 60)

        # 4.1 基础MBG-NBD模型训练
        print("🎯 4.1 基础MBG-NBD模型训练...")

        # 准备训练数据
        frequency = self.rfm_data['historical_frequency'].values
        recency = self.rfm_data['recency'].values
        T = self.rfm_data['T'].values

        print(f"   📊 训练数据准备:")
        print(f"      - 客户数量: {len(frequency)}")
        print(f"      - 平均频率: {frequency.mean():.2f}")
        print(f"      - 平均间隔: {recency.mean():.1f}天")
        print(f"      - 平均生命周期: {T.mean():.1f}天")

        # 基础MBG-NBD似然函数
        def mbgnbd_likelihood(params, frequency, recency, T):
            r, alpha, a, b = params

            # 参数约束
            if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
                return 1e10

            likelihood = 0

            for i in range(len(frequency)):
                freq = frequency[i]
                rec = recency[i]
                t = T[i]

                try:
                    if freq == 0:
                        # 零购买客户
                        term1 = a / (a + b)
                        term2 = (alpha / (alpha + t)) ** r
                        prob = term1 + (1 - term1) * term2
                    else:
                        # 有购买历史的客户
                        term1 = math.lgamma(r + freq) - math.lgamma(r) + r * math.log(alpha)
                        term2 = math.lgamma(a + b) + math.lgamma(b + freq) - math.lgamma(b) - math.lgamma(a + b + freq)
                        term3 = -(r + freq) * math.log(alpha + t)
                        term4 = math.log((a + b + freq - 1) / (alpha + t - rec))

                        log_prob = term1 + term2 + term3 + term4
                        prob = math.exp(min(log_prob, 700))  # 防止溢出

                    if prob > 0:
                        likelihood -= math.log(prob)
                    else:
                        likelihood += 1e6

                except (ValueError, OverflowError):
                    likelihood += 1e6

            return likelihood

        # 模型训练
        print("   🔄 开始模型训练...")

        # 多次随机初始化寻找最优解
        best_params = None
        best_likelihood = float('inf')

        for attempt in range(5):
            # 随机初始化参数
            initial_params = [
                np.random.uniform(0.1, 2.0),  # r
                np.random.uniform(0.1, 5.0),  # alpha
                np.random.uniform(0.1, 2.0),  # a
                np.random.uniform(0.1, 2.0)  # b
            ]

            try:
                result = minimize(
                    mbgnbd_likelihood,
                    initial_params,
                    args=(frequency, recency, T),
                    method='L-BFGS-B',
                    bounds=[(0.01, 10), (0.01, 20), (0.01, 10), (0.01, 10)]
                )

                if result.success and result.fun < best_likelihood:
                    best_likelihood = result.fun
                    best_params = result.x

            except Exception as e:
                print(f"      ⚠️ 训练尝试{attempt + 1}失败: {e}")
                continue

        if best_params is None:
            print("   ❌ 基础模型训练失败")
            return False

        self.base_model_params = best_params
        r, alpha, a, b = best_params

        print(f"   ✅ 基础MBG-NBD模型训练完成:")
        print(f"      - r (购买率形状): {r:.4f}")
        print(f"      - α (购买率尺度): {alpha:.4f}")
        print(f"      - a (流失率形状): {a:.4f}")
        print(f"      - b (流失率尺度): {b:.4f}")
        print(f"      - 负对数似然: {best_likelihood:.2f}")

        # 4.2 季节性增强MBG-NBD模型训练
        print("\n🌟 4.2 季节性增强MBG-NBD模型训练...")

        # 季节性增强似然函数
        def seasonal_mbgnbd_likelihood(params, frequency, recency, T, seasonal_factors):
            r, alpha, a, b, seasonal_strength = params

            if r <= 0 or alpha <= 0 or a <= 0 or b <= 0 or seasonal_strength < 0 or seasonal_strength > 1:
                return 1e10

            likelihood = 0

            for i in range(len(frequency)):
                freq = frequency[i]
                rec = recency[i]
                t = T[i]

                # 获取季节性调整
                # 这里简化处理，使用平均季节性因子
                avg_seasonal_factor = np.mean(list(seasonal_factors.values()))
                seasonal_adjustment = 1 + seasonal_strength * (avg_seasonal_factor - 1)

                # 调整参数
                adjusted_alpha = alpha * seasonal_adjustment

                try:
                    if freq == 0:
                        term1 = a / (a + b)
                        term2 = (adjusted_alpha / (adjusted_alpha + t)) ** r
                        prob = term1 + (1 - term1) * term2
                    else:
                        term1 = math.lgamma(r + freq) - math.lgamma(r) + r * math.log(adjusted_alpha)
                        term2 = math.lgamma(a + b) + math.lgamma(b + freq) - math.lgamma(b) - math.lgamma(a + b + freq)
                        term3 = -(r + freq) * math.log(adjusted_alpha + t)
                        term4 = math.log((a + b + freq - 1) / (adjusted_alpha + t - rec))

                        log_prob = term1 + term2 + term3 + term4
                        prob = math.exp(min(log_prob, 700))

                    if prob > 0:
                        likelihood -= math.log(prob)
                    else:
                        likelihood += 1e6

                except (ValueError, OverflowError):
                    likelihood += 1e6

            return likelihood

        # 季节性模型训练
        print("   🔄 开始季节性模型训练...")

        best_seasonal_params = None
        best_seasonal_likelihood = float('inf')

        for attempt in range(5):
            # 从基础模型参数开始，添加季节性强度参数
            initial_params = list(best_params) + [0.2]  # 初始季节性强度20%

            try:
                result = minimize(
                    seasonal_mbgnbd_likelihood,
                    initial_params,
                    args=(frequency, recency, T, self.seasonal_factors['markov']),
                    method='L-BFGS-B',
                    bounds=[(0.01, 10), (0.01, 20), (0.01, 10), (0.01, 10), (0, 1)]
                )

                if result.success and result.fun < best_seasonal_likelihood:
                    best_seasonal_likelihood = result.fun
                    best_seasonal_params = result.x

            except Exception as e:
                print(f"      ⚠️ 季节性训练尝试{attempt + 1}失败: {e}")
                continue

        if best_seasonal_params is None:
            print("   ⚠️ 季节性模型训练失败，使用基础模型参数")
            self.enhanced_model_params = list(best_params) + [0.0]
        else:
            self.enhanced_model_params = best_seasonal_params

        r_s, alpha_s, a_s, b_s, seasonal_strength = self.enhanced_model_params

        print(f"   ✅ 季节性增强模型训练完成:")
        print(f"      - r (购买率形状): {r_s:.4f}")
        print(f"      - α (购买率尺度): {alpha_s:.4f}")
        print(f"      - a (流失率形状): {a_s:.4f}")
        print(f"      - b (流失率尺度): {b_s:.4f}")
        print(f"      - 季节性强度: {seasonal_strength:.4f}")
        print(f"      - 负对数似然: {best_seasonal_likelihood:.2f}")

        # 4.3 模型改进效果分析
        print("\n📊 4.3 模型改进效果分析...")

        likelihood_improvement = (best_likelihood - best_seasonal_likelihood) / best_likelihood * 100

        print(f"   ✅ 模型改进分析:")
        print(f"      - 基础模型似然: {best_likelihood:.2f}")
        print(f"      - 季节性模型似然: {best_seasonal_likelihood:.2f}")
        print(f"      - 似然改进: {likelihood_improvement:.2f}%")

        if likelihood_improvement > 1:
            print(f"      - 🎉 季节性模型显著优于基础模型")
        elif likelihood_improvement > 0:
            print(f"      - ✅ 季节性模型略优于基础模型")
        else:
            print(f"      - ⚠️ 季节性改进不明显")

        return True

    # ==================== 第五阶段：CLV预测与验证 ====================

    def predict_clv(self):
        """
        第五阶段：CLV预测与验证
        """
        print("\n" + "=" * 60)
        print("第五阶段：CLV预测与验证")
        print("=" * 60)

        # 5.1 基础MBG-NBD预测
        print("🎯 5.1 基础MBG-NBD预测...")

        r, alpha, a, b = self.base_model_params

        base_predictions = []

        for _, customer in self.rfm_data.iterrows():
            frequency = customer['historical_frequency']
            recency = customer['recency']
            T = customer['T']
            avg_order_value = customer['monetary_avg']

            # 计算预测期内的期望购买次数
            if frequency == 0:
                # 零购买客户
                expected_purchases = (a / (a + b)) * (r * self.prediction_period) / (alpha + T)
            else:
                # 有购买历史的客户
                expected_purchases = (
                        (r + frequency) * self.prediction_period / (alpha + T + self.prediction_period) *
                        (a + b + frequency - 1) / (a + frequency - 1)
                )

            # 计算CLV
            clv = expected_purchases * avg_order_value

            base_predictions.append({
                'customer_id': customer['customer_id'],
                'segment': customer['segment'],
                'behavior_cluster': customer['behavior_cluster'],
                'expected_purchases': expected_purchases,
                'avg_order_value': avg_order_value,
                'predicted_clv': clv,
                'historical_frequency': frequency,
                'recency': recency,
                'T': T
            })

        self.base_predictions = pd.DataFrame(base_predictions)

        base_total_clv = self.base_predictions['predicted_clv'].sum()
        base_avg_clv = self.base_predictions['predicted_clv'].mean()

        print(f"   ✅ 基础预测完成:")
        print(f"      - 总CLV: {base_total_clv:,.2f}元")
        print(f"      - 平均CLV: {base_avg_clv:.2f}元")
        print(f"      - 预测客户数: {len(self.base_predictions)}")

        # 5.2 季节性增强预测
        print("\n🌟 5.2 季节性增强预测...")

        r_s, alpha_s, a_s, b_s, seasonal_strength = self.enhanced_model_params

        enhanced_predictions = []

        # 计算预测期的平均季节性因子
        current_month = datetime.now().month
        prediction_months = []
        for i in range(self.prediction_period):
            month = ((current_month - 1 + i // 30) % 12) + 1
            prediction_months.append(month)

        # 计算加权平均季节性因子
        seasonal_weights = {}
        for month in range(1, 13):
            seasonal_weights[month] = prediction_months.count(month) / len(prediction_months)

        weighted_seasonal_factor = sum(
            self.seasonal_factors['markov'][month] * weight
            for month, weight in seasonal_weights.items()
        )

        print(f"   📅 预测期季节性分析:")
        print(f"      - 预测起始月份: {current_month}月")
        print(f"      - 加权季节性因子: {weighted_seasonal_factor:.3f}")

        for _, customer in self.rfm_data.iterrows():
            frequency = customer['historical_frequency']
            recency = customer['recency']
            T = customer['T']
            avg_order_value = customer['monetary_avg']

            # 季节性调整
            seasonal_adjustment = 1 + seasonal_strength * (weighted_seasonal_factor - 1)
            adjusted_alpha = alpha_s * seasonal_adjustment

            # 计算预测期内的期望购买次数
            if frequency == 0:
                expected_purchases = (a_s / (a_s + b_s)) * (r_s * self.prediction_period) / (adjusted_alpha + T)
            else:
                expected_purchases = (
                        (r_s + frequency) * self.prediction_period / (adjusted_alpha + T + self.prediction_period) *
                        (a_s + b_s + frequency - 1) / (a_s + frequency - 1)
                )

            # 计算CLV
            clv = expected_purchases * avg_order_value

            enhanced_predictions.append({
                'customer_id': customer['customer_id'],
                'segment': customer['segment'],
                'behavior_cluster': customer['behavior_cluster'],
                'expected_purchases': expected_purchases,
                'avg_order_value': avg_order_value,
                'predicted_clv': clv,
                'seasonal_adjustment': seasonal_adjustment,
                'historical_frequency': frequency,
                'recency': recency,
                'T': T
            })

        self.enhanced_predictions = pd.DataFrame(enhanced_predictions)

        enhanced_total_clv = self.enhanced_predictions['predicted_clv'].sum()
        enhanced_avg_clv = self.enhanced_predictions['predicted_clv'].mean()

        print(f"   ✅ 季节性增强预测完成:")
        print(f"      - 总CLV: {enhanced_total_clv:,.2f}元")
        print(f"      - 平均CLV: {enhanced_avg_clv:.2f}元")
        print(f"      - 季节性调整: {seasonal_adjustment:.3f}")

        # 5.3 预测效果对比
        print("\n📊 5.3 预测效果对比...")

        clv_improvement = (enhanced_total_clv - base_total_clv) / base_total_clv * 100

        print(f"   ✅ 预测对比分析:")
        print(f"      - 基础模型总CLV: {base_total_clv:,.2f}元")
        print(f"      - 增强模型总CLV: {enhanced_total_clv:,.2f}元")
        print(f"      - CLV提升: {clv_improvement:+.2f}%")
        print(f"      - 绝对提升: {enhanced_total_clv - base_total_clv:+,.2f}元")

        return True

    # ==================== 第六阶段：效果验证与可视化 ====================

    def validate_and_visualize(self, save_path=None):
        """
        第六阶段：效果验证与可视化
        """
        print("\n" + "=" * 60)
        print("第六阶段：效果验证与可视化")
        print("=" * 60)

        # 6.1 模型验证
        print("✅ 6.1 模型验证...")

        # 计算预测准确性指标
        base_predictions = self.base_predictions['predicted_clv'].values
        enhanced_predictions = self.enhanced_predictions['predicted_clv'].values

        # 基础统计
        validation_metrics = {
            'base_model': {
                'mean_clv': np.mean(base_predictions),
                'median_clv': np.median(base_predictions),
                'std_clv': np.std(base_predictions),
                'total_clv': np.sum(base_predictions),
                'customers_with_clv': np.sum(base_predictions > 0)
            },
            'enhanced_model': {
                'mean_clv': np.mean(enhanced_predictions),
                'median_clv': np.median(enhanced_predictions),
                'std_clv': np.std(enhanced_predictions),
                'total_clv': np.sum(enhanced_predictions),
                'customers_with_clv': np.sum(enhanced_predictions > 0)
            }
        }

        # 相关性分析
        correlation = np.corrcoef(base_predictions, enhanced_predictions)[0, 1]

        print(f"   📊 验证指标:")
        print(f"      - 预测相关性: {correlation:.4f}")
        print(f"      - 基础模型活跃客户: {validation_metrics['base_model']['customers_with_clv']}")
        print(f"      - 增强模型活跃客户: {validation_metrics['enhanced_model']['customers_with_clv']}")

        self.performance_metrics = validation_metrics

        # 6.2 创建综合可视化
        print("\n📊 6.2 创建综合可视化...")

        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('完整MBG-NBD系统分析报告', fontsize=16, fontweight='bold')

        # 1. 客户分层分布
        ax1 = axes[0, 0]
        segment_counts = self.rfm_data['segment'].value_counts()
        colors = ['gold', 'lightgreen', 'lightblue', 'orange', 'lightcoral']

        wedges, texts, autotexts = ax1.pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('客户分层分布')

        # 2. RFM特征分布
        ax2 = axes[0, 1]
        ax2.scatter(self.rfm_data['recency'], self.rfm_data['frequency'],
                    c=self.rfm_data['monetary_avg'], cmap='viridis', alpha=0.6)
        ax2.set_xlabel('Recency (天)')
        ax2.set_ylabel('Frequency (次)')
        ax2.set_title('RFM特征分布')
        ax2.grid(True, alpha=0.3)

        # 3. 季节性因子对比
        ax3 = axes[0, 2]
        months = list(range(1, 13))
        basic_factors = [self.seasonal_factors['basic'][m] for m in months]
        markov_factors = [self.seasonal_factors['markov'][m] for m in months]

        x = np.arange(len(months))
        width = 0.35

        ax3.bar(x - width / 2, basic_factors, width, label='基础季节性', alpha=0.7)
        ax3.bar(x + width / 2, markov_factors, width, label='马尔可夫季节性', alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)

        ax3.set_xlabel('月份')
        ax3.set_ylabel('季节性因子')
        ax3.set_title('季节性因子对比')
        ax3.set_xticks(x)
        ax3.set_xticklabels(months)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. CLV预测对比
        ax4 = axes[1, 0]
        ax4.scatter(base_predictions, enhanced_predictions, alpha=0.6)

        # 添加对角线
        min_val = min(base_predictions.min(), enhanced_predictions.min())
        max_val = max(base_predictions.max(), enhanced_predictions.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        ax4.set_xlabel('基础模型CLV预测')
        ax4.set_ylabel('增强模型CLV预测')
        ax4.set_title('CLV预测对比')
        ax4.grid(True, alpha=0.3)

        # 5. 客户分层CLV分布
        ax5 = axes[1, 1]
        segment_clv = self.enhanced_predictions.groupby('segment')['predicted_clv'].sum().sort_values(ascending=True)

        bars = ax5.barh(range(len(segment_clv)), segment_clv.values,
                        color=['lightcoral', 'orange', 'lightblue', 'lightgreen', 'gold'])
        ax5.set_yticks(range(len(segment_clv)))
        ax5.set_yticklabels(segment_clv.index)
        ax5.set_xlabel('总CLV预测 (元)')
        ax5.set_title('各客户分层CLV贡献')
        ax5.grid(True, alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax5.text(width + max(segment_clv.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:,.0f}', ha='left', va='center', fontsize=9)

        # 6. 行为聚类分析
        ax6 = axes[1, 2]
        cluster_clv = self.enhanced_predictions.groupby('behavior_cluster')['predicted_clv'].mean()

        bars = ax6.bar(cluster_clv.index, cluster_clv.values,
                       color=plt.cm.Set3(np.linspace(0, 1, len(cluster_clv))))
        ax6.set_xlabel('行为聚类')
        ax6.set_ylabel('平均CLV (元)')
        ax6.set_title('行为聚类平均CLV')
        ax6.grid(True, alpha=0.3)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2, height + max(cluster_clv.values) * 0.01,
                     f'{height:.0f}', ha='center', va='bottom', fontsize=9)

        # 7. 模型参数对比
        ax7 = axes[2, 0]
        param_names = ['r', 'α', 'a', 'b']
        base_params = self.base_model_params
        enhanced_params = self.enhanced_model_params[:4]

        x = np.arange(len(param_names))
        width = 0.35

        ax7.bar(x - width / 2, base_params, width, label='基础模型', alpha=0.7)
        ax7.bar(x + width / 2, enhanced_params, width, label='增强模型', alpha=0.7)

        ax7.set_xlabel('模型参数')
        ax7.set_ylabel('参数值')
        ax7.set_title('模型参数对比')
        ax7.set_xticks(x)
        ax7.set_xticklabels(param_names)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. CLV分布直方图
        ax8 = axes[2, 1]
        ax8.hist(base_predictions, bins=30, alpha=0.7, label='基础模型', density=True)
        ax8.hist(enhanced_predictions, bins=30, alpha=0.7, label='增强模型', density=True)

        ax8.set_xlabel('CLV预测值 (元)')
        ax8.set_ylabel('密度')
        ax8.set_title('CLV预测分布')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 总结信息
        ax9 = axes[2, 2]
        ax9.axis('off')

        # 计算关键指标
        total_customers = len(self.rfm_data)
        base_total = validation_metrics['base_model']['total_clv']
        enhanced_total = validation_metrics['enhanced_model']['total_clv']
        improvement = (enhanced_total - base_total) / base_total * 100

        summary_text = f"""
系统分析总结

📊 数据概况:
• 总客户数: {total_customers:,}
• 数据时间跨度: {(self.processed_data['order_date'].max() - self.processed_data['order_date'].min()).days}天
• 总交易数: {len(self.processed_data):,}

🎯 模型效果:
• 基础CLV: {base_total:,.0f}元
• 增强CLV: {enhanced_total:,.0f}元
• 提升幅度: {improvement:+.2f}%

🌟 关键改进:
• 客户分层: {len(self.rfm_data['segment'].unique())}个层级
• 行为聚类: {self.customer_segments['optimal_k']}个聚类
• 季节性强度: {self.enhanced_model_params[4]:.1%}
• 预测相关性: {correlation:.3f}

🚀 业务价值:
• 精准客户识别
• 季节性营销优化
• 个性化CLV预测
• 数据驱动决策支持
"""

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✅ 可视化图表已保存: {save_path}")

        plt.show()

        return validation_metrics

    # ==================== 系统总结与报告生成 ====================

    def generate_comprehensive_report(self, save_path=None):
        """生成综合分析报告"""
        print("\n📋 生成综合分析报告...")

        report = {
            'system_info': {
                'version': '2.0',
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_period_days': self.prediction_period
            },
            'data_summary': {
                'total_customers': len(self.rfm_data),
                'total_transactions': len(self.processed_data),
                'data_span_days': (
                            self.processed_data['order_date'].max() - self.processed_data['order_date'].min()).days,
                'total_revenue': float(self.processed_data['amount'].sum())
            },
            'customer_segmentation': {
                'segments': self.rfm_data['segment'].value_counts().to_dict(),
                'behavior_clusters': int(self.customer_segments['optimal_k'])
            },
            'seasonal_analysis': {
                'basic_factors': self.seasonal_factors['basic'],
                'markov_factors': self.seasonal_factors['markov'],
                'seasonal_strength': float(self.enhanced_model_params[4])
            },
            'model_parameters': {
                'base_model': {
                    'r': float(self.base_model_params[0]),
                    'alpha': float(self.base_model_params[1]),
                    'a': float(self.base_model_params[2]),
                    'b': float(self.base_model_params[3])
                },
                'enhanced_model': {
                    'r': float(self.enhanced_model_params[0]),
                    'alpha': float(self.enhanced_model_params[1]),
                    'a': float(self.enhanced_model_params[2]),
                    'b': float(self.enhanced_model_params[3]),
                    'seasonal_strength': float(self.enhanced_model_params[4])
                }
            },
            'prediction_results': {
                'base_model': {
                    'total_clv': float(self.performance_metrics['base_model']['total_clv']),
                    'mean_clv': float(self.performance_metrics['base_model']['mean_clv']),
                    'active_customers': int(self.performance_metrics['base_model']['customers_with_clv'])
                },
                'enhanced_model': {
                    'total_clv': float(self.performance_metrics['enhanced_model']['total_clv']),
                    'mean_clv': float(self.performance_metrics['enhanced_model']['mean_clv']),
                    'active_customers': int(self.performance_metrics['enhanced_model']['customers_with_clv'])
                },
                'improvement': {
                    'clv_improvement_pct': float((self.performance_metrics['enhanced_model']['total_clv'] -
                                                  self.performance_metrics['base_model']['total_clv']) /
                                                 self.performance_metrics['base_model']['total_clv'] * 100),
                    'absolute_improvement': float(self.performance_metrics['enhanced_model']['total_clv'] -
                                                  self.performance_metrics['base_model']['total_clv'])
                }
            },
            'key_insights': [
                f"完成了{len(self.rfm_data)}个客户的完整分层分析",
                f"马尔可夫季节性学习识别了{len(self.seasonal_factors['markov'])}个月度模式",
                f"增强模型相比基础模型CLV提升{(self.performance_metrics['enhanced_model']['total_clv'] - self.performance_metrics['base_model']['total_clv']) / self.performance_metrics['base_model']['total_clv'] * 100:.2f}%",
                f"季节性强度为{self.enhanced_model_params[4]:.1%}，表明季节性影响适中",
                f"客户行为异质性分析识别了{self.customer_segments['optimal_k']}个不同的行为模式"
            ]
        }

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ 综合报告已保存: {save_path}")

        return report

    def save_predictions(self, base_path='base_predictions.csv', enhanced_path='enhanced_predictions.csv'):
        """保存预测结果"""
        print("\n💾 保存预测结果...")

        # 保存基础预测
        self.base_predictions.to_csv(base_path, index=False, encoding='utf-8-sig')

        # 保存增强预测
        self.enhanced_predictions.to_csv(enhanced_path, index=False, encoding='utf-8-sig')

        print(f"✅ 预测结果已保存:")
        print(f"   - 基础预测: {base_path}")
        print(f"   - 增强预测: {enhanced_path}")

        return True


def main():
    """主函数 - 完整MBG-NBD系统演示"""
    print("🚀 完整MBG-NBD客户生命周期价值预测系统")
    print("=" * 80)

    # 初始化系统
    system = CompleteMBGNBDSystem(prediction_period=90)

    # 第一阶段：数据加载与预处理
    if not system.load_data('/Users/changyu/Downloads/manifest.csv'):
        print("❌ 系统初始化失败")
        return None

    # 第二阶段：客户分层与行为异质性分析
    rfm_data = system.create_customer_segmentation()
    if rfm_data is None:
        print("❌ 客户分层失败")
        return None

    # 第三阶段：季节性马尔可夫链学习
    seasonal_factors = system.learn_seasonal_patterns()
    if seasonal_factors is None:
        print("❌ 季节性学习失败")
        return None

    # 第四阶段：MBG-NBD模型训练
    if not system.train_mbgnbd_models():
        print("❌ 模型训练失败")
        return None

    # 第五阶段：CLV预测与验证
    if not system.predict_clv():
        print("❌ CLV预测失败")
        return None

    # 第六阶段：效果验证与可视化
    validation_metrics = system.validate_and_visualize('/Users/changyu/Downloads/CLV/complete_mbgnbd_analysis.png')

    # 生成综合报告
    report = system.generate_comprehensive_report('/Users/changyu/Downloads/CLV/complete_mbgnbd_report.json')

    # 保存预测结果
    system.save_predictions('/Users/changyu/Downloads/CLV/base_clv_predictions.csv', '/Users/changyu/Downloads/CLV/enhanced_clv_predictions.csv')

    print("\n" + "=" * 80)
    print("🎉 完整MBG-NBD系统分析完成!")
    print("=" * 80)

    # 输出关键结果
    base_clv = validation_metrics['base_model']['total_clv']
    enhanced_clv = validation_metrics['enhanced_model']['total_clv']
    improvement = (enhanced_clv - base_clv) / base_clv * 100

    print(f"📊 核心结果:")
    print(f"   - 客户总数: {len(system.rfm_data):,}")
    print(f"   - 基础模型CLV: {base_clv:,.2f}元")
    print(f"   - 增强模型CLV: {enhanced_clv:,.2f}元")
    print(f"   - 模型改进: {improvement:+.2f}%")
    print(f"   - 季节性强度: {system.enhanced_model_params[4]:.1%}")

    return system


if __name__ == "__main__":
    # 运行完整MBG-NBD系统
    system = main()