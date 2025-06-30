#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包含季节性调整的完整MBG-NBD模型实现
集成马尔可夫季节性学习和预设季节性因子两种方法

作者: Manus AI
版本: 3.0 (集成季节性)
日期: 2025-01-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
from scipy.special import gamma, hyp2f1, loggamma
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import json
import pickle
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeasonalMBGNBDModel:
    """
    包含季节性调整的完整MBG-NBD模型实现

    功能包括：
    1. 数据预处理和质量检查
    2. RFM特征工程
    3. 季节性因子学习（马尔可夫方法 + 预设方法）
    4. MBG-NBD模型训练
    5. 季节性调整的CLV预测
    6. 模型验证和评估
    7. 结果可视化
    8. 模型保存和加载
    """

    def __init__(self, observation_period_end=None, prediction_period=90, seasonal_method='markov'):
        """
        初始化季节性MBG-NBD模型

        参数:
        observation_period_end: 观察期结束时间
        prediction_period: 预测期长度（天）
        seasonal_method: 季节性方法 ('markov', 'preset', 'none')
        """
        self.observation_period_end = observation_period_end
        self.prediction_period = prediction_period
        self.seasonal_method = seasonal_method

        # 模型参数
        self.params = None
        self.model_fitted = False

        # 季节性相关
        self.seasonal_factors = None
        self.transition_matrix = None
        self.emission_probabilities = None
        self.seasonal_learned = False

        # 数据存储
        self.raw_data = None
        self.processed_data = None
        self.rfm_data = None
        self.train_data = None
        self.test_data = None

        # 预测结果
        self.predictions = None
        self.clv_predictions = None
        self.seasonal_clv_predictions = None

        # 模型评估结果
        self.validation_results = None
        self.seasonal_validation_results = None

        logger.info(f"季节性MBG-NBD模型初始化完成，季节性方法: {seasonal_method}")

    def load_data(self, data_path, customer_col='customer_id', amount_col='amount', date_col='order_date'):
        """
        加载和预处理数据

        参数:
        data_path: 数据文件路径
        customer_col: 客户ID列名
        amount_col: 金额列名
        date_col: 日期列名
        """
        logger.info(f"开始加载数据: {data_path}")

        try:
            # 加载数据
            if data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(data_path)
            else:
                raise ValueError("不支持的文件格式，请使用CSV或Excel文件")

            # 标准化列名
            self.raw_data.columns = [customer_col, amount_col, date_col]
            self.raw_data.columns = ['customer_id', 'amount', 'order_date']

            # 数据类型转换
            self.raw_data['order_date'] = pd.to_datetime(self.raw_data['order_date'])
            self.raw_data['amount'] = pd.to_numeric(self.raw_data['amount'], errors='coerce')
            self.raw_data['month'] = self.raw_data['order_date'].dt.month
            self.raw_data['year'] = self.raw_data['order_date'].dt.year

            logger.info(f"数据加载完成: {len(self.raw_data)}条记录, {self.raw_data['customer_id'].nunique()}个客户")

            # 数据质量检查
            self._data_quality_check()

            # 数据预处理
            self._preprocess_data()

            return self.processed_data

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise

    def _data_quality_check(self):
        """数据质量检查"""
        logger.info("开始数据质量检查...")

        # 检查缺失值
        missing_data = self.raw_data.isnull().sum()
        if missing_data.sum() > 0:
            logger.warning(f"发现缺失值: {missing_data.to_dict()}")

        # 检查负值
        negative_amounts = (self.raw_data['amount'] < 0).sum()
        if negative_amounts > 0:
            logger.warning(f"发现{negative_amounts}条负金额记录")

        # 检查零值
        zero_amounts = (self.raw_data['amount'] == 0).sum()
        if zero_amounts > 0:
            logger.warning(f"发现{zero_amounts}条零金额记录")

        # 检查日期范围
        date_range = self.raw_data['order_date'].max() - self.raw_data['order_date'].min()
        logger.info(f"数据时间跨度: {date_range.days}天")

        # 检查客户交易频次
        customer_freq = self.raw_data.groupby('customer_id').size()
        logger.info(f"客户交易频次统计: 平均{customer_freq.mean():.1f}次, 中位数{customer_freq.median():.1f}次")

    def _preprocess_data(self):
        """数据预处理"""
        logger.info("开始数据预处理...")

        # 复制原始数据
        self.processed_data = self.raw_data.copy()

        # 移除缺失值
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.dropna()
        logger.info(f"移除缺失值: {initial_count - len(self.processed_data)}条")

        # 移除非正金额
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data[self.processed_data['amount'] > 0]
        logger.info(f"移除非正金额: {initial_count - len(self.processed_data)}条")

        # 设置观察期结束时间
        if self.observation_period_end is None:
            # 使用数据中80%的时间作为观察期
            date_range = self.processed_data['order_date'].max() - self.processed_data['order_date'].min()
            self.observation_period_end = self.processed_data['order_date'].min() + date_range * 0.8

        logger.info(f"观察期结束时间: {self.observation_period_end}")

        # 按时间排序
        self.processed_data = self.processed_data.sort_values(['customer_id', 'order_date'])

        logger.info(f"数据预处理完成: {len(self.processed_data)}条记录")

    def learn_seasonal_factors(self):
        """学习季节性因子"""
        if self.seasonal_method == 'none':
            logger.info("跳过季节性因子学习")
            self.seasonal_factors = {i: 1.0 for i in range(1, 13)}
            return self.seasonal_factors

        logger.info(f"开始学习季节性因子，方法: {self.seasonal_method}")

        if self.seasonal_method == 'preset':
            self.seasonal_factors = self._learn_preset_seasonal_factors()
        elif self.seasonal_method == 'markov':
            self.seasonal_factors = self._learn_markov_seasonal_factors()
        else:
            raise ValueError(f"不支持的季节性方法: {self.seasonal_method}")

        self.seasonal_learned = True

        logger.info("季节性因子学习完成:")
        for month, factor in self.seasonal_factors.items():
            logger.info(f"  {month}月: {factor:.3f}")

        return self.seasonal_factors

    def _learn_preset_seasonal_factors(self):
        """学习预设季节性因子（简单月度平均法）"""
        logger.info("使用预设方法学习季节性因子...")

        # 计算月度收入
        monthly_revenue = self.processed_data.groupby('month')['amount'].sum()
        monthly_avg = monthly_revenue.mean()

        # 计算季节性因子
        seasonal_factors = {}
        for month in range(1, 13):
            if month in monthly_revenue.index:
                seasonal_factors[month] = monthly_revenue[month] / monthly_avg
            else:
                seasonal_factors[month] = 1.0

        return seasonal_factors

    def _learn_markov_seasonal_factors(self):
        """学习马尔可夫季节性因子"""
        logger.info("使用马尔可夫方法学习季节性因子...")

        # 1. 学习状态转移矩阵
        self.transition_matrix = self._learn_transition_matrix()

        # 2. 学习发射概率
        self.emission_probabilities = self._learn_emission_probabilities()

        # 3. 计算综合季节性因子
        seasonal_factors = self._compute_markov_seasonal_factors()

        return seasonal_factors

    def _learn_transition_matrix(self):
        """学习月份间的状态转移矩阵"""
        logger.info("  学习状态转移矩阵...")

        # 初始化转移矩阵
        transition_matrix = np.zeros((12, 12))

        # 统计月份转移
        monthly_data = self.processed_data.groupby(['customer_id', 'year', 'month']).size().reset_index()
        monthly_data.columns = ['customer_id', 'year', 'month', 'transactions']
        monthly_data = monthly_data.sort_values(['customer_id', 'year', 'month'])

        # 统计转移次数
        for customer_id in monthly_data['customer_id'].unique():
            customer_data = monthly_data[monthly_data['customer_id'] == customer_id]

            for i in range(len(customer_data) - 1):
                current_month = customer_data.iloc[i]['month'] - 1  # 转为0-11索引
                next_month = customer_data.iloc[i + 1]['month'] - 1

                # 处理跨年情况
                current_year = customer_data.iloc[i]['year']
                next_year = customer_data.iloc[i + 1]['year']

                if next_year == current_year + 1 and customer_data.iloc[i]['month'] == 12 and customer_data.iloc[i + 1][
                    'month'] == 1:
                    transition_matrix[current_month, next_month] += 1
                elif next_year == current_year and customer_data.iloc[i + 1]['month'] == customer_data.iloc[i][
                    'month'] + 1:
                    transition_matrix[current_month, next_month] += 1

        # 归一化转移矩阵
        for i in range(12):
            row_sum = transition_matrix[i, :].sum()
            if row_sum > 0:
                transition_matrix[i, :] /= row_sum
            else:
                # 如果某个月没有转移数据，使用均匀分布
                transition_matrix[i, :] = 1 / 12

        return transition_matrix

    def _learn_emission_probabilities(self):
        """学习每个月份状态的发射概率（消费强度分布）"""
        logger.info("  学习发射概率分布...")

        emission_probs = {}

        for month in range(1, 13):
            month_data = self.processed_data[self.processed_data['month'] == month]['amount']

            if len(month_data) > 0:
                emission_probs[month] = {
                    'mean': month_data.mean(),
                    'std': month_data.std(),
                    'total_volume': month_data.sum(),
                    'transaction_count': len(month_data)
                }
            else:
                # 如果某月没有数据，使用全局平均
                global_mean = self.processed_data['amount'].mean()
                emission_probs[month] = {
                    'mean': global_mean,
                    'std': self.processed_data['amount'].std(),
                    'total_volume': 0,
                    'transaction_count': 0
                }

        return emission_probs

    def _compute_markov_seasonal_factors(self):
        """计算马尔可夫综合季节性因子"""
        logger.info("  计算马尔可夫综合季节性因子...")

        # 计算稳态分布
        steady_state = self._compute_steady_state()

        # 基于发射概率的季节性因子
        global_mean = np.mean([self.emission_probabilities[m]['mean'] for m in range(1, 13)])
        emission_factors = {}
        for month in range(1, 13):
            emission_factors[month] = self.emission_probabilities[month]['mean'] / global_mean

        # 马尔可夫综合因子 = 0.7 * 发射概率 + 0.3 * 稳态权重
        seasonal_factors = {}
        for month in range(1, 13):
            emission_weight = emission_factors[month]
            steady_weight = steady_state[month - 1] * 12  # 归一化到平均值1

            seasonal_factors[month] = 0.7 * emission_weight + 0.3 * steady_weight

        return seasonal_factors

    def _compute_steady_state(self):
        """计算马尔可夫链的稳态分布"""
        try:
            # 计算转移矩阵的特征向量
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

            # 找到特征值为1的特征向量（稳态分布）
            steady_state_index = np.argmin(np.abs(eigenvalues - 1))
            steady_state = np.real(eigenvectors[:, steady_state_index])

            # 归一化
            steady_state = steady_state / np.sum(steady_state)

            return steady_state
        except:
            # 如果计算失败，返回均匀分布
            return np.ones(12) / 12

    def create_rfm_features(self):
        """创建RFM特征"""
        logger.info("开始创建RFM特征...")

        # 分离观察期数据
        observation_data = self.processed_data[
            self.processed_data['order_date'] <= self.observation_period_end
            ]

        # 计算RFM特征
        rfm_features = []

        for customer_id in observation_data['customer_id'].unique():
            customer_data = observation_data[observation_data['customer_id'] == customer_id]

            if len(customer_data) == 0:
                continue

            # 计算特征
            first_purchase = customer_data['order_date'].min()
            last_purchase = customer_data['order_date'].max()

            # Frequency: 购买次数 - 1 (MBG-NBD约定)
            frequency = len(customer_data) - 1

            # Recency: 最后一次购买到观察期结束的时间（天）
            recency = (last_purchase - first_purchase).days

            # T: 客户生命周期长度（天）
            T = (self.observation_period_end - first_purchase).days

            # Monetary: 平均订单价值
            monetary = customer_data['amount'].mean()

            # 总消费金额
            total_amount = customer_data['amount'].sum()

            rfm_features.append({
                'customer_id': customer_id,
                'frequency': frequency,
                'recency': recency,
                'T': T,
                'monetary': monetary,
                'total_amount': total_amount,
                'first_purchase': first_purchase,
                'last_purchase': last_purchase,
                'transaction_count': len(customer_data)
            })

        self.rfm_data = pd.DataFrame(rfm_features)

        # 过滤无效数据
        initial_count = len(self.rfm_data)
        self.rfm_data = self.rfm_data[
            (self.rfm_data['T'] > 0) &
            (self.rfm_data['recency'] >= 0) &
            (self.rfm_data['recency'] <= self.rfm_data['T'])
            ]

        logger.info(f"RFM特征创建完成: {len(self.rfm_data)}个客户, 过滤{initial_count - len(self.rfm_data)}个无效客户")

        return self.rfm_data

    def split_data(self, test_size=0.2, random_state=42):
        """分割训练和测试数据"""
        logger.info(f"分割数据: 测试集比例{test_size}")

        if self.rfm_data is None:
            raise ValueError("请先创建RFM特征")

        # 创建分层标签，确保每个分层至少有2个样本
        frequency_bins = pd.cut(self.rfm_data['frequency'], bins=min(5, self.rfm_data['frequency'].nunique()),
                                labels=False)

        # 检查每个分层的样本数
        bin_counts = pd.Series(frequency_bins).value_counts()
        if (bin_counts < 2).any():
            # 如果有分层样本数少于2，则不使用分层
            logger.warning("某些分层样本数少于2，取消分层采样")
            self.train_data, self.test_data = train_test_split(
                self.rfm_data,
                test_size=test_size,
                random_state=random_state
            )
        else:
            self.train_data, self.test_data = train_test_split(
                self.rfm_data,
                test_size=test_size,
                random_state=random_state,
                stratify=frequency_bins
            )

        logger.info(f"数据分割完成: 训练集{len(self.train_data)}个客户, 测试集{len(self.test_data)}个客户")

        return self.train_data, self.test_data

    def _mbgnbd_likelihood(self, params, frequency, recency, T):
        """
        MBG-NBD模型的对数似然函数

        参数:
        params: [r, alpha, a, b] - 模型参数
        frequency: 购买频次
        recency: 最近购买时间
        T: 观察期长度
        """
        r, alpha, a, b = params

        # 参数约束
        if r <= 0 or alpha <= 0 or a <= 0 or b <= 0:
            return -np.inf

        # 计算似然函数
        try:
            # 第一部分：购买过程
            part1 = (
                    loggamma(r + frequency) - loggamma(r) +
                    r * np.log(alpha) - (r + frequency) * np.log(alpha + T)
            )

            # 第二部分：流失过程
            if frequency > 0:
                part2 = np.log(
                    a / b * (
                            hyp2f1(r + frequency, b + 1, a + b, recency / (alpha + T)) -
                            (recency / (alpha + T)) ** (a + b) *
                            hyp2f1(r + frequency, b + 1, a + b + 1, recency / (alpha + T))
                    )
                )
            else:
                part2 = np.log(a / (a + b))

            likelihood = part1 + part2

            # 处理数值问题
            if np.isnan(likelihood) or np.isinf(likelihood):
                return -1e10

            return likelihood

        except Exception:
            return -1e10

    def _negative_log_likelihood(self, params, data):
        """负对数似然函数（用于优化）"""
        total_ll = 0

        for _, row in data.iterrows():
            ll = self._mbgnbd_likelihood(
                params,
                row['frequency'],
                row['recency'],
                row['T']
            )
            total_ll += ll

        return -total_ll

    def fit(self, max_iter=1000, method='L-BFGS-B'):
        """
        训练MBG-NBD模型

        参数:
        max_iter: 最大迭代次数
        method: 优化方法
        """
        logger.info("开始训练MBG-NBD模型...")

        if self.train_data is None:
            raise ValueError("请先分割数据")

        # 初始参数
        initial_params = [1.0, 1.0, 1.0, 1.0]  # [r, alpha, a, b]

        # 参数边界
        bounds = [(0.01, 10), (0.01, 10), (0.01, 10), (0.01, 10)]

        # 多次随机初始化，选择最佳结果
        best_result = None
        best_likelihood = np.inf

        for i in range(5):  # 尝试5次不同的初始化
            # 随机初始化
            init_params = np.random.uniform(0.1, 2.0, 4)

            try:
                # 优化
                result = optimize.minimize(
                    self._negative_log_likelihood,
                    init_params,
                    args=(self.train_data,),
                    method=method,
                    bounds=bounds,
                    options={'maxiter': max_iter}
                )

                if result.success and result.fun < best_likelihood:
                    best_result = result
                    best_likelihood = result.fun

            except Exception as e:
                logger.warning(f"优化尝试{i + 1}失败: {e}")
                continue

        if best_result is None or not best_result.success:
            raise RuntimeError("模型训练失败，请检查数据质量")

        self.params = best_result.x
        self.model_fitted = True

        logger.info(f"模型训练完成")
        logger.info(
            f"参数: r={self.params[0]:.4f}, alpha={self.params[1]:.4f}, a={self.params[2]:.4f}, b={self.params[3]:.4f}")
        logger.info(f"负对数似然: {best_likelihood:.2f}")

        return self.params

    def predict_purchases(self, t, frequency=None, recency=None, T=None, data=None):
        """
        预测未来购买次数

        参数:
        t: 预测期长度
        frequency, recency, T: 单个客户的RFM特征
        data: 批量预测的数据
        """
        if not self.model_fitted:
            raise ValueError("请先训练模型")

        r, alpha, a, b = self.params

        if data is not None:
            # 批量预测
            predictions = []
            for _, row in data.iterrows():
                pred = self._predict_single_customer(
                    t, row['frequency'], row['recency'], row['T'], r, alpha, a, b
                )
                predictions.append(pred)
            return np.array(predictions)
        else:
            # 单个客户预测
            return self._predict_single_customer(t, frequency, recency, T, r, alpha, a, b)

    def _predict_single_customer(self, t, frequency, recency, T, r, alpha, a, b):
        """单个客户的购买预测"""
        try:
            # 计算条件期望购买次数
            if frequency == 0:
                # 没有重复购买的客户
                prediction = (a / (a + b)) * (r * t) / (alpha + T)
            else:
                # 有重复购买的客户
                prediction = (
                        (r + frequency) * t / (alpha + T + t) *
                        (a + b + frequency - 1) / (a + frequency - 1)
                )

            return max(0, prediction)  # 确保非负

        except Exception:
            return 0.0

    def predict_clv(self, prediction_period=None, discount_rate=0.01, apply_seasonality=True):
        """
        预测客户生命周期价值（包含季节性调整）

        参数:
        prediction_period: 预测期长度（天）
        discount_rate: 折现率
        apply_seasonality: 是否应用季节性调整
        """
        if not self.model_fitted:
            raise ValueError("请先训练模型")

        if prediction_period is None:
            prediction_period = self.prediction_period

        logger.info(f"开始预测CLV，预测期{prediction_period}天，季节性调整: {apply_seasonality}")

        # 预测购买次数
        predicted_purchases = self.predict_purchases(
            prediction_period,
            data=self.rfm_data
        )

        # 计算CLV
        clv_predictions = []

        for i, (_, customer) in enumerate(self.rfm_data.iterrows()):
            # 预期购买次数
            expected_purchases = predicted_purchases[i]

            # 平均订单价值
            avg_order_value = customer['monetary']

            # 基础预期收入
            base_expected_revenue = expected_purchases * avg_order_value

            # 季节性调整
            if apply_seasonality and self.seasonal_learned:
                # 计算预测期内的季节性调整
                seasonal_adjustment = self._calculate_seasonal_adjustment(prediction_period)
                seasonal_expected_revenue = base_expected_revenue * seasonal_adjustment
            else:
                seasonal_adjustment = 1.0
                seasonal_expected_revenue = base_expected_revenue

            # 折现
            if discount_rate > 0:
                discount_factor = 1 / (1 + discount_rate * prediction_period / 365)
                base_discounted_clv = base_expected_revenue * discount_factor
                seasonal_discounted_clv = seasonal_expected_revenue * discount_factor
            else:
                base_discounted_clv = base_expected_revenue
                seasonal_discounted_clv = seasonal_expected_revenue

            clv_predictions.append({
                'customer_id': customer['customer_id'],
                'predicted_purchases': expected_purchases,
                'avg_order_value': avg_order_value,
                'base_expected_revenue': base_expected_revenue,
                'seasonal_adjustment': seasonal_adjustment,
                'seasonal_expected_revenue': seasonal_expected_revenue,
                'base_discounted_clv': base_discounted_clv,
                'seasonal_discounted_clv': seasonal_discounted_clv,
                'historical_frequency': customer['frequency'],
                'historical_monetary': customer['monetary'],
                'historical_total': customer['total_amount']
            })

        if apply_seasonality:
            self.seasonal_clv_predictions = pd.DataFrame(clv_predictions)
            logger.info(
                f"季节性CLV预测完成: 平均CLV {self.seasonal_clv_predictions['seasonal_discounted_clv'].mean():.2f}")
            return self.seasonal_clv_predictions
        else:
            self.clv_predictions = pd.DataFrame(clv_predictions)
            logger.info(f"基础CLV预测完成: 平均CLV {self.clv_predictions['base_discounted_clv'].mean():.2f}")
            return self.clv_predictions

    def _calculate_seasonal_adjustment(self, prediction_period):
        """计算预测期内的季节性调整因子"""
        if not self.seasonal_learned:
            return 1.0

        # 计算预测期起始月份
        start_date = self.observation_period_end
        end_date = start_date + timedelta(days=prediction_period)

        # 计算预测期内各月份的权重
        current_date = start_date
        monthly_weights = {}
        total_days = 0

        while current_date < end_date:
            month = current_date.month

            # 计算该月在预测期内的天数
            month_end = min(
                end_date,
                datetime(current_date.year + (1 if current_date.month == 12 else 0),
                         (current_date.month % 12) + 1, 1)
            )
            days_in_period = (month_end - current_date).days

            if month not in monthly_weights:
                monthly_weights[month] = 0
            monthly_weights[month] += days_in_period
            total_days += days_in_period

            # 移动到下个月
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        # 计算加权季节性调整因子
        weighted_seasonal_factor = 0
        for month, days in monthly_weights.items():
            weight = days / total_days
            seasonal_factor = self.seasonal_factors.get(month, 1.0)
            weighted_seasonal_factor += weight * seasonal_factor

        return weighted_seasonal_factor

    def validate_model(self, apply_seasonality=True):
        """模型验证（包含季节性调整验证）"""
        if not self.model_fitted or self.test_data is None:
            raise ValueError("请先训练模型并分割数据")

        logger.info(f"开始模型验证，季节性调整: {apply_seasonality}")

        # 在测试集上预测
        test_predictions = self.predict_purchases(
            self.prediction_period,
            data=self.test_data
        )

        # 计算实际购买次数
        prediction_start = self.observation_period_end
        prediction_end = prediction_start + timedelta(days=self.prediction_period)

        actual_purchases = []
        for _, customer in self.test_data.iterrows():
            customer_future_data = self.processed_data[
                (self.processed_data['customer_id'] == customer['customer_id']) &
                (self.processed_data['order_date'] > prediction_start) &
                (self.processed_data['order_date'] <= prediction_end)
                ]
            actual_purchases.append(len(customer_future_data))

        actual_purchases = np.array(actual_purchases)

        # 季节性调整预测
        if apply_seasonality and self.seasonal_learned:
            seasonal_adjustment = self._calculate_seasonal_adjustment(self.prediction_period)
            seasonal_predictions = test_predictions * seasonal_adjustment
        else:
            seasonal_predictions = test_predictions
            seasonal_adjustment = 1.0

        # 计算评估指标
        def calculate_metrics(actual, predicted, name):
            mae = mean_absolute_error(actual, predicted)
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)

            # 计算MAPE（避免除零）
            mask = actual > 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            else:
                mape = np.nan

            # 计算相关系数
            correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else np.nan

            return {
                f'{name}_mae': mae,
                f'{name}_mse': mse,
                f'{name}_rmse': rmse,
                f'{name}_mape': mape,
                f'{name}_correlation': correlation,
                f'{name}_actual_mean': np.mean(actual),
                f'{name}_predicted_mean': np.mean(predicted),
                f'{name}_actual_std': np.std(actual),
                f'{name}_predicted_std': np.std(predicted)
            }

        # 基础模型验证
        base_results = calculate_metrics(actual_purchases, test_predictions, 'base')

        # 季节性模型验证
        seasonal_results = calculate_metrics(actual_purchases, seasonal_predictions, 'seasonal')

        # 合并结果
        validation_results = {**base_results, **seasonal_results}
        validation_results['seasonal_adjustment_factor'] = seasonal_adjustment

        if apply_seasonality:
            self.seasonal_validation_results = validation_results
        else:
            self.validation_results = validation_results

        logger.info(f"模型验证完成:")
        logger.info(
            f"  基础模型 - MAE: {base_results['base_mae']:.4f}, MAPE: {base_results['base_mape']:.2f}%, 相关系数: {base_results['base_correlation']:.4f}")
        if apply_seasonality:
            logger.info(
                f"  季节性模型 - MAE: {seasonal_results['seasonal_mae']:.4f}, MAPE: {seasonal_results['seasonal_mape']:.2f}%, 相关系数: {seasonal_results['seasonal_correlation']:.4f}")
            logger.info(f"  季节性调整因子: {seasonal_adjustment:.3f}")

        return validation_results

    def create_visualizations(self, save_path=None):
        """创建可视化图表（包含季节性分析）"""
        logger.info("创建可视化图表...")

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('季节性MBG-NBD模型分析结果', fontsize=16, fontweight='bold')

        # 1. RFM分布
        ax1 = axes[0, 0]
        ax1.hist(self.rfm_data['frequency'], bins=30, alpha=0.7, color='blue')
        ax1.set_xlabel('购买频次')
        ax1.set_ylabel('客户数量')
        ax1.set_title('购买频次分布')
        ax1.grid(True, alpha=0.3)

        # 2. 货币价值分布
        ax2 = axes[0, 1]
        ax2.hist(self.rfm_data['monetary'], bins=30, alpha=0.7, color='green')
        ax2.set_xlabel('平均订单价值')
        ax2.set_ylabel('客户数量')
        ax2.set_title('货币价值分布')
        ax2.grid(True, alpha=0.3)

        # 3. 季节性因子
        ax3 = axes[0, 2]
        if self.seasonal_learned:
            months = list(range(1, 13))
            factors = [self.seasonal_factors[m] for m in months]
            ax3.plot(months, factors, marker='o', linewidth=2, color='red')
            ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_xlabel('月份')
            ax3.set_ylabel('季节性因子')
            ax3.set_title(f'季节性因子 ({self.seasonal_method}方法)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(months)

        # 4. CLV对比（基础vs季节性）
        ax4 = axes[1, 0]
        if self.clv_predictions is not None and self.seasonal_clv_predictions is not None:
            ax4.scatter(
                self.clv_predictions['base_discounted_clv'],
                self.seasonal_clv_predictions['seasonal_discounted_clv'],
                alpha=0.6
            )
            max_clv = max(
                self.clv_predictions['base_discounted_clv'].max(),
                self.seasonal_clv_predictions['seasonal_discounted_clv'].max()
            )
            ax4.plot([0, max_clv], [0, max_clv], 'r--', alpha=0.8)
            ax4.set_xlabel('基础CLV')
            ax4.set_ylabel('季节性调整CLV')
            ax4.set_title('基础CLV vs 季节性CLV')
            ax4.grid(True, alpha=0.3)

        # 5. 季节性CLV分布
        ax5 = axes[1, 1]
        if self.seasonal_clv_predictions is not None:
            ax5.hist(self.seasonal_clv_predictions['seasonal_discounted_clv'], bins=30, alpha=0.7, color='purple')
            ax5.set_xlabel('季节性调整CLV')
            ax5.set_ylabel('客户数量')
            ax5.set_title('季节性CLV分布')
            ax5.grid(True, alpha=0.3)

        # 6. 频次vs货币价值散点图
        ax6 = axes[1, 2]
        scatter = ax6.scatter(
            self.rfm_data['frequency'],
            self.rfm_data['monetary'],
            alpha=0.6,
            c=self.rfm_data['T'],
            cmap='viridis'
        )
        ax6.set_xlabel('购买频次')
        ax6.set_ylabel('平均订单价值')
        ax6.set_title('频次 vs 货币价值')
        plt.colorbar(scatter, ax=ax6, label='客户生命周期(天)')

        # 7. 模型验证结果对比
        ax7 = axes[2, 0]
        if hasattr(self, 'seasonal_validation_results') and self.seasonal_validation_results:
            methods = ['基础模型', '季节性模型']
            mae_values = [
                self.seasonal_validation_results['base_mae'],
                self.seasonal_validation_results['seasonal_mae']
            ]
            bars = ax7.bar(methods, mae_values, color=['blue', 'red'], alpha=0.7)
            ax7.set_ylabel('MAE')
            ax7.set_title('模型验证对比')
            ax7.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, mae_values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                         f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # 8. 季节性调整效果
        ax8 = axes[2, 1]
        if self.seasonal_clv_predictions is not None:
            adjustments = self.seasonal_clv_predictions['seasonal_adjustment']
            ax8.hist(adjustments, bins=20, alpha=0.7, color='orange')
            ax8.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='无调整')
            ax8.set_xlabel('季节性调整因子')
            ax8.set_ylabel('客户数量')
            ax8.set_title('季节性调整因子分布')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # 9. Top客户CLV排名
        ax9 = axes[2, 2]
        if self.seasonal_clv_predictions is not None:
            top_customers = self.seasonal_clv_predictions.nlargest(15, 'seasonal_discounted_clv')
            ax9.barh(range(len(top_customers)), top_customers['seasonal_discounted_clv'])
            ax9.set_xlabel('季节性调整CLV')
            ax9.set_ylabel('客户排名')
            ax9.set_title('Top 15 客户CLV')
            ax9.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存: {save_path}")

        plt.show()

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'params': self.params,
            'observation_period_end': self.observation_period_end,
            'prediction_period': self.prediction_period,
            'seasonal_method': self.seasonal_method,
            'model_fitted': self.model_fitted,
            'seasonal_factors': self.seasonal_factors,
            'seasonal_learned': self.seasonal_learned,
            'transition_matrix': self.transition_matrix,
            'emission_probabilities': self.emission_probabilities,
            'validation_results': self.validation_results,
            'seasonal_validation_results': getattr(self, 'seasonal_validation_results', None)
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"季节性模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.params = model_data['params']
        self.observation_period_end = model_data['observation_period_end']
        self.prediction_period = model_data['prediction_period']
        self.seasonal_method = model_data.get('seasonal_method', 'none')
        self.model_fitted = model_data['model_fitted']
        self.seasonal_factors = model_data.get('seasonal_factors')
        self.seasonal_learned = model_data.get('seasonal_learned', False)
        self.transition_matrix = model_data.get('transition_matrix')
        self.emission_probabilities = model_data.get('emission_probabilities')
        self.validation_results = model_data.get('validation_results')
        self.seasonal_validation_results = model_data.get('seasonal_validation_results')

        logger.info(f"季节性模型已加载: {filepath}")

    def generate_report(self, save_path=None):
        """生成分析报告"""
        report = {
            'model_summary': {
                'model_type': 'Seasonal MBG-NBD',
                'seasonal_method': self.seasonal_method,
                'observation_period_end': str(self.observation_period_end),
                'prediction_period_days': self.prediction_period,
                'total_customers': len(self.rfm_data) if self.rfm_data is not None else 0,
                'model_fitted': self.model_fitted,
                'seasonal_learned': self.seasonal_learned
            },
            'model_parameters': {
                'r': float(self.params[0]) if self.params is not None else None,
                'alpha': float(self.params[1]) if self.params is not None else None,
                'a': float(self.params[2]) if self.params is not None else None,
                'b': float(self.params[3]) if self.params is not None else None
            } if self.params is not None else None,
            'seasonal_factors': {str(k): float(v) for k, v in
                                 self.seasonal_factors.items()} if self.seasonal_factors else None,
            'validation_results': self.validation_results,
            'seasonal_validation_results': getattr(self, 'seasonal_validation_results', None),
            'clv_summary': {
                'base_total_clv': float(
                    self.clv_predictions['base_discounted_clv'].sum()) if self.clv_predictions is not None else None,
                'base_average_clv': float(
                    self.clv_predictions['base_discounted_clv'].mean()) if self.clv_predictions is not None else None,
                'seasonal_total_clv': float(self.seasonal_clv_predictions[
                                                'seasonal_discounted_clv'].sum()) if self.seasonal_clv_predictions is not None else None,
                'seasonal_average_clv': float(self.seasonal_clv_predictions[
                                                  'seasonal_discounted_clv'].mean()) if self.seasonal_clv_predictions is not None else None,
                'seasonal_improvement': float(
                    (self.seasonal_clv_predictions['seasonal_discounted_clv'].sum() -
                     self.clv_predictions['base_discounted_clv'].sum()) /
                    self.clv_predictions['base_discounted_clv'].sum() * 100
                ) if (self.clv_predictions is not None and self.seasonal_clv_predictions is not None) else None
            } if (self.clv_predictions is not None or self.seasonal_clv_predictions is not None) else None
        }

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"分析报告已保存: {save_path}")

        return report


def main():
    """主函数 - 完整的季节性MBG-NBD使用示例"""
    print("🚀 季节性MBG-NBD模型完整示例")
    print("=" * 60)

    # 1. 初始化模型（使用马尔可夫季节性方法）
    model = SeasonalMBGNBDModel(
        prediction_period=90,
        seasonal_method='markov'  # 可选: 'markov', 'preset', 'none'
    )

    # 2. 加载数据
    try:
        data = model.load_data('/home/ubuntu/upload/manifest.csv')
        print(f"✅ 数据加载成功: {len(data)}条记录")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # 3. 学习季节性因子
    seasonal_factors = model.learn_seasonal_factors()
    print(f"✅ 季节性因子学习完成: {model.seasonal_method}方法")

    # 4. 创建RFM特征
    rfm_data = model.create_rfm_features()
    print(f"✅ RFM特征创建完成: {len(rfm_data)}个客户")

    # 5. 分割数据
    train_data, test_data = model.split_data(test_size=0.2)
    print(f"✅ 数据分割完成: 训练集{len(train_data)}, 测试集{len(test_data)}")

    # 6. 训练模型
    try:
        params = model.fit()
        print(f"✅ 模型训练完成")
        print(f"   参数: r={params[0]:.4f}, alpha={params[1]:.4f}, a={params[2]:.4f}, b={params[3]:.4f}")
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return

    # 7. 预测基础CLV
    base_clv = model.predict_clv(apply_seasonality=False)
    print(f"✅ 基础CLV预测完成: 平均CLV {base_clv['base_discounted_clv'].mean():.2f}")

    # 8. 预测季节性调整CLV
    seasonal_clv = model.predict_clv(apply_seasonality=True)
    print(f"✅ 季节性CLV预测完成: 平均CLV {seasonal_clv['seasonal_discounted_clv'].mean():.2f}")

    # 9. 计算季节性改进
    base_total = base_clv['base_discounted_clv'].sum()
    seasonal_total = seasonal_clv['seasonal_discounted_clv'].sum()
    improvement = (seasonal_total - base_total) / base_total * 100
    print(f"✅ 季节性改进: {improvement:+.2f}%")

    # 10. 模型验证
    try:
        validation_results = model.validate_model(apply_seasonality=True)
        print(f"✅ 模型验证完成:")
        print(f"   基础模型 - MAE: {validation_results['base_mae']:.4f}, MAPE: {validation_results['base_mape']:.2f}%")
        print(
            f"   季节性模型 - MAE: {validation_results['seasonal_mae']:.4f}, MAPE: {validation_results['seasonal_mape']:.2f}%")
    except Exception as e:
        print(f"⚠️ 模型验证失败: {e}")

    # 11. 创建可视化
    model.create_visualizations('/home/ubuntu/seasonal_mbgnbd_analysis.png')
    print("✅ 可视化图表已创建")

    # 12. 保存模型
    model.save_model('/home/ubuntu/seasonal_mbgnbd_model.pkl')
    print("✅ 季节性模型已保存")

    # 13. 生成报告
    report = model.generate_report('/home/ubuntu/seasonal_mbgnbd_report.json')
    print("✅ 分析报告已生成")

    # 14. 保存结果
    if seasonal_clv is not None:
        seasonal_clv.to_csv('/home/ubuntu/seasonal_clv_predictions.csv', index=False, encoding='utf-8-sig')
        print("✅ 季节性CLV预测结果已保存")

    print("\n🎉 季节性MBG-NBD模型分析完成!")
    print(f"📊 总客户数: {len(rfm_data)}")
    print(f"💰 基础总CLV: {base_total:.2f}")
    print(f"🌟 季节性总CLV: {seasonal_total:.2f}")
    print(f"📈 季节性改进: {improvement:+.2f}%")
    print(f"🎯 季节性方法: {model.seasonal_method}")

    return model


if __name__ == "__main__":
    # 运行完整示例
    model = main()