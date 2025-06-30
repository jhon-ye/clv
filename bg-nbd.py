#!/usr/bin/env python3
"""
改进的BG-NBD模型实现
基于真实数据特征优化的模型配置和训练流程
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json
import os
import pickle
from typing import Dict, Tuple, Optional, Any
import logging
from datetime import datetime

# 设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedBGNBDModel:
    """
    改进的BG-NBD模型
    包含优化的先验分布、采样策略和收敛诊断
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化模型

        Args:
            config: 模型配置参数
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.trace = None
        self.model_summary = None
        self.convergence_diagnostics = {}

    def _get_default_config(self) -> Dict:
        """获取默认模型配置"""
        return {
            'priors': {
                'a_prior': {'dist': 'Gamma', 'kwargs': {'alpha': 1.0, 'beta': 1.0}},
                'b_prior': {'dist': 'Gamma', 'kwargs': {'alpha': 2.0, 'beta': 2.0}},
                'alpha_prior': {'dist': 'Gamma', 'kwargs': {'alpha': 1.0, 'beta': 1.0}},
                'r_prior': {'dist': 'Gamma', 'kwargs': {'alpha': 1.0, 'beta': 1.0}}
            },
            'sampling': {
                'draws': 3000,
                'tune': 2000,
                'chains': 4,
                'cores': 4,
                'target_accept': 0.95,
                'max_treedepth': 12,
                'random_seed': 42
            },
            'convergence': {
                'rhat_threshold': 1.1,
                'ess_threshold': 400,
                'energy_threshold': 0.05
            }
        }

    def _adaptive_prior_configuration(self, rfm_data: pd.DataFrame) -> Dict:
        """
        基于数据特征的自适应先验配置

        Args:
            rfm_data: RFM-T数据

        Returns:
            优化的先验配置
        """
        logger.info("配置自适应先验分布...")

        # 分析数据特征
        freq_stats = rfm_data['frequency'].describe()
        monetary_stats = rfm_data['monetary_value'].describe()
        recency_stats = rfm_data['recency'].describe()
        T_stats = rfm_data['T'].describe()

        # 基于数据分布特征调整先验
        freq_mean = freq_stats['mean']
        freq_std = freq_stats['std']

        # 避免除零错误
        if freq_std > 0:
            freq_alpha = max(1.0, (freq_mean ** 2) / (freq_std ** 2))
            freq_beta = max(1.0, freq_mean / (freq_std ** 2))
        else:
            freq_alpha = 1.0
            freq_beta = 1.0

        adaptive_priors = {
            'a_prior': {
                'dist': 'Gamma',
                'kwargs': {'alpha': freq_alpha, 'beta': freq_beta}
            },
            'b_prior': {
                'dist': 'Gamma',
                'kwargs': {'alpha': 2.0, 'beta': 2.0}
            },
            'alpha_prior': {
                'dist': 'Gamma',
                'kwargs': {'alpha': 1.0, 'beta': 1.0}
            },
            'r_prior': {
                'dist': 'Gamma',
                'kwargs': {'alpha': 1.0, 'beta': 1.0}
            }
        }

        logger.info(f"自适应先验配置完成:")
        logger.info(f"  a_prior: Gamma(α={freq_alpha:.2f}, β={freq_beta:.2f})")
        logger.info(f"  基于频次统计: 均值={freq_mean:.2f}, 标准差={freq_std:.2f}")

        return adaptive_priors

    def _dynamic_sampling_config(self, data_size: int) -> Dict:
        """
        基于数据规模的动态采样配置

        Args:
            data_size: 数据样本量

        Returns:
            优化的采样配置
        """
        base_config = self.config['sampling'].copy()

        # 根据数据规模调整采样参数
        if data_size > 10000:
            # 大数据集：增加采样次数
            base_config['draws'] = 4000
            base_config['tune'] = 3000
        elif data_size > 5000:
            # 中等数据集：标准配置
            base_config['draws'] = 3000
            base_config['tune'] = 2000
        else:
            # 小数据集：减少采样次数但增加链数
            base_config['draws'] = 2000
            base_config['tune'] = 1500
            base_config['chains'] = 6

        logger.info(f"动态采样配置 (数据量={data_size}):")
        logger.info(f"  draws={base_config['draws']}, tune={base_config['tune']}")
        logger.info(f"  chains={base_config['chains']}, target_accept={base_config['target_accept']}")

        return base_config

    def build_model(self, rfm_data: pd.DataFrame, use_adaptive_priors: bool = True) -> None:
        """
        构建BG-NBD模型

        Args:
            rfm_data: RFM-T数据
            use_adaptive_priors: 是否使用自适应先验
        """
        logger.info("构建BG-NBD模型...")

        # 准备数据
        frequency = rfm_data['frequency'].values
        recency = rfm_data['recency'].values
        T = rfm_data['T'].values

        # 选择先验配置
        if use_adaptive_priors:
            priors = self._adaptive_prior_configuration(rfm_data)
        else:
            priors = self.config['priors']

        # 构建PyMC模型
        with pm.Model() as model:
            # 先验分布
            a = self._create_prior('a', priors['a_prior'])
            b = self._create_prior('b', priors['b_prior'])
            alpha = self._create_prior('alpha', priors['alpha_prior'])
            r = self._create_prior('r', priors['r_prior'])

            # BG-NBD似然函数
            def bgnbd_logp(frequency, recency, T, a, b, alpha, r):
                """BG-NBD对数似然函数"""

                # 避免数值问题
                a = pm.math.maximum(a, 1e-8)
                b = pm.math.maximum(b, 1e-8)
                alpha = pm.math.maximum(alpha, 1e-8)
                r = pm.math.maximum(r, 1e-8)

                # 计算对数似然
                # 第一部分：购买过程
                logp_purchase = (
                        pm.math.gammaln(r + frequency) - pm.math.gammaln(r) +
                        r * pm.math.log(alpha) - (r + frequency) * pm.math.log(alpha + T)
                )

                # 第二部分：流失过程
                # 对于frequency > 0的客户
                mask_freq_pos = frequency > 0

                logp_churn_pos = pm.math.switch(
                    mask_freq_pos,
                    pm.math.log(
                        pm.math.gammaln(a + b + frequency - 1) - pm.math.gammaln(a) - pm.math.gammaln(b) +
                        pm.math.gammaln(a + frequency - 1) * pm.math.gammaln(b) / pm.math.gammaln(a + b) +
                        (a + frequency - 1) * pm.math.log(recency) + b * pm.math.log(T - recency) -
                        (a + b + frequency - 1) * pm.math.log(T)
                    ),
                    0.0
                )

                # 对于frequency = 0的客户
                logp_churn_zero = pm.math.switch(
                    ~mask_freq_pos,
                    pm.math.log(
                        pm.math.gammaln(a + b) / (pm.math.gammaln(a) * pm.math.gammaln(b)) *
                        pm.math.pow(T, -(a + b))
                    ),
                    0.0
                )

                logp_churn = logp_churn_pos + logp_churn_zero

                return logp_purchase + logp_churn

            # 自定义似然
            likelihood = pm.DensityDist(
                'likelihood',
                logp=lambda value: bgnbd_logp(frequency, recency, T, a, b, alpha, r),
                observed=np.zeros(len(frequency))  # 占位符
            )

        self.model = model
        logger.info("BG-NBD模型构建完成")

    def _create_prior(self, name: str, prior_config: Dict) -> Any:
        """创建先验分布"""
        dist_name = prior_config['dist']
        kwargs = prior_config['kwargs']

        if dist_name == 'Gamma':
            return pm.Gamma(name, alpha=kwargs['alpha'], beta=kwargs['beta'])
        elif dist_name == 'HalfNormal':
            return pm.HalfNormal(name, sigma=kwargs['sigma'])
        elif dist_name == 'Exponential':
            return pm.Exponential(name, lam=kwargs['lam'])
        else:
            raise ValueError(f"不支持的分布类型: {dist_name}")

    def fit(self, rfm_data: pd.DataFrame, use_adaptive_priors: bool = True) -> None:
        """
        训练模型

        Args:
            rfm_data: RFM-T数据
            use_adaptive_priors: 是否使用自适应先验
        """
        logger.info("开始模型训练...")

        # 构建模型
        self.build_model(rfm_data, use_adaptive_priors)

        # 获取采样配置
        sampling_config = self._dynamic_sampling_config(len(rfm_data))

        # MCMC采样
        with self.model:
            logger.info("开始MCMC采样...")
            start_time = datetime.now()

            try:
                self.trace = pm.sample(
                    draws=sampling_config['draws'],
                    tune=sampling_config['tune'],
                    chains=sampling_config['chains'],
                    cores=sampling_config['cores'],
                    target_accept=sampling_config['target_accept'],
                    max_treedepth=sampling_config['max_treedepth'],
                    random_seed=sampling_config['random_seed'],
                    return_inferencedata=True
                )

                end_time = datetime.now()
                training_time = (end_time - start_time).total_seconds()

                logger.info(f"MCMC采样完成，耗时: {training_time:.1f} 秒")

                # 执行收敛诊断
                self._perform_convergence_diagnostics()

                # 生成模型摘要
                self._generate_model_summary()

            except Exception as e:
                logger.error(f"模型训练失败: {e}")
                raise

    def _perform_convergence_diagnostics(self) -> None:
        """执行收敛性诊断"""
        logger.info("执行收敛性诊断...")

        # R-hat统计量
        rhat = az.rhat(self.trace)
        max_rhat = float(rhat.max())

        # 有效样本量
        ess = az.ess(self.trace)
        min_ess = float(ess.min())

        # 能量诊断
        energy_stats = az.bfmi(self.trace)
        min_bfmi = float(energy_stats.min()) if hasattr(energy_stats, 'min') else float(energy_stats)

        # 发散转换
        divergent = self.trace.sample_stats.diverging.sum().values
        total_divergent = int(divergent.sum()) if hasattr(divergent, 'sum') else int(divergent)

        # 最大树深度
        max_treedepth = self.trace.sample_stats.tree_depth.max().values
        max_td = int(max_treedepth.max()) if hasattr(max_treedepth, 'max') else int(max_treedepth)

        self.convergence_diagnostics = {
            'rhat_max': max_rhat,
            'ess_min': min_ess,
            'bfmi_min': min_bfmi,
            'divergent_transitions': total_divergent,
            'max_treedepth': max_td,
            'convergence_status': self._assess_convergence(max_rhat, min_ess, min_bfmi, total_divergent)
        }

        logger.info("收敛性诊断结果:")
        logger.info(f"  R-hat最大值: {max_rhat:.4f} (< {self.config['convergence']['rhat_threshold']})")
        logger.info(f"  ESS最小值: {min_ess:.0f} (> {self.config['convergence']['ess_threshold']})")
        logger.info(f"  BFMI最小值: {min_bfmi:.4f}")
        logger.info(f"  发散转换: {total_divergent} 次")
        logger.info(f"  收敛状态: {self.convergence_diagnostics['convergence_status']}")

    def _assess_convergence(self, max_rhat: float, min_ess: float,
                            min_bfmi: float, divergent: int) -> str:
        """评估收敛状态"""
        issues = []

        if max_rhat > self.config['convergence']['rhat_threshold']:
            issues.append(f"R-hat过高({max_rhat:.4f})")

        if min_ess < self.config['convergence']['ess_threshold']:
            issues.append(f"ESS过低({min_ess:.0f})")

        if min_bfmi < self.config['convergence']['energy_threshold']:
            issues.append(f"能量诊断异常({min_bfmi:.4f})")

        if divergent > 0:
            issues.append(f"存在发散转换({divergent}次)")

        if not issues:
            return "已收敛"
        else:
            return f"收敛问题: {'; '.join(issues)}"

    def _generate_model_summary(self) -> None:
        """生成模型摘要"""
        logger.info("生成模型摘要...")

        # 参数统计摘要
        summary = az.summary(self.trace, var_names=['a', 'b', 'alpha', 'r'])

        self.model_summary = {
            'parameters': summary.to_dict(),
            'convergence': self.convergence_diagnostics,
            'model_info': {
                'chains': self.trace.posterior.chain.size,
                'draws': self.trace.posterior.draw.size,
                'parameters': len(summary)
            }
        }

        logger.info("模型参数估计:")
        for param in ['a', 'b', 'alpha', 'r']:
            if param in summary.index:
                mean_val = summary.loc[param, 'mean']
                std_val = summary.loc[param, 'sd']
                hdi_lower = summary.loc[param, 'hdi_3%']
                hdi_upper = summary.loc[param, 'hdi_97%']
                logger.info(f"  {param}: {mean_val:.4f} ± {std_val:.4f} [{hdi_lower:.4f}, {hdi_upper:.4f}]")

    def predict_clv(self, rfm_data: pd.DataFrame, prediction_period: int = 365) -> pd.DataFrame:
        """
        预测客户生命周期价值

        Args:
            rfm_data: RFM-T数据
            prediction_period: 预测期长度（天）

        Returns:
            包含CLV预测的DataFrame
        """
        logger.info(f"预测未来{prediction_period}天的CLV...")

        if self.trace is None:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        # 获取参数后验样本
        posterior_samples = self.trace.posterior
        a_samples = posterior_samples['a'].values.flatten()
        b_samples = posterior_samples['b'].values.flatten()
        alpha_samples = posterior_samples['alpha'].values.flatten()
        r_samples = posterior_samples['r'].values.flatten()

        # 准备预测结果
        predictions = []

        for idx, row in rfm_data.iterrows():
            customer_id = row['customer_id']
            frequency = row['frequency']
            recency = row['recency']
            T = row['T']
            monetary_value = row['monetary_value']

            # 计算预期购买次数
            expected_purchases = self._calculate_expected_purchases(
                frequency, recency, T, prediction_period,
                a_samples, b_samples, alpha_samples, r_samples
            )

            # 计算CLV
            clv_mean = expected_purchases['mean'] * monetary_value
            clv_std = expected_purchases['std'] * monetary_value
            clv_lower = expected_purchases['lower'] * monetary_value
            clv_upper = expected_purchases['upper'] * monetary_value

            predictions.append({
                'customer_id': customer_id,
                'historical_frequency': frequency,
                'historical_monetary': monetary_value,
                'predicted_purchases_mean': expected_purchases['mean'],
                'predicted_purchases_std': expected_purchases['std'],
                'predicted_purchases_lower': expected_purchases['lower'],
                'predicted_purchases_upper': expected_purchases['upper'],
                'predicted_clv_mean': clv_mean,
                'predicted_clv_std': clv_std,
                'predicted_clv_lower': clv_lower,
                'predicted_clv_upper': clv_upper,
                'prediction_period_days': prediction_period
            })

        predictions_df = pd.DataFrame(predictions)

        logger.info(f"CLV预测完成，平均CLV: {predictions_df['predicted_clv_mean'].mean():.2f}")

        return predictions_df

    def _calculate_expected_purchases(self, frequency: int, recency: int, T: int,
                                      prediction_period: int, a_samples: np.ndarray,
                                      b_samples: np.ndarray, alpha_samples: np.ndarray,
                                      r_samples: np.ndarray) -> Dict[str, float]:
        """计算预期购买次数"""

        # 计算每个后验样本的预期购买次数
        expected_purchases_samples = []

        for a, b, alpha, r in zip(a_samples, b_samples, alpha_samples, r_samples):
            # P(alive)
            if frequency > 0:
                p_alive = 1 / (1 + (b / (b + T - recency)) *
                               ((alpha + T) / alpha) ** (r + frequency))
            else:
                p_alive = 1 / (1 + (b / (b + T)) * ((alpha + T) / alpha) ** r)

            # 预期购买率
            lambda_rate = (r + frequency) / (alpha + T)

            # 预期购买次数
            expected_purchases = p_alive * lambda_rate * prediction_period
            expected_purchases_samples.append(expected_purchases)

        expected_purchases_samples = np.array(expected_purchases_samples)

        return {
            'mean': np.mean(expected_purchases_samples),
            'std': np.std(expected_purchases_samples),
            'lower': np.percentile(expected_purchases_samples, 2.5),
            'upper': np.percentile(expected_purchases_samples, 97.5)
        }

    def create_diagnostic_plots(self, output_dir: str = './output') -> None:
        """创建诊断图表"""
        logger.info("创建模型诊断图表...")

        os.makedirs(output_dir, exist_ok=True)

        # 1. 轨迹图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MCMC轨迹图', fontsize=16)

        params = ['a', 'b', 'alpha', 'r']
        for i, param in enumerate(params):
            ax = axes[i // 2, i % 2]
            az.plot_trace(self.trace, var_names=[param], axes=ax)
            ax.set_title(f'参数 {param}')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/mcmc_trace_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 参数分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('参数后验分布', fontsize=16)

        for i, param in enumerate(params):
            ax = axes[i // 2, i % 2]
            az.plot_posterior(self.trace, var_names=[param], ax=ax)
            ax.set_title(f'参数 {param} 后验分布')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/parameter_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 收敛诊断图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('收敛诊断', fontsize=16)

        # R-hat
        rhat_values = az.rhat(self.trace)
        axes[0, 0].bar(range(len(params)), [float(rhat_values[p]) for p in params])
        axes[0, 0].set_xticks(range(len(params)))
        axes[0, 0].set_xticklabels(params)
        axes[0, 0].axhline(y=1.1, color='red', linestyle='--', label='阈值')
        axes[0, 0].set_title('R-hat统计量')
        axes[0, 0].legend()

        # ESS
        ess_values = az.ess(self.trace)
        axes[0, 1].bar(range(len(params)), [float(ess_values[p]) for p in params])
        axes[0, 1].set_xticks(range(len(params)))
        axes[0, 1].set_xticklabels(params)
        axes[0, 1].axhline(y=400, color='red', linestyle='--', label='阈值')
        axes[0, 1].set_title('有效样本量 (ESS)')
        axes[0, 1].legend()

        # 能量图
        az.plot_energy(self.trace, ax=axes[1, 0])
        axes[1, 0].set_title('能量诊断')

        # 森林图
        az.plot_forest(self.trace, var_names=params, ax=axes[1, 1])
        axes[1, 1].set_title('参数估计森林图')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"诊断图表已保存到: {output_dir}")

    def save_model(self, output_dir: str = './output') -> None:
        """保存模型"""
        logger.info("保存模型...")

        os.makedirs(output_dir, exist_ok=True)

        # 保存trace
        if self.trace is not None:
            self.trace.to_netcdf(f'{output_dir}/model_trace.nc')

        # 保存模型摘要
        if self.model_summary is not None:
            with open(f'{output_dir}/model_summary.json', 'w', encoding='utf-8') as f:
                json.dump(self.model_summary, f, indent=2, ensure_ascii=False, default=str)

        # 保存配置
        with open(f'{output_dir}/model_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info(f"模型已保存到: {output_dir}")

    def load_model(self, model_dir: str) -> None:
        """加载模型"""
        logger.info(f"从 {model_dir} 加载模型...")

        # 加载trace
        trace_path = f'{model_dir}/model_trace.nc'
        if os.path.exists(trace_path):
            self.trace = az.from_netcdf(trace_path)

        # 加载模型摘要
        summary_path = f'{model_dir}/model_summary.json'
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                self.model_summary = json.load(f)

        # 加载配置
        config_path = f'{model_dir}/model_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

        logger.info("模型加载完成")


def main():
    """主函数 - 演示模型训练流程"""

    # 加载预处理后的数据
    try:
        rfm_data = pd.read_csv('/Users/changyu/Downloads/CLV/rfm_data.csv')
        logger.info(f"加载RFM数据: {len(rfm_data)} 个客户")
    except FileNotFoundError:
        logger.error("未找到预处理数据，请先运行数据预处理模块")
        return

    # 创建模型
    model = ImprovedBGNBDModel()

    # 训练模型
    try:
        model.fit(rfm_data, use_adaptive_priors=True)

        # 创建诊断图表
        model.create_diagnostic_plots('/Users/changyu/Downloads/CLV/clv_model_output')

        # 预测CLV
        predictions = model.predict_clv(rfm_data, prediction_period=365)

        # 保存结果
        predictions.to_csv('/Users/changyu/Downloads/CLV/clv_model_output/clv_predictions.csv', index=False)
        model.save_model('/Users/changyu/Downloads/CLV/clv_model_output')

        print("\n" + "=" * 60)
        print("BG-NBD模型训练完成!")
        print("=" * 60)

        print(f"\n收敛诊断:")
        print(f"  收敛状态: {model.convergence_diagnostics['convergence_status']}")
        print(f"  R-hat最大值: {model.convergence_diagnostics['rhat_max']:.4f}")
        print(f"  ESS最小值: {model.convergence_diagnostics['ess_min']:.0f}")

        print(f"\nCLV预测结果:")
        print(f"  平均CLV: ¥{predictions['predicted_clv_mean'].mean():.2f}")
        print(f"  CLV中位数: ¥{predictions['predicted_clv_mean'].median():.2f}")
        print(f"  CLV标准差: ¥{predictions['predicted_clv_mean'].std():.2f}")

        print(f"\n输出文件:")
        print(f"  - CLV预测: ./clv_model_output/clv_predictions.csv")
        print(f"  - 模型文件: ./clv_model_output/model_trace.nc")
        print(f"  - 诊断图表: ./clv_model_output/")

    except Exception as e:
        logger.error(f"模型训练失败: {e}")
        raise


if __name__ == "__main__":
    main()