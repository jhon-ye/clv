# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的客户分层系统
基于预测CLV + 客户活跃度 + 购买行为的综合分层逻辑
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class OptimizedCustomerSegmentation:
    def __init__(self):
        self.data = None
        self.segmentation_results = None
        self.clv_thresholds = {}

    def load_data(self):
        """加载CLV预测数据"""
        print("🚀 优化的客户分层系统")
        print("=" * 60)
        print("📋 分层逻辑设计原则:")
        print("   1. 以预测CLV为核心价值指标")
        print("   2. 结合客户活跃度（Recency）")
        print("   3. 考虑购买行为稳定性（Frequency）")
        print("   4. 确保分层占比符合商业常识")
        print("   5. 便于制定差异化营销策略")
        print("=" * 60)

        try:
            self.data = pd.read_csv("/Users/changyu/Downloads/CLV/manifest2_enhanced_clv_predictions.csv")
            print(f"✅ 数据加载成功: {len(self.data):,}个客户")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False

    def analyze_data_characteristics(self):
        """深度分析数据特征，为分层设计提供依据"""
        print("\n📊 1. 数据特征深度分析")
        print("-" * 50)

        # CLV分析
        clv_stats = self.data["predicted_clv"].describe()
        print("CLV分布特征:")
        print(f'  - 总客户数: {len(self.data):,}')
        print(f'  - 总预测CLV: {self.data["predicted_clv"].sum():,.2f}元')
        print(f'  - 平均CLV: {clv_stats["mean"]:,.2f}元')
        print(f'  - 中位数CLV: {clv_stats["50%"]:,.2f}元')
        print(f'  - 标准差: {clv_stats["std"]:,.2f}元')
        print(f'  - 变异系数: {clv_stats["std"] / clv_stats["mean"]:.2f}')

        # 计算关键分位数
        percentiles = [50, 70, 80, 85, 90, 95, 98, 99, 99.5]
        print(f'\nCLV分位数分析:\n')
        for p in percentiles:
            value = np.percentile(self.data["predicted_clv"], p)
            self.clv_thresholds[p] = value
            count = (self.data["predicted_clv"] >= value).sum()
            print(f'  - {p:4.1f}%分位数: {value:8,.2f}元 (≥此值: {count:,}人, {count / len(self.data) * 100:.1f}%)\n')

        # Recency分析
        recency_stats = self.data["recency"].describe(percentiles=[.25, .5, .75, .9])
        print(f'\nRecency分布特征:\n')
        print(f'  - 平均: {recency_stats["mean"]:.1f}天\n')
        print(f'  - 中位数: {recency_stats["50%"]:.1f}天\n')
        print(f'  - 75%分位数: {recency_stats["75%"]:.1f}天\n')
        print(f'  - 90%分位数: {recency_stats["90%"]:.1f}天\n')

        # Frequency分析
        frequency_stats = self.data["frequency"].describe(percentiles=[.25, .5, .75, .9])
        print(f'\nFrequency分布特征:\n')
        print(f'  - 平均: {frequency_stats["mean"]:.1f}次\n')
        print(f'  - 中位数: {frequency_stats["50%"]:.1f}次\n')
        print(f'  - 75%分位数: {frequency_stats["75%"]:.1f}次\n')
        print(f'  - 90%分位数: {frequency_stats["90%"]:.1f}次\n')

        # 零频率客户分析
        zero_freq = (self.data["frequency"] == 0).sum()
        print(f'\n特殊客户群体:\n')
        print(f'  - 零频率客户: {zero_freq:,}人 ({zero_freq / len(self.data) * 100:.1f}%)\n')

        return self.clv_thresholds

    def design_segmentation_logic(self):
        """设计优化的分层逻辑"""
        print("\n🎯 2. 设计优化的分层逻辑")
        print("-" * 50)

        print("分层设计思路:")
        print("  🏆 Champions (冠军客户): 2-5% - 超高CLV + 高活跃度")
        print("  💎 VIP (重要客户): 8-12% - 高CLV + 中高活跃度")
        print("  🤝 Loyal (忠诚客户): 15-20% - 中高CLV + 稳定购买")
        print("  🌱 Potential (潜力客户): 25-35% - 中等CLV + 有成长空间")
        print("  ⚠️  At_Risk (风险客户): 5-10% - 曾经高价值但活跃度下降")
        print("  😴 Hibernating (休眠客户): 15-25% - 低活跃度但有历史价值")
        print("  💔 Lost (流失客户): 20-30% - 低CLV + 低活跃度")

        def optimized_segmentation(row):
            """优化的分层逻辑"""
            clv = row["predicted_clv"]
            recency = row["recency"]
            frequency = row["frequency"]
            monetary = row["monetary"]

            # 定义CLV阈值
            ultra_high_clv = self.clv_thresholds.get(99.5, 0)  # 前0.5%
            very_high_clv = self.clv_thresholds.get(98, 0)  # 前2%
            high_clv = self.clv_thresholds.get(90, 0)  # 前10%
            medium_clv = self.clv_thresholds.get(70, 0)  # 前30%
            low_clv = self.clv_thresholds.get(50, 0)  # 前50%

            # 定义活跃度阈值 (天)
            very_active_days = 15  # 30天内
            active_days = 30  # 90天内
            semi_active_days = 60  # 180天内
            inactive_days = 90  # 365天内
            long_inactive_days = 180  # 1.5年未购买

            # 定义频率阈值 (根据实际数据分布调整)
            if not hasattr(self, "_frequency_thresholds"):
                self._frequency_thresholds = {
                    "high": np.percentile(self.data["frequency"], 90),
                    "medium": np.percentile(self.data["frequency"], 50),
                    "low": np.percentile(self.data["frequency"], 10)
                }
            high_freq = self._frequency_thresholds["high"]
            medium_freq = self._frequency_thresholds["medium"]
            low_freq = self._frequency_thresholds["low"]

            # 分层逻辑 - 调整优先级，确保Recency低的客户优先被分类到活跃层
            # 优先判断活跃客户，避免近期有订单的客户被错误归类

            # 1. Champions: CLV极高且近期活跃
            if recency <= very_active_days and clv >= ultra_high_clv:
                return "Champions"

            # 2. VIP: CLV高且活跃
            elif recency <= active_days and clv >= very_high_clv:
                return "VIP"

            # 3. Loyal: CLV中高且稳定购买 (Recency在半活跃期内)
            elif recency <= semi_active_days and clv >= high_clv:
                return "Loyal"

            # 4. Potential: CLV中等，有成长潜力 (近期活跃，但CLV或频率未达高价值)
            elif recency <= active_days and clv >= low_clv:
                return "Potential"

            # 5. At_Risk: CLV高但活跃度下降 (曾经高价值，现在有流失风险)
            # 确保Recency在半活跃期到非活跃期之间，且CLV较高
            elif clv >= high_clv and recency > semi_active_days and recency <= inactive_days:
                return "At_Risk"

            # 6. Hibernating: CLV中等偏低，但长时间不活跃 (休眠客户)
            # 确保Recency在活跃期到非活跃期之间，且CLV中等偏低
            elif clv >= low_clv and recency > active_days and recency <= inactive_days:
                return "Hibernating"

            # 7. Lost: 严格定义，CLV低且长期不活跃，或零频率且长期不活跃
            # 只有当Recency超过一个很长的时间（例如1.5年）才考虑为Lost
            elif recency > long_inactive_days:
                return "Lost"
            # 零频率客户且Recency超过inactive_days
            elif frequency == 0 and recency > inactive_days:
                return "Lost"
            # 兜底：如果以上条件都不满足，但CLV极低且Recency也较长，可以考虑为Lost
            elif clv < low_clv and recency > semi_active_days:
                return "Lost"
            # 否则，归类为其他（例如，如果Recency很低，但CLV也低，则可能是新客户或低价值活跃客户，不应直接Lost）
            else:
                return "Potential"  # 兜底，避免将近期活跃客户错误归为Lost

        # 应用分层逻辑
        self.data["optimized_segment"] = self.data.apply(optimized_segmentation, axis=1)

        return True

    def analyze_segmentation_results(self):
        """分析分层结果"""
        print("\n📈 3. 分层结果分析")
        print("-" * 50)

        # 计算分层统计
        segment_stats = self.data.groupby("optimized_segment").agg({
            "customer_id": "count",
            "predicted_clv": ["sum", "mean", "std"],
            "recency": ["mean", "std"],
            "frequency": ["mean", "std"],
            "monetary": ["mean", "std"]
        }).round(2)

        # 扁平化列名
        segment_stats.columns = [
            "customer_count", "total_clv", "avg_clv", "clv_std",
            "avg_recency", "recency_std", "avg_frequency", "frequency_std",
            "avg_monetary", "monetary_std"
        ]

        # 计算占比
        total_customers = len(self.data)
        total_clv = self.data["predicted_clv"].sum()

        segment_stats["customer_pct"] = (segment_stats["customer_count"] / total_customers * 100).round(1)
        segment_stats["clv_pct"] = (segment_stats["total_clv"] / total_clv * 100).round(1)
        segment_stats["clv_efficiency"] = (segment_stats["clv_pct"] / segment_stats["customer_pct"]).round(2)

        self.segmentation_results = segment_stats

        # 按CLV贡献排序显示
        sorted_segments = segment_stats.sort_values("clv_pct", ascending=False)

        print("优化分层结果:")
        print(f"{'分层':<12} {'客户数':<7} {'客户%':<6} {'CLV%':<7} {'效率':<6} {'平均CLV':<10} {'平均Recency':<12}")
        print("-" * 75)

        for segment in sorted_segments.index:
            row = sorted_segments.loc[segment]
            print(f"{segment:<12} {row['customer_count']:.0f} {row['customer_pct']:.1f} "
                  f"{row['clv_pct']:.1f} {row['clv_efficiency']:.2f} "
                  f"{row['avg_clv']:.2f} {row['avg_recency']:.1f}")

        # 分层质量评估
        print(f'\n分层质量评估:\n')
        champions_pct = segment_stats.loc["Champions", "customer_pct"] if "Champions" in segment_stats.index else 0
        vip_pct = segment_stats.loc["VIP", "customer_pct"] if "VIP" in segment_stats.index else 0
        high_value_pct = champions_pct + vip_pct

        print(f'  - Champions占比: {champions_pct:.1f}% (目标: 2-5%)\n')
        print(f'  - VIP占比: {vip_pct:.1f}% (目标: 8-12%)\n')
        print(f'  - 高价值客户总占比: {high_value_pct:.1f}% (目标: 10-17%)\n')

        if "Champions" in segment_stats.index:
            champions_clv_contrib = segment_stats.loc["Champions", "clv_pct"]
            print(f'  - Champions CLV贡献: {champions_clv_contrib:.1f}% (期望: >30%)\n')

        return segment_stats

    def compare_with_original(self):
        """与原分层方法对比"""
        print("\n🔍 4. 与原分层方法对比")
        print("-" * 50)

        # 原分层统计
        original_stats = self.data.groupby("segment").agg({
            "customer_id": "count",
            "predicted_clv": ["sum", "mean"]
        }).round(2)
        original_stats.columns = ["orig_count", "orig_total_clv", "orig_avg_clv"]

        total_clv = self.data["predicted_clv"].sum()
        original_stats["orig_pct"] = (original_stats["orig_count"] / len(self.data) * 100).round(1)
        original_stats["orig_clv_pct"] = (original_stats["orig_total_clv"] / total_clv * 100).round(1)

        # 新分层统计
        new_stats = self.data.groupby("optimized_segment").agg({
            "customer_id": "count",
            "predicted_clv": ["sum", "mean"]
        }).round(2)
        new_stats.columns = ["new_count", "new_total_clv", "new_avg_clv"]
        new_stats["new_pct"] = (new_stats["new_count"] / len(self.data) * 100).round(1)
        new_stats["new_clv_pct"] = (new_stats["new_total_clv"] / total_clv * 100).round(1)

        print("原分层 vs 新分层对比:")
        print(f'\n原分层 (基于RFM):\n')
        for segment in original_stats.sort_values("orig_clv_pct", ascending=False).index:
            row = original_stats.loc[segment]
            print(f'  {segment}: {row["orig_count"]:.0f}人 ({row["orig_pct"]:.1f}%) - '
                  f'CLV贡献{row["orig_clv_pct"]:.1f}% - 平均CLV{row["orig_avg_clv"]:.2f}元\n')

        print(f'\n新分层 (优化逻辑):\n')
        for segment in new_stats.sort_values("new_clv_pct", ascending=False).index:
            row = new_stats.loc[segment]
            print(f'  {segment}: {row["new_count"]:.0f}人 ({row["new_pct"]:.1f}%) - '
                  f'CLV贡献{row["new_clv_pct"]:.1f}% - 平均CLV{row["new_avg_clv"]:.2f}元\n')

        return original_stats, new_stats

    def create_comprehensive_visualization(self):
        """创建综合可视化分析"""
        print("\n📊 5. 创建综合可视化分析")
        print("-" * 50)

        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle("优化客户分层综合分析", fontsize=18, fontweight="bold")

        # 1. CLV分布直方图
        ax1 = axes[0, 0]
        ax1.hist(self.data["predicted_clv"], bins=50, alpha=0.7, edgecolor="black", color="skyblue")
        ax1.set_xlabel("预测CLV (元)")
        ax1.set_ylabel("客户数量")
        ax1.set_title("CLV分布直方图")
        ax1.grid(True, alpha=0.3)

        # 2. CLV分布箱线图（对数尺度）
        ax2 = axes[0, 1]
        clv_positive = self.data[self.data["predicted_clv"] > 0]["predicted_clv"]
        ax2.boxplot(np.log10(clv_positive))
        ax2.set_ylabel("log10(CLV)")
        ax2.set_title("CLV分布箱线图（对数尺度）")
        ax2.grid(True, alpha=0.3)

        # 3. 原分层分布
        ax3 = axes[0, 2]
        original_counts = self.data["segment"].value_counts()
        colors_orig = plt.cm.Set3(np.linspace(0, 1, len(original_counts)))
        ax3.pie(original_counts.values, labels=original_counts.index, autopct="%1.1f%%", colors=colors_orig)
        ax3.set_title("原分层分布")

        # 4. 新分层分布
        ax4 = axes[0, 3]
        new_counts = self.data["optimized_segment"].value_counts()
        colors_new = plt.cm.Set2(np.linspace(0, 1, len(new_counts)))
        ax4.pie(new_counts.values, labels=new_counts.index, autopct="%1.1f%%", colors=colors_new)
        ax4.set_title("新分层分布")

        # 5. CLV vs Recency散点图（新分层）
        ax5 = axes[1, 0]
        segment_colors = {"Champions": "red", "VIP": "purple", "Loyal": "blue",
                          "Potential": "green", "At_Risk": "orange",
                          "Hibernating": "brown", "Lost": "gray"}

        for segment in self.data["optimized_segment"].unique():
            segment_data = self.data[self.data["optimized_segment"] == segment]
            ax5.scatter(segment_data["recency"], segment_data["predicted_clv"],
                        c=segment_colors.get(segment, "black"), label=segment, alpha=0.6, s=15)

        ax5.set_xlabel("Recency (天)")
        ax5.set_ylabel("预测CLV (元)")
        ax5.set_title("CLV vs Recency 散点图（新分层）")
        ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax5.grid(True, alpha=0.3)

        # 6. 分层CLV贡献对比
        ax6 = axes[1, 1]

        # 原分层CLV贡献
        orig_clv_contrib = self.data.groupby("segment")["predicted_clv"].sum().sort_values(ascending=False)
        orig_clv_pct = orig_clv_contrib / orig_clv_contrib.sum() * 100

        # 新分层CLV贡献
        new_clv_contrib = self.data.groupby("optimized_segment")["predicted_clv"].sum().sort_values(ascending=False)
        new_clv_pct = new_clv_contrib / new_clv_contrib.sum() * 100

        x = np.arange(max(len(orig_clv_pct), len(new_clv_pct)))
        width = 0.35

        # 补齐长度
        orig_values = list(orig_clv_pct.values) + [0] * (len(x) - len(orig_clv_pct))
        new_values = list(new_clv_pct.values) + [0] * (len(x) - len(new_clv_pct))

        ax6.bar(x - width / 2, orig_values[:len(x)], width, label="原分层", alpha=0.7)
        ax6.bar(x + width / 2, new_values[:len(x)], width, label="新分层", alpha=0.7)

        ax6.set_xlabel("分层排名")
        ax6.set_ylabel("CLV贡献占比 (%)")
        ax6.set_title("分层CLV贡献对比")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 各分层平均CLV对比
        ax7 = axes[1, 2]

        orig_avg_clv = self.data.groupby("segment")["predicted_clv"].mean().sort_values(ascending=False)
        new_avg_clv = self.data.groupby("optimized_segment")["predicted_clv"].mean().sort_values(ascending=False)

        # 取前5个分层进行对比
        top_n = 5
        orig_top = orig_avg_clv.head(top_n)
        new_top = new_avg_clv.head(top_n)

        x = np.arange(top_n)
        ax7.bar(x - 0.2, orig_top.values, 0.4, label="原分层", alpha=0.7)
        ax7.bar(x + 0.2, new_top.values, 0.4, label="新分层", alpha=0.7)

        ax7.set_xlabel("分层")
        ax7.set_ylabel("平均CLV (元)")
        ax7.set_title("各分层平均CLV对比（Top5）")
        ax7.set_xticks(x)
        ax7.set_xticklabels([f'Top{i + 1}' for i in range(top_n)])
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 分层效率分析
        ax8 = axes[1, 3]

        if hasattr(self, "segmentation_results"):
            efficiency_data = self.segmentation_results["clv_efficiency"].sort_values(ascending=False)
            efficiency_data.plot(kind="bar", ax=ax8, color="lightcoral")
            ax8.set_xlabel("分层")
            ax8.set_ylabel("CLV效率 (CLV% / 客户%)")
            ax8.set_title("各分层CLV效率")
            ax8.grid(True, alpha=0.3)

        # 9. Recency vs Frequency 散点图
        ax9 = axes[2, 0]
        for segment in self.data["optimized_segment"].unique():
            segment_data = self.data[self.data["optimized_segment"] == segment]
            ax9.scatter(segment_data["recency"], segment_data["frequency"],
                        c=segment_colors.get(segment, "black"), label=segment, alpha=0.6, s=15)
        ax9.set_xlabel("Recency (天)")
        ax9.set_ylabel("Frequency (次)")
        ax9.set_title("Recency vs Frequency 散点图（新分层）")
        ax9.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax9.grid(True, alpha=0.3)

        # 10. 各分层平均Recency对比
        ax10 = axes[2, 1]
        new_avg_recency = self.data.groupby("optimized_segment")["recency"].mean().sort_values(ascending=True)
        new_avg_recency.plot(kind="bar", ax=ax10, color="lightgreen")
        ax10.set_xlabel("分层")
        ax10.set_ylabel("平均Recency (天)")
        ax10.set_title("各分层平均Recency")
        ax10.grid(True, alpha=0.3)

        # 11. 各分层平均Frequency对比
        ax11 = axes[2, 2]
        new_avg_frequency = self.data.groupby("optimized_segment")["frequency"].mean().sort_values(ascending=False)
        new_avg_frequency.plot(kind="bar", ax=ax11, color="lightsalmon")
        ax11.set_xlabel("分层")
        ax11.set_ylabel("平均Frequency (次)")
        ax11.set_title("各分层平均Frequency")
        ax11.grid(True, alpha=0.3)

        # 12. 各分层平均Monetary对比
        ax12 = axes[2, 3]
        new_avg_monetary = self.data.groupby("optimized_segment")["monetary"].mean().sort_values(ascending=False)
        new_avg_monetary.plot(kind="bar", ax=ax12, color="lightsteelblue")
        ax12.set_xlabel("分层")
        ax12.set_ylabel("平均Monetary (元)")
        ax12.set_title("各分层平均Monetary")
        ax12.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig("/Users/changyu/Downloads/CLV/optimized_customer_segmentation.png")
        print("✅ 综合可视化图表已保存为 optimized_customer_segmentation.png")
        plt.close()

    def save_results(self):
        """保存分层结果和报告"""
        print("\n💾 6. 保存分层结果和报告")
        print("-" * 50)

        # 保存带有分层结果的CSV
        self.data.to_csv("/Users/changyu/Downloads/CLV/clv_based_segmentation_results.csv", index=False)
        print("✅ 带有分层结果的CSV已保存为 clv_based_segmentation_results.csv")

        # 保存分层统计报告
        report = {
            "segmentation_summary": self.segmentation_results.to_dict(orient="index"),
            "clv_thresholds": self.clv_thresholds
        }
        with open("/Users/changyu/Downloads/CLV/clv_based_segmentation_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        print("✅ 分层统计报告已保存为 clv_based_segmentation_report.json")


if __name__ == "__main__":
    segmentation = OptimizedCustomerSegmentation()
    if segmentation.load_data():
        segmentation.analyze_data_characteristics()
        segmentation.design_segmentation_logic()
        segmentation.analyze_segmentation_results()
        segmentation.compare_with_original()
        segmentation.create_comprehensive_visualization()
        segmentation.save_results()




