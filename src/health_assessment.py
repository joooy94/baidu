"""
投矾智慧控制系统 - 投加设备健康度评估模块
基于流量偏差、故障频率、运行稳定性、控制响应性四维指标评估投矾泵健康度
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import PUMP_POINTS, HEALTH_WEIGHTS, HEALTH_GRADES, ALARM_LEVELS
from data_loader import load_pump_data


@dataclass
class HealthIndicator:
    """单个健康指标"""
    name: str              # 指标名称
    value: float           # 原始值
    score: float           # 归一化分数 (0~100)
    weight: float          # 权重
    detail: str            # 说明


@dataclass
class PumpHealthResult:
    """单台泵的健康度评估结果"""
    pump_id: str
    overall_score: float              # 综合健康度 (0~100)
    grade: str                        # 等级描述
    indicators: list[HealthIndicator] # 各维度指标
    timestamp_range: tuple            # 评估数据时间范围
    warnings: list[str]               # 告警信息


def _calc_flow_deviation(flow_df: pd.DataFrame) -> HealthIndicator:
    """
    计算流量偏差率指标。

    定义：|实际流量 - 设定流量| / 设定流量的滑动平均
    使用滑动窗口均值作为"设定/期望流量"，计算偏差比例。

    评分：
    - 偏差率 < 5%  → 100分
    - 偏差率 5~10% → 80~100分
    - 偏差率 10~20% → 60~80分
    - 偏差率 20~50% → 30~60分
    - 偏差率 > 50%  → 0~30分
    """
    if flow_df.empty:
        return HealthIndicator("流量偏差率", 0, 50, HEALTH_WEIGHTS["flow_deviation"], "无数据")

    valid = flow_df[flow_df["is_valid"]]["value"].dropna()
    if len(valid) < 30:
        return HealthIndicator("流量偏差率", 0, 50, HEALTH_WEIGHTS["flow_deviation"], "数据不足")

    # 用滑动窗口均值作为期望值
    rolling_mean = valid.rolling(window=30, min_periods=10).mean()
    deviation_rate = (valid - rolling_mean).abs() / rolling_mean.replace(0, np.nan)
    deviation_rate = deviation_rate.dropna()

    if deviation_rate.empty:
        return HealthIndicator("流量偏差率", 0, 50, HEALTH_WEIGHTS["flow_deviation"], "无法计算偏差")

    avg_deviation = deviation_rate.mean()

    # 评分映射
    if avg_deviation < 0.05:
        score = 100
    elif avg_deviation < 0.10:
        score = 100 - (avg_deviation - 0.05) / 0.05 * 20
    elif avg_deviation < 0.20:
        score = 80 - (avg_deviation - 0.10) / 0.10 * 20
    elif avg_deviation < 0.50:
        score = 60 - (avg_deviation - 0.20) / 0.30 * 30
    else:
        score = max(0, 30 - (avg_deviation - 0.50) / 0.50 * 30)

    return HealthIndicator(
        name="流量偏差率",
        value=float(avg_deviation),
        score=float(score),
        weight=HEALTH_WEIGHTS["flow_deviation"],
        detail=f"平均偏差率: {avg_deviation*100:.1f}%",
    )


def _calc_fault_frequency(error_df: pd.DataFrame, total_hours: float) -> HealthIndicator:
    """
    计算故障频率指标。

    定义：单位时间内故障(error=1)出现频次
    评分：
    - 0次/天   → 100分
    - <1次/天  → 80~100分
    - 1~5次/天 → 50~80分
    - >5次/天  → 0~50分
    """
    if error_df.empty or total_hours <= 0:
        return HealthIndicator("故障频率", 0, 50, HEALTH_WEIGHTS["fault_frequency"], "无数据")

    valid = error_df[error_df["is_valid"]]
    fault_count = (valid["value"] == 1).sum()
    fault_per_day = fault_count / (total_hours / 24) if total_hours > 0 else 0

    # 评分映射
    if fault_per_day == 0:
        score = 100
    elif fault_per_day < 1:
        score = 100 - fault_per_day * 20
    elif fault_per_day < 5:
        score = 80 - (fault_per_day - 1) / 4 * 30
    else:
        score = max(0, 50 - (fault_per_day - 5) / 10 * 50)

    return HealthIndicator(
        name="故障频率",
        value=float(fault_per_day),
        score=float(score),
        weight=HEALTH_WEIGHTS["fault_frequency"],
        detail=f"故障{fault_count}次，频率{fault_per_day:.2f}次/天",
    )


def _calc_stability(flow_df: pd.DataFrame) -> HealthIndicator:
    """
    计算运行稳定性指标。

    定义：流量的变异系数 (CV = std/mean)
    评分：
    - CV < 0.05  → 100分
    - CV 0.05~0.1 → 80~100分
    - CV 0.1~0.3  → 50~80分
    - CV > 0.3    → 0~50分
    """
    if flow_df.empty:
        return HealthIndicator("运行稳定性", 0, 50, HEALTH_WEIGHTS["stability"], "无数据")

    # 只取运行中（流量>0）的有效数据
    valid = flow_df[(flow_df["is_valid"]) & (flow_df["value"] > 0)]["value"].dropna()
    if len(valid) < 30:
        return HealthIndicator("运行稳定性", 0, 50, HEALTH_WEIGHTS["stability"], "有效运行数据不足")

    mean_val = valid.mean()
    std_val = valid.std()
    cv = std_val / mean_val if mean_val > 0 else 1.0

    # 评分映射
    if cv < 0.05:
        score = 100
    elif cv < 0.10:
        score = 100 - (cv - 0.05) / 0.05 * 20
    elif cv < 0.30:
        score = 80 - (cv - 0.10) / 0.20 * 30
    else:
        score = max(0, 50 - (cv - 0.30) / 0.70 * 50)

    return HealthIndicator(
        name="运行稳定性",
        value=float(cv),
        score=float(score),
        weight=HEALTH_WEIGHTS["stability"],
        detail=f"变异系数(CV): {cv:.4f}",
    )


def _calc_responsiveness(auto_df: pd.DataFrame, remote_df: pd.DataFrame) -> HealthIndicator:
    """
    计算控制响应性指标。

    定义：自动/远控状态的一致性
    - 检查自动(auto=1)和远控(remote=1)状态是否一致
    - 运行时 auto 和 remote 应同时为1
    - 一致率越高越好

    评分：
    - 一致率 > 95% → 100分
    - 一致率 80~95% → 70~100分
    - 一致率 < 80% → 0~70分
    """
    if auto_df.empty or remote_df.empty:
        return HealthIndicator("控制响应性", 0, 50, HEALTH_WEIGHTS["responsiveness"], "无数据")

    # 合并 auto 和 remote 数据
    merged = pd.merge(
        auto_df[["timestamp", "value"]].rename(columns={"value": "auto"}),
        remote_df[["timestamp", "value"]].rename(columns={"value": "remote"}),
        on="timestamp",
        how="inner",
    )

    if merged.empty:
        return HealthIndicator("控制响应性", 0, 50, HEALTH_WEIGHTS["responsiveness"], "无匹配数据")

    # 计算一致性：auto 和 remote 应该相同
    merged["consistent"] = merged["auto"] == merged["remote"]
    consistency_rate = merged["consistent"].mean()

    # 评分映射
    if consistency_rate > 0.95:
        score = 100
    elif consistency_rate > 0.80:
        score = 70 + (consistency_rate - 0.80) / 0.15 * 30
    else:
        score = max(0, consistency_rate / 0.80 * 70)

    return HealthIndicator(
        name="控制响应性",
        value=float(consistency_rate),
        score=float(score),
        weight=HEALTH_WEIGHTS["responsiveness"],
        detail=f"状态一致率: {consistency_rate*100:.1f}%",
    )


def assess_pump_health(
    pump_id: str,
    date_range: tuple[str, str] | None = None,
) -> PumpHealthResult:
    """
    评估单台投矾泵的健康度。

    Args:
        pump_id: 泵编号，如 "P1"
        date_range: 日期范围

    Returns:
        PumpHealthResult 包含综合评分和各维度指标
    """
    data = load_pump_data(pump_id, date_range=date_range)

    flow_df = data.get("flow", pd.DataFrame())
    error_df = data.get("error", pd.DataFrame())
    auto_df = data.get("auto", pd.DataFrame())
    remote_df = data.get("remote", pd.DataFrame())

    # 计算时间跨度
    all_dfs = [df for df in [flow_df, error_df] if not df.empty]
    if all_dfs:
        all_ts = pd.concat([df["timestamp"] for df in all_dfs])
        ts_min = all_ts.min()
        ts_max = all_ts.max()
        total_hours = (ts_max - ts_min).total_seconds() / 3600
    else:
        ts_min = ts_max = pd.Timestamp.now()
        total_hours = 0

    # 计算各指标
    indicators = [
        _calc_flow_deviation(flow_df),
        _calc_fault_frequency(error_df, total_hours),
        _calc_stability(flow_df),
        _calc_responsiveness(auto_df, remote_df),
    ]

    # 加权求和
    overall_score = sum(ind.score * ind.weight for ind in indicators)

    # 确定等级
    grade = "未知"
    for (low, high), g in HEALTH_GRADES.items():
        if low <= overall_score < high:
            grade = g
            break

    # 生成告警
    warnings = []
    for ind in indicators:
        if ind.score < 50:
            warnings.append(f"[严重] {ind.name}: {ind.detail}")
        elif ind.score < 70:
            warnings.append(f"[警告] {ind.name}: {ind.detail}")

    return PumpHealthResult(
        pump_id=pump_id,
        overall_score=round(overall_score, 1),
        grade=grade,
        indicators=indicators,
        timestamp_range=(ts_min, ts_max),
        warnings=warnings,
    )


def generate_health_report(
    results: list[PumpHealthResult],
    title: str = "投加设备健康度评估报告",
) -> str:
    """生成文本格式的健康度评估报告"""
    lines = [f"{'='*70}", f"  {title}", f"{'='*70}", ""]

    for r in results:
        lines.append(f"【{r.pump_id}】 综合健康度: {r.overall_score:.1f}/100 ({r.grade})")
        lines.append(f"  评估时间范围: {r.timestamp_range[0]} ~ {r.timestamp_range[1]}")
        for ind in r.indicators:
            lines.append(
                f"  - {ind.name}: {ind.score:.1f}分 (权重{ind.weight*100:.0f}%) | {ind.detail}"
            )
        if r.warnings:
            lines.append("  告警:")
            for w in r.warnings:
                lines.append(f"    {w}")
        lines.append("")

    # 汇总
    lines.append("-" * 70)
    lines.append("汇总:")
    for r in results:
        status_icon = "+" if r.overall_score >= 70 else "!"
        lines.append(f"  [{status_icon}] {r.pump_id}: {r.overall_score:.1f} ({r.grade})")

    return "\n".join(lines)


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    date_range = ("2025-07-01", "2025-07-31")

    results = []
    for pump_id in ["P1", "P2", "P3", "P4", "P5"]:
        print(f"正在评估 {pump_id}...")
        result = assess_pump_health(pump_id, date_range=date_range)
        results.append(result)

    report = generate_health_report(results)
    print(report)
