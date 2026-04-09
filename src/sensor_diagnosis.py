"""
投矾智慧控制系统 - 传感器诊断模块
实现传感器数据缺失、漂移、异跳检测，分级报警
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ABSENCE_THRESHOLDS, DRIFT_THRESHOLDS, JUMP_THRESHOLDS,
    ALARM_LEVELS, SENSOR_POINTS,
)
from data_loader import load_sensor_data


@dataclass
class DiagnosisResult:
    """单条诊断结果"""
    sensor_type: str          # 传感器类型
    sensor_name: str          # 传感器名称
    anomaly_type: str         # 异常类型: data_absence, data_drift, data_jump
    alarm_level: int          # 报警等级: 1(最高), 2, 3(最低)
    alarm_level_name: str     # 报警等级名称
    start_time: datetime      # 异常开始时间
    end_time: datetime        # 异常结束时间
    duration_min: int         # 持续时间(分钟)
    detail: str               # 详细描述
    threshold: float = 0.0    # 触发阈值
    actual_value: float = 0.0 # 实际检测值


def detect_data_absence(
    df: pd.DataFrame,
    sensor_type: str,
    sensor_name: str,
) -> list[DiagnosisResult]:
    """
    检测数据缺失异常。

    判定逻辑（按概要设计）：
    - 连续10min无有效数据 → 三级报警
    - 连续30min无有效数据 → 二级报警
    - 连续60min无有效数据 → 一级报警
    """
    results = []
    if df.empty:
        return results

    df = df.sort_values("timestamp").copy()
    df["is_missing"] = ~df["is_valid"]

    # 找出连续缺失的段
    df["group"] = (df["is_missing"] != df["is_missing"].shift()).cumsum()
    missing_groups = df[df["is_missing"]].groupby("group")

    for _, group in missing_groups:
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]
        duration = len(group)

        # 确定报警等级（取最高等级，从一级开始检查）
        alarm_level = 4  # 未触发
        for level in [1, 2, 3]:
            if duration >= ABSENCE_THRESHOLDS[level]:
                alarm_level = level
                break

        if alarm_level <= 3:
            results.append(DiagnosisResult(
                sensor_type=sensor_type,
                sensor_name=sensor_name,
                anomaly_type="data_absence",
                alarm_level=alarm_level,
                alarm_level_name=ALARM_LEVELS[alarm_level],
                start_time=start,
                end_time=end,
                duration_min=duration,
                detail=f"连续{duration}分钟无有效数据",
                threshold=ABSENCE_THRESHOLDS[alarm_level],
                actual_value=float(duration),
            ))

    return results


def detect_data_drift(
    df: pd.DataFrame,
    sensor_type: str,
    sensor_name: str,
) -> list[DiagnosisResult]:
    """
    检测数据漂移/偏差异常。

    判定逻辑（按概要设计和传感器类型分）：
    - 电磁流量计: ±20%/±40%/±50% 对应 三级/二级/一级
    - 浊度仪: ±15%/±40%/±50% 对应 三级/二级/一级
    - 液位计: 绝对偏差>10cm → 二级报警
    """
    results = []
    thresholds = DRIFT_THRESHOLDS.get(sensor_type)
    if thresholds is None or df.empty:
        return results

    df = df.sort_values("timestamp").copy()
    valid_df = df[df["is_valid"]].copy()

    if len(valid_df) < 30:
        return results

    # 计算历史均值（使用前30分钟滑动窗口）
    valid_df["hist_mean"] = valid_df["value"].rolling(window=30, min_periods=10).mean()

    for level in [3, 2, 1]:
        cfg = thresholds.get(level)
        if cfg is None:
            continue

        duration_min = cfg["duration_min"]
        is_abs = "threshold_abs_cm" in cfg
        threshold = cfg.get("threshold", None)

        # 计算偏差
        if is_abs:
            # 液位计用绝对偏差
            abs_threshold = cfg["threshold_abs_cm"]
            valid_df["deviation"] = (valid_df["value"] - valid_df["hist_mean"]).abs()
            exceeded = valid_df[valid_df["deviation"] > abs_threshold].copy()
        else:
            # 其他用百分比偏差
            valid_df["deviation_pct"] = (
                (valid_df["value"] - valid_df["hist_mean"]) / valid_df["hist_mean"].replace(0, np.nan)
            ).abs()
            exceeded = valid_df[valid_df["deviation_pct"] > threshold].copy()

        if exceeded.empty:
            continue

        # 找连续超限段
        exceeded["idx_diff"] = exceeded.index.to_series().diff().fillna(1)
        exceeded["group"] = (exceeded["idx_diff"] != 1).cumsum()

        for _, group in exceeded.groupby("group"):
            if len(group) < duration_min:
                continue

            start = group["timestamp"].iloc[0]
            end = group["timestamp"].iloc[-1]
            dur = len(group)

            if is_abs:
                actual = group["deviation"].max()
                detail = f"液位偏差{actual:.1f}cm超过阈值{abs_threshold}cm，持续{dur}分钟"
            else:
                actual = group["deviation_pct"].max()
                detail = f"数据偏差{actual*100:.1f}%超过阈值{threshold*100:.0f}%，持续{dur}分钟(要求≥{duration_min}min)"

            results.append(DiagnosisResult(
                sensor_type=sensor_type,
                sensor_name=sensor_name,
                anomaly_type="data_drift",
                alarm_level=level,
                alarm_level_name=ALARM_LEVELS[level],
                start_time=start,
                end_time=end,
                duration_min=dur,
                detail=detail,
                threshold=threshold if not is_abs else abs_threshold,
                actual_value=float(actual),
            ))

    return results


def detect_data_jump(
    df: pd.DataFrame,
    sensor_type: str,
    sensor_name: str,
) -> list[DiagnosisResult]:
    """
    检测数据异跳异常。

    判定逻辑：
    - 出水浊度仪: 变化率>50NTU/min
    - 进水浊度仪/电磁流量计: >正常波动3倍
    - 液位计: >30cm/10min
    """
    results = []
    jump_cfg = JUMP_THRESHOLDS.get(sensor_type)
    if jump_cfg is None or df.empty:
        return results

    df = df.sort_values("timestamp").copy()
    valid_df = df[df["is_valid"]].copy()

    if len(valid_df) < 2:
        return results

    # 计算变化率（每分钟）
    valid_df["rate"] = valid_df["value"].diff().abs()

    if "rate_per_min" in jump_cfg:
        # 固定阈值模式（如出水浊度仪 >50 NTU/min）
        threshold = jump_cfg["rate_per_min"]
        # 检查连续3个采样周期
        exceeded = valid_df[valid_df["rate"] > threshold].copy()
        if not exceeded.empty:
            exceeded["idx_diff"] = exceeded.index.to_series().diff().fillna(1)
            exceeded["group"] = (exceeded["idx_diff"] != 1).cumsum()
            for _, group in exceeded.groupby("group"):
                if len(group) >= 3:
                    start = group["timestamp"].iloc[0]
                    end = group["timestamp"].iloc[-1]
                    max_rate = group["rate"].max()
                    results.append(DiagnosisResult(
                        sensor_type=sensor_type,
                        sensor_name=sensor_name,
                        anomaly_type="data_jump",
                        alarm_level=3,
                        alarm_level_name=ALARM_LEVELS[3],
                        start_time=start,
                        end_time=end,
                        duration_min=len(group),
                        detail=f"变化率{max_rate:.2f}超过阈值{threshold}/min，连续{len(group)}个采样周期",
                        threshold=threshold,
                        actual_value=float(max_rate),
                    ))

    elif "normal_multiple" in jump_cfg:
        # 正常波动倍数模式（如进水浊度仪、电磁流量计）
        normal_std = valid_df["rate"].std()
        if np.isnan(normal_std) or normal_std == 0:
            return results
        threshold = normal_std * jump_cfg["normal_multiple"]
        exceeded = valid_df[valid_df["rate"] > threshold].copy()
        if not exceeded.empty:
            for _, row in exceeded.iterrows():
                results.append(DiagnosisResult(
                    sensor_type=sensor_type,
                    sensor_name=sensor_name,
                    anomaly_type="data_jump",
                    alarm_level=3,
                    alarm_level_name=ALARM_LEVELS[3],
                    start_time=row["timestamp"],
                    end_time=row["timestamp"],
                    duration_min=1,
                    detail=f"变化率{row['rate']:.2f}超过正常波动{jump_cfg['normal_multiple']}倍(阈值{threshold:.2f})",
                    threshold=float(threshold),
                    actual_value=float(row["rate"]),
                ))

    elif "change_cm_per_10min" in jump_cfg:
        # 液位计模式：>30cm/10min
        threshold = jump_cfg["change_cm_per_10min"]
        # 计算10分钟窗口内的变化
        valid_df["change_10min"] = (valid_df["value"] - valid_df["value"].shift(10)).abs()
        exceeded = valid_df[valid_df["change_10min"] > threshold].copy()
        if not exceeded.empty:
            for _, row in exceeded.iterrows():
                results.append(DiagnosisResult(
                    sensor_type=sensor_type,
                    sensor_name=sensor_name,
                    anomaly_type="data_jump",
                    alarm_level=3,
                    alarm_level_name=ALARM_LEVELS[3],
                    start_time=row["timestamp"],
                    end_time=row["timestamp"],
                    duration_min=10,
                    detail=f"10分钟内液位变化{row['change_10min']:.1f}cm超过阈值{threshold}cm",
                    threshold=float(threshold),
                    actual_value=float(row["change_10min"]),
                ))

    return results


def diagnose_sensor(
    sensor_type: str,
    sensor_name: str | None = None,
    date_range: tuple[str, str] | None = None,
) -> list[DiagnosisResult]:
    """
    对指定类型传感器执行完整诊断（缺失+漂移+异跳）。

    Args:
        sensor_type: 传感器类型
        sensor_name: 传感器名称（None则诊断全部）
        date_range: 日期范围

    Returns:
        诊断结果列表
    """
    data = load_sensor_data(sensor_type, sensor_name=sensor_name, date_range=date_range)
    all_results = []

    for name, df in data.items():
        # 数据缺失检测
        all_results.extend(detect_data_absence(df, sensor_type, name))
        # 数据漂移检测
        all_results.extend(detect_data_drift(df, sensor_type, name))
        # 数据异跳检测
        all_results.extend(detect_data_jump(df, sensor_type, name))

    # 按报警等级排序（高优先级在前）
    all_results.sort(key=lambda r: r.alarm_level)
    return all_results


def generate_diagnosis_report(
    results: list[DiagnosisResult],
    title: str = "传感器诊断报告",
) -> str:
    """生成文本格式的诊断报告"""
    lines = [f"{'='*70}", f"  {title}", f"{'='*70}", ""]

    if not results:
        lines.append("  未检测到异常。")
        return "\n".join(lines)

    # 按传感器类型和异常类型分组统计
    by_type: dict[str, dict[str, list]] = {}
    for r in results:
        if r.sensor_type not in by_type:
            by_type[r.sensor_type] = {}
        if r.anomaly_type not in by_type[r.sensor_type]:
            by_type[r.sensor_type][r.anomaly_type] = []
        by_type[r.sensor_type][r.anomaly_type].append(r)

    anomaly_names = {
        "data_absence": "数据缺失",
        "data_drift": "数据漂移/偏差",
        "data_jump": "数据异跳",
    }

    for stype, anomalies in sorted(by_type.items()):
        lines.append(f"【{stype}】")
        for atype, items in sorted(anomalies.items()):
            lines.append(f"  {anomaly_names.get(atype, atype)}: {len(items)} 条")
            # 按报警等级统计
            for level in [1, 2, 3]:
                count = sum(1 for r in items if r.alarm_level == level)
                if count > 0:
                    lines.append(f"    {ALARM_LEVELS[level]}报警: {count} 条")
        lines.append("")

    # 详细列表（只显示前50条）
    lines.append("-" * 70)
    lines.append("详细异常列表（前50条）:")
    lines.append("-" * 70)
    for i, r in enumerate(results[:50]):
        lines.append(
            f"  [{r.alarm_level_name}] {r.sensor_type}/{r.sensor_name} "
            f"| {r.anomaly_type} | {r.start_time} ~ {r.end_time} "
            f"| {r.detail}"
        )

    if len(results) > 50:
        lines.append(f"  ... 还有 {len(results) - 50} 条未显示")

    return "\n".join(lines)


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    # 测试诊断（使用2025年7月数据）
    date_range = ("2025-07-01", "2025-07-31")

    all_results = []
    for sensor_type in ["出水浊度仪", "进水浊度仪", "储液池液位计", "电磁流量计"]:
        print(f"\n正在诊断 {sensor_type}...")
        results = diagnose_sensor(sensor_type, date_range=date_range)
        all_results.extend(results)
        print(f"  发现 {len(results)} 条异常")

    report = generate_diagnosis_report(all_results)
    print(report)
