"""
投矾智慧控制系统 - 可视化输出模块
生成健康度雷达图、趋势折线图、诊断异常统计图等
"""

import os
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from datetime import datetime

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "STHeiti", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

from config import PUMP_POINTS, SENSOR_POINTS
from data_loader import load_pump_data, load_sensor_data
from sensor_diagnosis import diagnose_sensor, DiagnosisResult
from health_assessment import assess_pump_health, PumpHealthResult
from equipment_diagnosis import (
    diagnose_all_equipment, EquipmentDiagnosisResult,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _ensure_output_dir():
    """确保输出目录存在，不存在则自动创建"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================== 健康度可视化 ====================

def plot_health_radar(results: list[PumpHealthResult], save_path: str | None = None) -> str:
    """
    绘制投矾泵健康度雷达图（多泵对比）。

    Args:
        results: 多台泵的健康度评估结果
        save_path: 保存路径，None则自动生成

    Returns:
        保存的图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "health_radar.png")

    categories = ["流量偏差率", "故障频率", "运行稳定性", "控制响应性"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    for i, result in enumerate(results):
        values = [ind.score for ind in result.indicators]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=result.pump_id, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.set_title("投矾泵健康度雷达图", fontsize=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_health_bar(results: list[PumpHealthResult], save_path: str | None = None) -> str:
    """
    绘制投矾泵综合健康度柱状图。

    Args:
        results: 多台泵的健康度评估结果
        save_path: 保存路径

    Returns:
        图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "health_bar.png")

    pump_ids = [r.pump_id for r in results]
    scores = [r.overall_score for r in results]
    grades = [r.grade for r in results]

    # 按等级着色
    color_map = {"优秀": "#2ecc71", "良好": "#3498db", "轻微异常": "#f39c12", "异常": "#e74c3c"}
    colors = [color_map.get(g, "#95a5a6") for g in grades]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(pump_ids, scores, color=colors, edgecolor="white", linewidth=1.5, width=0.6)

    for bar, score, grade in zip(bars, scores, grades):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{score:.1f}\n({grade})", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_ylabel("健康度分数", fontsize=13)
    ax.set_xlabel("投矾泵编号", fontsize=13)
    ax.set_title("投矾泵综合健康度评估", fontsize=16)
    ax.axhline(y=90, color="#2ecc71", linestyle="--", alpha=0.5, label="优秀线(90)")
    ax.axhline(y=70, color="#3498db", linestyle="--", alpha=0.5, label="良好线(70)")
    ax.axhline(y=50, color="#f39c12", linestyle="--", alpha=0.5, label="异常线(50)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_health_trend(
    pump_id: str,
    date_range: tuple[str, str],
    window_days: int = 7,
    save_path: str | None = None,
) -> str:
    """
    绘制单台泵的健康度趋势折线图（按天滚动计算）。

    Args:
        pump_id: 泵编号
        date_range: 日期范围
        window_days: 滑动窗口天数
        save_path: 保存路径

    Returns:
        图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"health_trend_{pump_id}.png")

    # 按天计算健康度
    from datetime import timedelta
    start = datetime.strptime(date_range[0], "%Y-%m-%d")
    end = datetime.strptime(date_range[1], "%Y-%m-%d")

    dates = []
    scores = []
    current = start
    while current <= end:
        day_str = current.strftime("%Y-%m-%d")
        result = assess_pump_health(pump_id, date_range=(day_str, day_str))
        dates.append(current)
        scores.append(result.overall_score)
        current += timedelta(days=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, scores, "-o", color="#3498db", linewidth=1.5, markersize=3, label=pump_id)

    ax.fill_between(dates, scores, alpha=0.15, color="#3498db")
    ax.axhline(y=90, color="#2ecc71", linestyle="--", alpha=0.5)
    ax.axhline(y=70, color="#f39c12", linestyle="--", alpha=0.5)
    ax.axhline(y=50, color="#e74c3c", linestyle="--", alpha=0.5)

    ax.set_ylim(0, 105)
    ax.set_ylabel("健康度分数", fontsize=13)
    ax.set_xlabel("日期", fontsize=13)
    ax.set_title(f"{pump_id} 健康度趋势", fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ==================== 传感器诊断可视化 ====================

def plot_diagnosis_summary(
    results: list[DiagnosisResult],
    save_path: str | None = None,
) -> str:
    """
    绘制传感器诊断异常统计图（按类型和等级分组）。

    Args:
        results: 诊断结果列表
        save_path: 保存路径

    Returns:
        图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "diagnosis_summary.png")

    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "未检测到异常", ha="center", va="center", fontsize=20)
        ax.set_title("传感器诊断报告", fontsize=16)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    anomaly_names = {
        "data_absence": "数据缺失",
        "data_drift": "数据漂移",
        "data_jump": "数据异跳",
    }

    # 按传感器类型+异常类型分组
    df = pd.DataFrame([{
        "sensor_type": r.sensor_type,
        "anomaly_type": anomaly_names.get(r.anomaly_type, r.anomaly_type),
        "alarm_level": r.alarm_level,
        "alarm_name": r.alarm_level_name,
    } for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：按传感器类型的异常数量
    type_counts = df.groupby("sensor_type").size().sort_values(ascending=True)
    type_counts.plot.barh(ax=axes[0], color="#3498db", edgecolor="white")
    axes[0].set_xlabel("异常数量", fontsize=12)
    axes[0].set_title("各传感器类型异常数量", fontsize=14)
    axes[0].grid(axis="x", alpha=0.3)
    for i, (idx, val) in enumerate(type_counts.items()):
        axes[0].text(val + 5, i, str(val), va="center", fontsize=10)

    # 右图：按报警等级的异常数量（堆叠）
    level_colors = {1: "#e74c3c", 2: "#f39c12", 3: "#3498db"}
    pivot = df.groupby(["sensor_type", "alarm_name"]).size().unstack(fill_value=0)
    alarm_order = ["一级", "二级", "三级"]
    for a in alarm_order:
        if a not in pivot.columns:
            pivot[a] = 0
    pivot = pivot[alarm_order]
    pivot.plot.barh(ax=axes[1], stacked=True, color=[level_colors[1], level_colors[2], level_colors[3]],
                    edgecolor="white")
    axes[1].set_xlabel("异常数量", fontsize=12)
    axes[1].set_title("各传感器类型报警等级分布", fontsize=14)
    axes[1].legend(title="报警等级", fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("传感器智慧诊断报告", fontsize=18, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sensor_timeseries(
    sensor_type: str,
    sensor_name: str,
    date_range: tuple[str, str],
    results: list[DiagnosisResult] | None = None,
    save_path: str | None = None,
) -> str:
    """
    绘制传感器数据时序图，并标注异常点。

    Args:
        sensor_type: 传感器类型
        sensor_name: 传感器名称
        date_range: 日期范围
        results: 可选的诊断结果，用于标注异常区域
        save_path: 保存路径

    Returns:
        图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        safe_name = sensor_name.replace("#", "N").replace("/", "_")
        save_path = os.path.join(OUTPUT_DIR, f"sensor_{safe_name}.png")

    data = load_sensor_data(sensor_type, sensor_name=sensor_name, date_range=date_range)
    if sensor_name not in data:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, f"无数据: {sensor_name}", ha="center", va="center", fontsize=16)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    df = data[sensor_name]
    fig, ax = plt.subplots(figsize=(14, 5))

    # 绘制有效数据
    valid = df[df["is_valid"]]
    ax.plot(valid["timestamp"], valid["value"], "-", color="#2c3e50", linewidth=0.8, alpha=0.8, label="数据值")

    # 标注缺失区域
    missing = df[~df["is_valid"]]
    if not missing.empty:
        ax.scatter(missing["timestamp"], [ax.get_ylim()[0]] * len(missing),
                   color="#e74c3c", s=2, alpha=0.5, label="缺失数据")

    # 标注诊断异常
    if results:
        level_colors = {1: "#e74c3c", 2: "#f39c12", 3: "#95a5a6"}
        for r in results:
            if r.sensor_name == sensor_name:
                ax.axvspan(r.start_time, r.end_time,
                           alpha=0.15, color=level_colors.get(r.alarm_level, "#bdc3c7"))

    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("数值", fontsize=12)
    ax.set_title(f"{sensor_type} - {sensor_name}", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_pump_flow_comparison(
    date_range: tuple[str, str],
    save_path: str | None = None,
) -> str:
    """
    绘制各泵流量对比图。

    Args:
        date_range: 日期范围
        save_path: 保存路径

    Returns:
        图片路径
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "pump_flow_comparison.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]

    for i, pump_id in enumerate(["P1", "P2", "P3", "P4", "P5"]):
        data = load_pump_data(pump_id, date_range=date_range)
        flow_df = data.get("flow", pd.DataFrame())
        if flow_df.empty:
            continue
        valid = flow_df[flow_df["is_valid"] & (flow_df["value"] > 0)]
        if valid.empty:
            continue
        # 降采样：每60分钟取均值
        valid = valid.set_index("timestamp")
        resampled = valid["value"].resample("60min").mean()
        ax.plot(resampled.index, resampled.values, "-", linewidth=1, alpha=0.8,
                color=colors[i], label=f"{pump_id}")

    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("流量 (L/h)", fontsize=12)
    ax.set_title("各投矾泵实时流量对比", fontsize=16)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ==================== 设备级诊断可视化 ====================

def plot_equipment_diagnosis(
    results: list[EquipmentDiagnosisResult],
    save_path: str | None = None,
) -> str:
    """
    绘制设备级诊断统计图。

    左图：按诊断类别的异常数量（分报警等级堆叠）
    右图：按异常类型的分布饼图
    """
    _ensure_output_dir()
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "equipment_diagnosis.png")

    if not results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "未检测到设备异常", ha="center", va="center", fontsize=20)
        ax.set_title("设备智慧诊断报告", fontsize=16)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path

    anomaly_names = {
        "pump_fault": "泵故障",
        "no_flow_in_auto": "自动无流量",
        "state_inconsistent": "状态不一致",
        "pipe_blockage": "管道堵塞",
        "pipe_leak": "管道泄漏",
        "valve_fault": "阀门故障",
        "valve_stuck": "阀门卡死",
        "valve_signal_conflict": "信号矛盾",
        "level_high": "液位超高",
        "level_low": "液位超低",
        "level_fast_change": "液位变化异常",
        "comm_total_loss": "通讯完全中断",
        "comm_intermittent": "通讯间歇中断",
    }

    df = pd.DataFrame([{
        "category": r.category,
        "anomaly_type": anomaly_names.get(r.anomaly_type, r.anomaly_type),
        "alarm_level": r.alarm_level,
        "alarm_name": r.alarm_level_name,
        "target": r.target,
    } for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左图：按诊断类别堆叠
    level_colors = {1: "#e74c3c", 2: "#f39c12", 3: "#3498db"}
    pivot = df.groupby(["category", "alarm_name"]).size().unstack(fill_value=0)
    alarm_order = ["一级", "二级", "三级"]
    for a in alarm_order:
        if a not in pivot.columns:
            pivot[a] = 0
    pivot = pivot[alarm_order]
    pivot.plot.barh(ax=axes[0], stacked=True,
                    color=[level_colors[1], level_colors[2], level_colors[3]],
                    edgecolor="white")
    axes[0].set_xlabel("异常数量", fontsize=12)
    axes[0].set_title("各诊断类别报警等级分布", fontsize=14)
    axes[0].legend(title="报警等级", fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    # 右图：按异常类型分布（饼图，取前10）
    type_counts = df["anomaly_type"].value_counts()
    top_types = type_counts.head(10)
    if len(type_counts) > 10:
        other_count = type_counts.iloc[10:].sum()
        top_types = pd.concat([top_types, pd.Series({"其他": other_count})])

    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(top_types)))
    axes[1].pie(top_types.values, labels=top_types.index, autopct="%1.1f%%",
                colors=pie_colors, startangle=90)
    axes[1].set_title(f"异常类型分布 (共{len(results)}条)", fontsize=14)

    plt.suptitle("设备智慧诊断统计", fontsize=18, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ==================== 一键生成全部图表 ====================

def generate_all_charts(date_range: tuple[str, str] | None = None) -> dict[str, str]:
    """
    一键生成所有可视化图表。

    Args:
        date_range: 日期范围，默认最近一个月

    Returns:
        {图表名称: 文件路径} 字典
    """
    if date_range is None:
        date_range = ("2025-07-01", "2025-07-31")

    charts = {}
    print("正在生成可视化图表...")

    # 1. 健康度雷达图
    print("  1. 健康度雷达图...")
    pump_results = [assess_pump_health(pid, date_range=date_range) for pid in ["P1", "P2", "P3", "P4", "P5"]]
    charts["health_radar"] = plot_health_radar(pump_results)
    print(f"     -> {charts['health_radar']}")

    # 2. 健康度柱状图
    print("  2. 健康度柱状图...")
    charts["health_bar"] = plot_health_bar(pump_results)
    print(f"     -> {charts['health_bar']}")

    # 3. 各泵健康度趋势
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        print(f"  3. {pid} 健康度趋势...")
        charts[f"health_trend_{pid}"] = plot_health_trend(pid, date_range)
        print(f"     -> {charts[f'health_trend_{pid}']}")

    # 4. 传感器诊断统计
    print("  4. 传感器诊断统计图...")
    all_diag = []
    for st in ["出水浊度仪", "进水浊度仪", "储液池液位计"]:
        all_diag.extend(diagnose_sensor(st, date_range=date_range))
    charts["diagnosis_summary"] = plot_diagnosis_summary(all_diag)
    print(f"     -> {charts['diagnosis_summary']}")

    # 5. 传感器时序图（选代表性传感器）
    print("  5. 传感器时序图...")
    sample_sensors = [
        ("出水浊度仪", "沉淀池出水_1#浊度"),
        ("储液池液位计", "T投矾新1号储液池液位"),
    ]
    for stype, sname in sample_sensors:
        safe_name = sname.replace("#", "N").replace("/", "_")
        diag = [r for r in all_diag if r.sensor_name == sname]
        path = plot_sensor_timeseries(stype, sname, date_range, diag)
        charts[f"sensor_{safe_name}"] = path
        print(f"     -> {path}")

    # 6. 泵流量对比
    print("  6. 泵流量对比图...")
    charts["pump_flow"] = plot_pump_flow_comparison(date_range)
    print(f"     -> {charts['pump_flow']}")

    # 7. 设备级诊断统计图
    print("  7. 设备智慧诊断统计图...")
    eq_results = diagnose_all_equipment(date_range=date_range)
    charts["equipment_diagnosis"] = plot_equipment_diagnosis(eq_results)
    print(f"     -> {charts['equipment_diagnosis']}")

    print(f"\n共生成 {len(charts)} 张图表，保存在 {OUTPUT_DIR}/ 目录下")
    return charts


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    import sys
    date_range = ("2025-07-01", "2025-07-07")  # 默认一周
    if len(sys.argv) >= 3:
        date_range = (sys.argv[1], sys.argv[2])

    charts = generate_all_charts(date_range)
    for name, path in charts.items():
        print(f"  {name}: {path}")
