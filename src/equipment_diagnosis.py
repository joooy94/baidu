"""
投矾智慧控制系统 - 设备级智慧诊断模块
实现投加泵异常诊断、管道堵塞/泄漏诊断、阀门异常诊断、
储液池异常诊断、PLC通讯中断诊断
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    ALARM_LEVELS, PUMP_POINTS, SENSOR_POINTS,
)
from data_loader import load_pump_data, load_sensor_data, load_csv_data


@dataclass
class EquipmentDiagnosisResult:
    """设备诊断结果"""
    category: str            # 诊断类别（投加泵/投加管道/阀门/储液池/PLC通讯）
    target: str              # 诊断对象名称
    anomaly_type: str        # 异常类型
    alarm_level: int         # 报警等级 1/2/3
    alarm_level_name: str    # 报警等级名称
    start_time: datetime     # 异常开始时间
    end_time: datetime       # 异常结束时间
    duration_min: int        # 持续时间(分钟)
    detail: str              # 详细描述


# ==================== 1. 投加泵异常诊断 ====================

def diagnose_pump(
    pump_id: str,
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """
    投加泵异常诊断。

    检测项（按概要设计）：
    1. 泵故障信号（error=1） → 二类（轻微异常）
    2. 泵自动模式但无流量输出 → 三类（影响设备使用）
    3. 泵远控状态与自动状态不一致
    """
    results = []
    data = load_pump_data(pump_id, date_range=date_range)

    error_df = data.get("error", pd.DataFrame())
    flow_df = data.get("flow", pd.DataFrame())
    auto_df = data.get("auto", pd.DataFrame())
    remote_df = data.get("remote", pd.DataFrame())

    if error_df.empty and flow_df.empty:
        return results

    # --- 1.1 故障信号检测 ---
    if not error_df.empty:
        valid_error = error_df[error_df["is_valid"]]
        fault_mask = valid_error["value"] == 1
        if fault_mask.any():
            fault_df = valid_error[fault_mask].copy()
            fault_df["group"] = (fault_df.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in fault_df.groupby("group"):
                start = group["timestamp"].iloc[0]
                end = group["timestamp"].iloc[-1]
                dur = len(group)
                results.append(EquipmentDiagnosisResult(
                    category="投加泵异常诊断",
                    target=pump_id,
                    anomaly_type="pump_fault",
                    alarm_level=2,
                    alarm_level_name=ALARM_LEVELS[2],
                    start_time=start,
                    end_time=end,
                    duration_min=dur,
                    detail=f"{pump_id}故障信号持续{dur}分钟",
                ))

    # --- 1.2 自动模式无流量输出 ---
    if not flow_df.empty and not auto_df.empty:
        merged = pd.merge(
            flow_df[["timestamp", "value"]].rename(columns={"value": "flow"}),
            auto_df[["timestamp", "value"]].rename(columns={"value": "auto"}),
            on="timestamp", how="inner",
        )
        # 自动模式(auto=1)但流量为0
        merged["no_flow_in_auto"] = (merged["auto"] == 1) & (merged["flow"] <= 0)
        abnormal = merged[merged["no_flow_in_auto"]].copy()
        if not abnormal.empty:
            abnormal["group"] = (abnormal.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in abnormal.groupby("group"):
                if len(group) >= 5:  # 持续5分钟以上才报警
                    results.append(EquipmentDiagnosisResult(
                        category="投加泵异常诊断",
                        target=pump_id,
                        anomaly_type="no_flow_in_auto",
                        alarm_level=1,
                        alarm_level_name=ALARM_LEVELS[1],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{pump_id}自动模式运行但流量为0，持续{len(group)}分钟",
                    ))

    # --- 1.3 远控/自动状态不一致 ---
    if not auto_df.empty and not remote_df.empty:
        merged = pd.merge(
            auto_df[["timestamp", "value"]].rename(columns={"value": "auto"}),
            remote_df[["timestamp", "value"]].rename(columns={"value": "remote"}),
            on="timestamp", how="inner",
        )
        merged["inconsistent"] = merged["auto"] != merged["remote"]
        inconsistent = merged[merged["inconsistent"]].copy()
        if not inconsistent.empty:
            inconsistent["group"] = (inconsistent.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in inconsistent.groupby("group"):
                if len(group) >= 10:  # 持续10分钟
                    results.append(EquipmentDiagnosisResult(
                        category="投加泵异常诊断",
                        target=pump_id,
                        anomaly_type="state_inconsistent",
                        alarm_level=3,
                        alarm_level_name=ALARM_LEVELS[3],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{pump_id}自动/远控状态不一致，持续{len(group)}分钟",
                    ))

    return results


# ==================== 2. 投加管道堵塞/泄漏诊断 ====================

def diagnose_pipeline(
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """
    投加管道堵塞/泄漏诊断。

    设计文档逻辑：
    - 泵有流量输出但管道流量计读数低 → 堵塞
    - 储液池液位下降但泵未运行 → 泄漏
    - 泵流量与管道流量计偏差过大 → 堵塞或泄漏
    """
    results = []

    # 加载泵流量和管道流量计
    pump_flows = {}
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        data = load_pump_data(pid, date_range=date_range)
        flow_df = data.get("flow", pd.DataFrame())
        if not flow_df.empty:
            pump_flows[pid] = flow_df

    pipe_data = load_sensor_data("投加流量计", date_range=date_range)

    # --- 2.1 管道堵塞检测（泵有输出，管道流量计读数低） ---
    for pid, pump_flow_df in pump_flows.items():
        pump_idx = int(pid[1]) - 1  # P1->index 0
        pipe_name = list(pipe_data.keys())[pump_idx] if pump_idx < len(pipe_data) else None

        if pipe_name is None:
            continue

        pipe_df = pipe_data[pipe_name]

        # 取有效泵流量数据（泵有输出时）
        pump_valid = pump_flow_df[(pump_flow_df["is_valid"]) & (pump_flow_df["value"] > 0)]
        pipe_valid = pipe_df[pipe_df["is_valid"]]

        merged = pd.merge(
            pump_valid[["timestamp", "value"]].rename(columns={"value": "pump_flow"}),
            pipe_valid[["timestamp", "value"]].rename(columns={"value": "pipe_flow"}),
            on="timestamp", how="inner",
        )

        if merged.empty:
            continue

        # 泵有输出但管道流量接近0
        blocked = merged[merged["pipe_flow"] < merged["pump_flow"] * 0.1].copy()
        if not blocked.empty:
            blocked["group"] = (blocked.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in blocked.groupby("group"):
                if len(group) >= 5:
                    results.append(EquipmentDiagnosisResult(
                        category="投加管道异常诊断",
                        target=f"{pid}管道",
                        anomaly_type="pipe_blockage",
                        alarm_level=2,
                        alarm_level_name=ALARM_LEVELS[2],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{pid}泵有输出但管道流量接近0，疑似堵塞，持续{len(group)}分钟",
                    ))

    # --- 2.2 管道泄漏检测（储液池液位异常下降） ---
    tank_data = load_sensor_data("储液池液位计", date_range=date_range)
    for tank_name, tank_df in tank_data.items():
        valid = tank_df[tank_df["is_valid"]].copy()
        if len(valid) < 60:
            continue

        # 计算液位变化速率 (cm/min)
        valid["level_change"] = valid["value"].diff()
        # 液位快速下降（< -0.5cm/min 持续10分钟，非补液期间）
        valid["dropping"] = valid["level_change"] < -0.5
        drops = valid[valid["dropping"]].copy()
        if not drops.empty:
            drops["group"] = (drops.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in drops.groupby("group"):
                if len(group) >= 10:
                    total_drop = group["level_change"].sum()
                    results.append(EquipmentDiagnosisResult(
                        category="投加管道异常诊断",
                        target=tank_name,
                        anomaly_type="pipe_leak",
                        alarm_level=1,
                        alarm_level_name=ALARM_LEVELS[1],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{tank_name}液位异常下降{abs(total_drop):.1f}cm，疑似泄漏",
                    ))

    return results


# ==================== 3. 阀门异常诊断 ====================

def diagnose_valve(
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """
    阀门异常诊断（电动阀门）。

    检测项（按概要设计）：
    1. 故障汇总信号=1 → 阀门故障
    2. 既未开到位也未关到位 → 阀门卡死/不到位
    3. 同时开到位和关到位 → 信号矛盾
    """
    results = []
    valve_data = load_sensor_data("阀门", date_range=date_range)

    # 按阀门编号分组
    valve_groups = {
        "1号出液阀": {
            "close": "T投矾新_1号出液阀_关到位",
            "open": "T投矾新_1号出液阀_开到位",
            "error": "T投矾新_1号出液阀_故障汇总",
        },
        "2号出液阀": {
            "close": "T投矾新_2号出液阀_关到位",
            "open": "T投矾新_2号出液阀_开到位",
            "error": "T投矾新_2号出液阀_故障汇总",
        },
        "3号出液阀": {
            "close": "T投矾新_3号出液阀_关到位",
            "open": "T投矾新_3号出液阀_开到位",
            "error": "T投矾新_3号出液阀_故障汇总",
        },
    }

    for valve_name, signals in valve_groups.items():
        close_df = valve_data.get(signals["close"], pd.DataFrame())
        open_df = valve_data.get(signals["open"], pd.DataFrame())
        error_df = valve_data.get(signals["error"], pd.DataFrame())

        # --- 3.1 故障信号 ---
        if not error_df.empty:
            valid = error_df[error_df["is_valid"]]
            fault_mask = valid["value"] == 1
            if fault_mask.any():
                fault_df = valid[fault_mask].copy()
                fault_df["group"] = (fault_df.index.to_series().diff().fillna(1) != 1).cumsum()
                for _, group in fault_df.groupby("group"):
                    results.append(EquipmentDiagnosisResult(
                        category="阀门异常诊断",
                        target=valve_name,
                        anomaly_type="valve_fault",
                        alarm_level=2,
                        alarm_level_name=ALARM_LEVELS[2],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{valve_name}故障信号触发，持续{len(group)}分钟",
                    ))

        # --- 3.2 阀门卡死（既不开也不关）---
        if not close_df.empty and not open_df.empty:
            merged = pd.merge(
                close_df[["timestamp", "value"]].rename(columns={"value": "close"}),
                open_df[["timestamp", "value"]].rename(columns={"value": "open"}),
                on="timestamp", how="inner",
            )
            # 关到位=0且开到位=0 → 阀门处于中间位置
            stuck = merged[(merged["close"] == 0) & (merged["open"] == 0)].copy()
            if not stuck.empty:
                stuck["group"] = (stuck.index.to_series().diff().fillna(1) != 1).cumsum()
                for _, group in stuck.groupby("group"):
                    if len(group) >= 3:
                        results.append(EquipmentDiagnosisResult(
                            category="阀门异常诊断",
                            target=valve_name,
                            anomaly_type="valve_stuck",
                            alarm_level=3,
                            alarm_level_name=ALARM_LEVELS[3],
                            start_time=group["timestamp"].iloc[0],
                            end_time=group["timestamp"].iloc[-1],
                            duration_min=len(group),
                            detail=f"{valve_name}既未开到位也未关到位，持续{len(group)}分钟",
                        ))

            # --- 3.3 信号矛盾（同时开和关）---
            conflict = merged[(merged["close"] == 1) & (merged["open"] == 1)].copy()
            if not conflict.empty:
                conflict["group"] = (conflict.index.to_series().diff().fillna(1) != 1).cumsum()
                for _, group in conflict.groupby("group"):
                    results.append(EquipmentDiagnosisResult(
                        category="阀门异常诊断",
                        target=valve_name,
                        anomaly_type="valve_signal_conflict",
                        alarm_level=2,
                        alarm_level_name=ALARM_LEVELS[2],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{valve_name}开到位和关到位信号同时为1，信号矛盾",
                    ))

    return results


# ==================== 4. 储液池异常诊断 ====================

def diagnose_tank(
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """
    储液池异常诊断。

    检测项：
    1. 液位超上限/超下限 → 报警
    2. 液位变化速率异常（非补液期间快速变化）
    3. 液位异跳（配合传感器诊断中的异跳检测）
    """
    results = []
    tank_data = load_sensor_data("储液池液位计", date_range=date_range)

    # 液位阈值（单位：米，可根据现场调整）
    LEVEL_HIGH = 3.0   # 高液位报警线
    LEVEL_LOW = 0.3    # 低液位报警线
    LEVEL_CRITICAL_LOW = 0.1  # 极低液位

    for tank_name, df in tank_data.items():
        valid = df[df["is_valid"]].copy()
        if valid.empty:
            continue

        # --- 4.1 液位超限 ---
        # 高液位
        high = valid[valid["value"] > LEVEL_HIGH].copy()
        if not high.empty:
            high["group"] = (high.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in high.groupby("group"):
                if len(group) >= 5:
                    max_level = group["value"].max()
                    results.append(EquipmentDiagnosisResult(
                        category="储液池异常诊断",
                        target=tank_name,
                        anomaly_type="level_high",
                        alarm_level=2,
                        alarm_level_name=ALARM_LEVELS[2],
                        start_time=group["timestamp"].iloc[0],
                        end_time=group["timestamp"].iloc[-1],
                        duration_min=len(group),
                        detail=f"{tank_name}液位{max_level:.2f}m超过上限{LEVEL_HIGH}m",
                    ))

        # 低液位
        low = valid[valid["value"] < LEVEL_LOW].copy()
        if not low.empty:
            low["group"] = (low.index.to_series().diff().fillna(1) != 1).cumsum()
            for _, group in low.groupby("group"):
                min_level = group["value"].min()
                level = 1 if min_level < LEVEL_CRITICAL_LOW else 2
                results.append(EquipmentDiagnosisResult(
                    category="储液池异常诊断",
                    target=tank_name,
                    anomaly_type="level_low",
                    alarm_level=level,
                    alarm_level_name=ALARM_LEVELS[level],
                    start_time=group["timestamp"].iloc[0],
                    end_time=group["timestamp"].iloc[-1],
                    duration_min=len(group),
                    detail=f"{tank_name}液位{min_level:.2f}m低于下限{LEVEL_LOW}m",
                ))

        # --- 4.2 液位变化速率异常 ---
        if len(valid) >= 30:
            valid["change_rate"] = valid["value"].diff()  # cm/min
            # 快速上升（>0.3m/min，可能是异常补液）
            fast_rise = valid[valid["change_rate"] > 0.3].copy()
            if not fast_rise.empty:
                fast_rise["group"] = (fast_rise.index.to_series().diff().fillna(1) != 1).cumsum()
                for _, group in fast_rise.groupby("group"):
                    if len(group) >= 5:
                        results.append(EquipmentDiagnosisResult(
                            category="储液池异常诊断",
                            target=tank_name,
                            anomaly_type="level_fast_change",
                            alarm_level=3,
                            alarm_level_name=ALARM_LEVELS[3],
                            start_time=group["timestamp"].iloc[0],
                            end_time=group["timestamp"].iloc[-1],
                            duration_min=len(group),
                            detail=f"{tank_name}液位快速上升，速率>{0.3}m/min，持续{len(group)}分钟",
                        ))

    return results


# ==================== 5. PLC通讯中断诊断 ====================

def diagnose_plc_communication(
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """
    PLC通讯中断诊断。

    检测项：
    1. 所有数据同时缺失 → PLC通讯中断
    2. 单一数据源全部缺失 → 该子系统通讯中断

    按概要设计：
    - PLC与生产交互系统通讯超时>3次或无心跳信号 → 一级报警
    """
    results = []

    # 检查各数据源的数据完整性
    sources = {
        "ZHJY(投矾系统)": {
            "tagtable": "ZHJY",
            "tagindexes": [455, 459, 463],  # 取代表性点位
        },
        "ShuiZhi(水质系统)": {
            "tagtable": "ShuiZhi",
            "tagindexes": [84, 85],
        },
        "shuizhi(出水浊度)": {
            "tagtable": "shuizhi",
            "tagindexes": [4, 5],
        },
    }

    for source_name, cfg in sources.items():
        raw = load_csv_data(cfg["tagtable"], tagindexes=cfg["tagindexes"], date_range=date_range)
        if raw.empty:
            results.append(EquipmentDiagnosisResult(
                category="PLC通讯中断诊断",
                target=source_name,
                anomaly_type="comm_total_loss",
                alarm_level=1,
                alarm_level_name=ALARM_LEVELS[1],
                start_time=datetime.strptime(date_range[0], "%Y-%m-%d") if date_range else datetime.now(),
                end_time=datetime.strptime(date_range[1], "%Y-%m-%d") if date_range else datetime.now(),
                duration_min=0,
                detail=f"{source_name}在指定时间范围内无任何数据，疑似通讯完全中断",
            ))
            continue

        # 检查数据中断段（同一小时内全部缺失）
        from data_loader import mark_invalid_values
        processed = mark_invalid_values(raw)
        valid = processed[processed["is_valid"]]
        if valid.empty:
            results.append(EquipmentDiagnosisResult(
                category="PLC通讯中断诊断",
                target=source_name,
                anomaly_type="comm_total_loss",
                alarm_level=1,
                alarm_level_name=ALARM_LEVELS[1],
                start_time=datetime.strptime(date_range[0], "%Y-%m-%d") if date_range else datetime.now(),
                end_time=datetime.strptime(date_range[1], "%Y-%m-%d") if date_range else datetime.now(),
                duration_min=0,
                detail=f"{source_name}所有数据均为异常标记值，通讯可能中断",
            ))
            continue

        # 检查按小时的数据覆盖情况
        ts_range_start = processed["timestamp"].min()
        ts_range_end = processed["timestamp"].max()
        total_hours = int((ts_range_end - ts_range_start).total_seconds() / 3600) + 1

        valid_hours = valid["timestamp"].dt.floor("h").nunique()
        missing_hours = total_hours - valid_hours

        if missing_hours > total_hours * 0.1:  # 超过10%的小时缺失数据
            results.append(EquipmentDiagnosisResult(
                category="PLC通讯中断诊断",
                target=source_name,
                anomaly_type="comm_intermittent",
                alarm_level=2,
                alarm_level_name=ALARM_LEVELS[2],
                start_time=ts_range_start,
                end_time=ts_range_end,
                duration_min=missing_hours * 60,
                detail=f"{source_name}数据覆盖率为{valid_hours/total_hours*100:.1f}%，缺失{missing_hours}小时数据",
            ))

    return results


# ==================== 综合诊断入口 ====================

def diagnose_all_equipment(
    date_range: tuple[str, str] | None = None,
) -> list[EquipmentDiagnosisResult]:
    """执行全部设备级诊断"""
    all_results = []

    # 1. 投加泵诊断
    for pid in ["P1", "P2", "P3", "P4", "P5"]:
        all_results.extend(diagnose_pump(pid, date_range=date_range))

    # 2. 管道诊断
    all_results.extend(diagnose_pipeline(date_range=date_range))

    # 3. 阀门诊断
    all_results.extend(diagnose_valve(date_range=date_range))

    # 4. 储液池诊断
    all_results.extend(diagnose_tank(date_range=date_range))

    # 5. PLC通讯诊断
    all_results.extend(diagnose_plc_communication(date_range=date_range))

    # 按报警等级排序
    all_results.sort(key=lambda r: r.alarm_level)
    return all_results


def generate_equipment_diagnosis_report(
    results: list[EquipmentDiagnosisResult],
    title: str = "设备智慧诊断报告",
) -> str:
    """生成设备诊断文本报告"""
    lines = [f"{'='*70}", f"  {title}", f"{'='*70}", ""]

    if not results:
        lines.append("  未检测到设备异常。")
        return "\n".join(lines)

    # 按诊断类别分组
    by_category: dict[str, list] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)

    anomaly_type_names = {
        "pump_fault": "泵故障",
        "no_flow_in_auto": "自动模式无流量",
        "state_inconsistent": "状态不一致",
        "pipe_blockage": "管道堵塞",
        "pipe_leak": "管道泄漏",
        "valve_fault": "阀门故障",
        "valve_stuck": "阀门卡死",
        "valve_signal_conflict": "阀门信号矛盾",
        "level_high": "液位超高",
        "level_low": "液位超低",
        "level_fast_change": "液位变化异常",
        "comm_total_loss": "通讯完全中断",
        "comm_intermittent": "通讯间歇中断",
    }

    for cat, items in sorted(by_category.items()):
        lines.append(f"【{cat}】")
        # 按异常类型统计
        by_type: dict[str, list] = {}
        for item in items:
            if item.anomaly_type not in by_type:
                by_type[item.anomaly_type] = []
            by_type[item.anomaly_type].append(item)

        for atype, atype_items in sorted(by_type.items()):
            name = anomaly_type_names.get(atype, atype)
            lines.append(f"  {name}: {len(atype_items)} 条")
            for level in [1, 2, 3]:
                count = sum(1 for r in atype_items if r.alarm_level == level)
                if count > 0:
                    lines.append(f"    {ALARM_LEVELS[level]}报警: {count} 条")
        lines.append("")

    # 详细列表
    lines.append("-" * 70)
    lines.append("详细异常列表（前80条）:")
    lines.append("-" * 70)
    for i, r in enumerate(results[:80]):
        lines.append(
            f"  [{r.alarm_level_name}] {r.category}/{r.target} "
            f"| {r.anomaly_type} | {r.start_time} ~ {r.end_time} "
            f"| {r.detail}"
        )
    if len(results) > 80:
        lines.append(f"  ... 还有 {len(results) - 80} 条未显示")

    return "\n".join(lines)


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    date_range = ("2025-07-01", "2025-07-07")

    print("正在执行设备级智慧诊断...")
    all_results = diagnose_all_equipment(date_range=date_range)
    print(f"共发现 {len(all_results)} 条设备异常\n")

    report = generate_equipment_diagnosis_report(all_results)
    print(report)
