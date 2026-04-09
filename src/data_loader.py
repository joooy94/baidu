"""
投矾智慧控制系统 - 数据加载与预处理
将CSV中的T01~T60展开为分钟级时间序列
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import (
    EXCEL_PATH, CSV_FILES, SENSOR_POINTS, PUMP_POINTS,
    INVALID_MARKERS,
)


def load_tag_mapping() -> dict:
    """从Excel加载点位映射表"""
    df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
    mapping = {}
    for _, row in df.iterrows():
        tagtable = row.get("tagtable")
        tagindex = row.get("tagindex")
        tagname = row.get("tagname", "")
        tname = row.get("TNAME1", "")
        device = row.get("设备", "")
        if pd.notna(tagtable) and pd.notna(tagindex):
            tagtable = str(tagtable).strip()
            tagindex = int(tagindex)
            if tagtable not in mapping:
                mapping[tagtable] = {}
            mapping[tagtable][tagindex] = {
                "name": tname if pd.notna(tname) else "",
                "device": device if pd.notna(device) else "",
                "tagname": tagname if pd.notna(tagname) else "",
            }
    return mapping


def _expand_row(row: pd.Series) -> pd.DataFrame:
    """将单行 T01~T60 展开为60条分钟级记录"""
    date_day = str(row["DateDay"])
    date_hour = int(row["DateHour"])
    tagindex = int(row["TagIndex"])

    base_date = datetime.strptime(date_day, "%Y-%m-%d")
    base_dt = base_date.replace(hour=date_hour)

    records = []
    for m in range(1, 61):
        col = f"T{m:02d}"
        val = row[col]
        ts = base_dt + timedelta(minutes=m - 1)
        records.append({
            "timestamp": ts,
            "tagindex": tagindex,
            "value": val,
        })
    return pd.DataFrame(records)


def load_csv_data(
    tagtable: str,
    tagindexes: list[int] | None = None,
    date_range: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """
    加载指定 tagtable 的 CSV 数据并展开为分钟级时间序列。

    Args:
        tagtable: 数据表名，如 "ZHJY", "ShuiZhi"
        tagindexes: 要加载的 tagindex 列表，None 则加载全部
        date_range: (start_date, end_date) 过滤日期范围，格式 "YYYY-MM-DD"

    Returns:
        DataFrame with columns: [timestamp, tagindex, value]
    """
    csv_files = CSV_FILES.get(tagtable, [])
    if not csv_files:
        print(f"[WARN] No CSV files found for tagtable={tagtable}")
        return pd.DataFrame(columns=["timestamp", "tagindex", "value"])

    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue

        # 按 tagindex 过滤
        if tagindexes is not None:
            df = df[df["TagIndex"].isin(tagindexes)]

        # 按日期范围过滤
        if date_range is not None:
            df["DateDay"] = df["DateDay"].astype(str)
            df = df[(df["DateDay"] >= date_range[0]) & (df["DateDay"] <= date_range[1])]

        if df.empty:
            continue

        # 展开每一行
        t_cols = [f"T{i:02d}" for i in range(1, 61)]
        required = ["TagIndex", "DateDay", "DateHour"] + t_cols
        if not all(c in df.columns for c in required):
            print(f"[WARN] Missing columns in {f}, skipping")
            continue

        expanded = pd.concat(
            [_expand_row(row) for _, row in df.iterrows()],
            ignore_index=True,
        )
        all_dfs.append(expanded)

    if not all_dfs:
        return pd.DataFrame(columns=["timestamp", "tagindex", "value"])

    result = pd.concat(all_dfs, ignore_index=True)
    result.sort_values(["tagindex", "timestamp"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def mark_invalid_values(df: pd.DataFrame, custom_markers: list | None = None) -> pd.DataFrame:
    """
    将异常标记值替换为 NaN。

    Args:
        df: 包含 value 列的 DataFrame
        custom_markers: 自定义异常值标记列表

    Returns:
        DataFrame，异常值被替换为 NaN，并增加 is_valid 列
    """
    markers = custom_markers or INVALID_MARKERS
    df = df.copy()
    df["is_valid"] = True

    for marker in markers:
        mask = np.isclose(df["value"], marker, atol=0.001)
        df.loc[mask, "value"] = np.nan
        df.loc[mask, "is_valid"] = False

    # 将明显不合理的0值标记也纳入（根据设备类型后续细化）
    return df


def load_sensor_data(
    sensor_type: str,
    sensor_name: str | None = None,
    date_range: tuple[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    加载指定传感器类型的所有传感器数据。

    Args:
        sensor_type: 传感器类型，如 "出水浊度仪", "储液池液位计"
        sensor_name: 传感器名称，None 则加载该类型全部
        date_range: 日期范围

    Returns:
        {sensor_name: DataFrame} 字典
    """
    sensor_cfg = SENSOR_POINTS.get(sensor_type)
    if sensor_cfg is None:
        print(f"[WARN] Unknown sensor type: {sensor_type}")
        return {}

    tagtable = sensor_cfg["tagtable"]
    sensors = sensor_cfg["sensors"]

    if sensor_name is not None:
        if sensor_name not in sensors:
            print(f"[WARN] Sensor '{sensor_name}' not found in {sensor_type}")
            return {}
        sensors = {sensor_name: sensors[sensor_name]}

    tagindexes = [s["tagindex"] for s in sensors.values()]
    raw = load_csv_data(tagtable, tagindexes=tagindexes, date_range=date_range)

    if raw.empty:
        return {}

    raw = mark_invalid_values(raw)

    result = {}
    index_to_name = {s["tagindex"]: name for name, s in sensors.items()}
    for name, cfg in sensors.items():
        idx = cfg["tagindex"]
        sensor_df = raw[raw["tagindex"] == idx].copy()
        sensor_df.drop(columns=["tagindex"], inplace=True)
        result[name] = sensor_df

    return result


def load_pump_data(
    pump_id: str,
    date_range: tuple[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    加载指定投矾泵的所有数据（流量、故障、状态等）。

    Args:
        pump_id: 泵编号，如 "P1"
        date_range: 日期范围

    Returns:
        {metric_name: DataFrame} 字典，metric_name 如 "flow", "error", "auto", "remote"
    """
    pump_cfg = PUMP_POINTS.get(pump_id)
    if pump_cfg is None:
        print(f"[WARN] Unknown pump: {pump_id}")
        return {}

    tagtable = pump_cfg["tagtable"]
    tagindexes = [cfg["tagindex"] for cfg in pump_cfg.values() if isinstance(cfg, dict) and "tagindex" in cfg]

    raw = load_csv_data(tagtable, tagindexes=tagindexes, date_range=date_range)
    if raw.empty:
        return {}

    raw = mark_invalid_values(raw)

    result = {}
    for metric in ["flow", "error", "auto", "remote"]:
        idx = pump_cfg[metric]["tagindex"]
        metric_df = raw[raw["tagindex"] == idx].copy()
        metric_df.drop(columns=["tagindex"], inplace=True)
        result[metric] = metric_df

    return result


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("数据加载测试")
    print("=" * 60)

    # 测试加载点位映射
    mapping = load_tag_mapping()
    print(f"\n点位映射表共 {sum(len(v) for v in mapping.values())} 个点位")
    for table, indexes in mapping.items():
        print(f"  {table}: {len(indexes)} 个点位")

    # 测试加载出水浊度仪数据（取最近1个月做演示）
    print("\n--- 出水浊度仪数据 ---")
    data = load_sensor_data("出水浊度仪", date_range=("2025-07-01", "2025-07-31"))
    for name, df in data.items():
        print(f"  {name}: {len(df)} 条记录, 时间范围 {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        valid = df["is_valid"].sum()
        print(f"    有效数据: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)")

    # 测试加载投矾泵数据
    print("\n--- 投矾泵P1数据 ---")
    pump_data = load_pump_data("P1", date_range=("2025-07-01", "2025-07-31"))
    for metric, df in pump_data.items():
        print(f"  {metric}: {len(df)} 条记录")

    # 测试加载储液池液位计
    print("\n--- 储液池液位计数据 ---")
    data = load_sensor_data("储液池液位计", date_range=("2025-07-01", "2025-07-31"))
    for name, df in data.items():
        print(f"  {name}: {len(df)} 条记录, 有效 {df['is_valid'].sum()}/{len(df)}")
