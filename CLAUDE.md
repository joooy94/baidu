# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

投矾智慧控制系统（智慧投矾）— 为广州市自来水有限公司北部水厂设计的混凝剂（矾液）投加智慧控制系统。基于《概要设计说明书-V1.7》实现两大核心功能：**投加设备健康度评估**和**传感器智慧诊断**。

## Commands

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_sensor_diagnosis.py -v

# Run a specific test
python3 -m pytest tests/test_data_loader.py::TestLoadCsvData::test_load_zhjy_returns_dataframe -v

# Generate all visualization charts (accepts date range args)
python3 -m src.visualization 2025-07-01 2025-07-07

# Run individual modules (has __main__ CLI for quick validation)
python3 -m src.data_loader
python3 -m src.sensor_diagnosis
python3 -m src.health_assessment
```

## Architecture

Data flows bottom-up through four layers:

```
src/config.py          — Point mappings (tagtable → tagindex), thresholds, weights
    ↓
src/data_loader.py     — CSV loading, T01~T60 → minute-level time series expansion, invalid value marking
    ↓
src/sensor_diagnosis.py / src/health_assessment.py  — Diagnosis logic & health scoring
    ↓
src/visualization.py   — Chart generation (radar, bar, trend, timeseries)
```

### Key data model

- **Raw CSV format**: `Bid, TagIndex, DateDay, DateHour, T01~T60` — each row is one hour of minute-level readings for one sensor point
- **Expanded format**: `timestamp, tagindex, value, is_valid` — one row per minute
- **Point mapping**: `tagtable` (e.g. "ZHJY") + `tagindex` (integer) identifies a specific sensor/device point, mapped in `docs/投矾系统的智慧诊断点位.xlsx`

### Data sources (tagtable → CSV)

| tagtable | Source | Contents |
|---|---|---|
| `ZHJY` | `data/原始数据/ZHJY__*.csv` | Pump flow/error/auto/remote, valves, tank levels, pipe flow meters |
| `ShuiZhi` | `data/原始数据/ShuiZhi_*.csv` | Inlet turbidity (reaction pool) |
| `shuizhi` | `data/原始数据/ssts/ssts_*.csv` | Outlet turbidity (sedimentation pool) — monthly files |
| `flowpress` | `data/原始数据/流量数据/flowpress_*.csv` | Electromagnetic flow meters — 2024 only |

### Sensor diagnosis (sensor_diagnosis.py)

Three anomaly detection types, each with three alarm levels (1=highest):
- **Data absence** (`detect_data_absence`): consecutive minutes of missing/invalid data → 10min(三级)/30min(二级)/60min(一级)
- **Data drift** (`detect_data_drift`): deviation from rolling historical mean → thresholds vary by sensor type (e.g. turbidity ±15%/±40%/±50%)
- **Data jump** (`detect_data_jump`): rate-of-change anomaly → fixed threshold (turbidity: 50NTU/min) or normal-fluctuation multiple (flow/level: 3x)

### Health assessment (health_assessment.py)

Four weighted indicators per pump (P1~P5):
- Flow deviation rate (30%) — `|actual - rolling_mean| / rolling_mean`
- Fault frequency (30%) — error=1 occurrences per day
- Running stability (20%) — coefficient of variation (CV = std/mean) of flow
- Control responsiveness (20%) — auto/remote state consistency

Score 0~100 → grades: 优秀(90+)/良好(70~90)/轻微异常(50~70)/异常(<50)

## Dependencies

```
pandas, numpy, python-docx, openpyxl, matplotlib, pytest
```

Install: `pip3 install pandas numpy python-docx openpyxl matplotlib pytest`

## Key design document

`docs/概要设计说明书-V1.7-2026.2.3-投帆智慧控制系统.docx` — contains full system design including threshold definitions, alarm rules, and MPC control architecture. Refer to it when modifying diagnosis thresholds or health scoring logic.
