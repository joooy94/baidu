"""tests/test_sensor_diagnosis.py - 传感器诊断模块单元测试"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from sensor_diagnosis import (
    detect_data_absence, detect_data_drift, detect_data_jump,
    diagnose_sensor, DiagnosisResult,
)


def _make_timestamp_series(n: int, start: str = "2025-07-01") -> list[datetime]:
    """生成连续分钟级时间戳"""
    base = datetime.strptime(start, "%Y-%m-%d")
    return [base + timedelta(minutes=i) for i in range(n)]


class TestDetectDataAbsence:
    """数据缺失检测测试"""

    def test_no_absence_no_alarm(self):
        """全部有效数据不应触发报警"""
        ts = _make_timestamp_series(100)
        df = pd.DataFrame({
            "timestamp": ts,
            "value": np.random.uniform(0.5, 1.5, 100),
            "is_valid": [True] * 100,
        })
        results = detect_data_absence(df, "出水浊度仪", "test")
        assert len(results) == 0

    def test_10min_absence_triggers_level3(self):
        """连续10分钟缺失应触发三级报警"""
        ts = _make_timestamp_series(100)
        is_valid = [True] * 30 + [False] * 15 + [True] * 55
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 100,
            "is_valid": is_valid,
        })
        results = detect_data_absence(df, "出水浊度仪", "test")
        level3 = [r for r in results if r.alarm_level == 3]
        assert len(level3) >= 1

    def test_30min_absence_triggers_level2_or_higher(self):
        """连续35分钟缺失应触发二级或更高报警"""
        ts = _make_timestamp_series(100)
        is_valid = [True] * 10 + [False] * 35 + [True] * 55
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 100,
            "is_valid": is_valid,
        })
        results = detect_data_absence(df, "出水浊度仪", "test")
        assert len(results) >= 1
        # 35分钟>=30分钟阈值，报警等级应<=2(二级或一级)
        assert results[0].alarm_level <= 2

    def test_60min_absence_triggers_level1(self):
        """连续65分钟缺失应触发一级报警"""
        ts = _make_timestamp_series(200)
        is_valid = [True] * 10 + [False] * 65 + [True] * 125
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 200,
            "is_valid": is_valid,
        })
        results = detect_data_absence(df, "出水浊度仪", "test")
        assert len(results) >= 1
        # 65分钟>=60分钟阈值，应触发一级
        assert results[0].alarm_level == 1

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        results = detect_data_absence(df, "出水浊度仪", "test")
        assert len(results) == 0

    def test_result_has_correct_fields(self):
        ts = _make_timestamp_series(50)
        is_valid = [False] * 15 + [True] * 35
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 50,
            "is_valid": is_valid,
        })
        results = detect_data_absence(df, "出水浊度仪", "test_sensor")
        assert len(results) >= 1
        r = results[0]
        assert r.sensor_type == "出水浊度仪"
        assert r.sensor_name == "test_sensor"
        assert r.anomaly_type == "data_absence"
        assert r.alarm_level in [1, 2, 3]


class TestDetectDataDrift:
    """数据漂移检测测试"""

    def test_stable_data_no_alarm(self):
        """稳定数据不应触发漂移报警"""
        ts = _make_timestamp_series(200)
        values = [1.0] * 200
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_drift(df, "出水浊度仪", "test")
        assert len(results) == 0

    def test_large_drift_triggers_alarm(self):
        """大幅漂移应触发报警"""
        ts = _make_timestamp_series(200)
        # 前100个正常，后100个漂移到3倍
        values = [1.0] * 100 + [3.0] * 100
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_drift(df, "出水浊度仪", "test")
        assert len(results) >= 1
        # 200%偏差应触发一级报警(>50%)
        level1 = [r for r in results if r.alarm_level == 1]
        assert len(level1) >= 1

    def test_unknown_sensor_type_returns_empty(self):
        ts = _make_timestamp_series(200)
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 200,
            "is_valid": [True] * 200,
        })
        results = detect_data_drift(df, "未知传感器", "test")
        assert len(results) == 0

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        results = detect_data_drift(df, "出水浊度仪", "test")
        assert len(results) == 0

    def test_insufficient_data_returns_empty(self):
        """少于30个数据点不应触发"""
        ts = _make_timestamp_series(10)
        df = pd.DataFrame({
            "timestamp": ts,
            "value": [1.0] * 10,
            "is_valid": [True] * 10,
        })
        results = detect_data_drift(df, "出水浊度仪", "test")
        assert len(results) == 0


class TestDetectDataJump:
    """数据异跳检测测试"""

    def test_smooth_data_no_alarm(self):
        """平滑数据不应触发异跳"""
        ts = _make_timestamp_series(200)
        values = np.linspace(1.0, 1.5, 200).tolist()
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_jump(df, "出水浊度仪", "test")
        assert len(results) == 0

    def test_sudden_jump_triggers_alarm(self):
        """突然跳变应触发异跳"""
        ts = _make_timestamp_series(200)
        # 正常值中插入一个大幅跳变
        values = [1.0] * 50 + [100.0, 1.0] + [1.0] * 148
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_jump(df, "进水浊度仪", "test")
        # 应检测到跳变（变化率>正常波动3倍）
        assert len(results) >= 1

    def test_turbidity_fixed_threshold(self):
        """出水浊度仪用固定阈值50NTU/min"""
        ts = _make_timestamp_series(200)
        values = [1.0] * 100
        # 连续4个大于50NTU/min的变化
        for i in range(4):
            values.append(60.0 * (i + 1))
        values += [1.0] * 96
        df = pd.DataFrame({
            "timestamp": ts[:len(values)],
            "value": values,
            "is_valid": [True] * len(values),
        })
        results = detect_data_jump(df, "出水浊度仪", "test")
        assert len(results) >= 1

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        results = detect_data_jump(df, "出水浊度仪", "test")
        assert len(results) == 0


class TestDiagnoseSensor:
    """综合诊断入口测试"""

    def test_diagnose_returns_list(self):
        """diagnose_sensor应返回列表"""
        results = diagnose_sensor("出水浊度仪", sensor_name="沉淀池出水_1#浊度",
                                   date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, DiagnosisResult)

    def test_diagnose_unknown_type_returns_empty(self):
        """未知传感器类型应返回空列表"""
        results = diagnose_sensor("不存在的类型", date_range=("2025-07-01", "2025-07-01"))
        assert len(results) == 0


class TestGenerateDiagnosisReport:
    """诊断报告生成测试"""

    def test_empty_results_report(self):
        """空结果应生成"未检测到异常"报告"""
        from sensor_diagnosis import generate_diagnosis_report
        report = generate_diagnosis_report([])
        assert "未检测到异常" in report

    def test_report_with_results(self):
        """有结果时报告应包含传感器类型"""
        from sensor_diagnosis import generate_diagnosis_report
        r = DiagnosisResult(
            sensor_type="出水浊度仪",
            sensor_name="测试",
            anomaly_type="data_absence",
            alarm_level=3,
            alarm_level_name="三级",
            start_time=datetime(2025, 7, 1, 10, 0),
            end_time=datetime(2025, 7, 1, 10, 15),
            duration_min=15,
            detail="连续15分钟无有效数据",
        )
        report = generate_diagnosis_report([r])
        assert "出水浊度仪" in report
        assert "数据缺失" in report

    def test_report_custom_title(self):
        """报告应支持自定义标题"""
        from sensor_diagnosis import generate_diagnosis_report
        report = generate_diagnosis_report([], title="自定义报告")
        assert "自定义报告" in report


class TestDetectDataJumpLevelMeter:
    """液位计异跳检测测试"""

    def test_level_sudden_change_triggers_alarm(self):
        """液位10分钟内变化超30cm应触发报警"""
        ts = _make_timestamp_series(200)
        values = [100.0] * 50 + [140.0] * 50 + [100.0] * 100
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_jump(df, "储液池液位计", "test")
        assert len(results) >= 1

    def test_level_slow_change_no_alarm(self):
        """液位缓慢变化不应触发异跳"""
        ts = _make_timestamp_series(200)
        values = np.linspace(100.0, 105.0, 200).tolist()
        df = pd.DataFrame({
            "timestamp": ts,
            "value": values,
            "is_valid": [True] * 200,
        })
        results = detect_data_jump(df, "储液池液位计", "test")
        assert len(results) == 0
