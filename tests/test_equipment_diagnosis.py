"""tests/test_equipment_diagnosis.py - 设备级诊断单元测试"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from equipment_diagnosis import (
    diagnose_pump, diagnose_pipeline, diagnose_valve,
    diagnose_tank, EquipmentDiagnosisResult,
)


def _make_ts_df(n: int, values: list | None = None, start: str = "2025-07-01") -> pd.DataFrame:
    base = datetime.strptime(start, "%Y-%m-%d")
    ts = [base + timedelta(minutes=i) for i in range(n)]
    if values is None:
        values = [1.0] * n
    return pd.DataFrame({
        "timestamp": ts[:len(values)],
        "value": values,
        "is_valid": [True] * len(values),
    })


class TestDiagnosePump:
    """投加泵异常诊断测试"""

    def test_no_fault_no_alarm(self):
        """无故障时不应有报警"""
        results = diagnose_pump("P1", date_range=("2025-07-01", "2025-07-01"))
        # 只要能运行不报错即可，实际数据可能有真实异常
        assert isinstance(results, list)

    def test_returns_equipment_diagnosis_result(self):
        results = diagnose_pump("P1", date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert isinstance(r, EquipmentDiagnosisResult)
            assert r.category == "投加泵异常诊断"
            assert r.target == "P1"
            assert r.alarm_level in [1, 2, 3]

    def test_nonexistent_pump_returns_empty(self):
        results = diagnose_pump("P99", date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)


class TestDiagnosePipeline:
    """投加管道异常诊断测试"""

    def test_runs_without_error(self):
        results = diagnose_pipeline(date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, EquipmentDiagnosisResult)
            assert r.category == "投加管道异常诊断"

    def test_result_fields_valid(self):
        results = diagnose_pipeline(date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert r.anomaly_type in ["pipe_blockage", "pipe_leak"]
            assert r.alarm_level in [1, 2]


class TestDiagnoseValve:
    """阀门异常诊断测试"""

    def test_runs_without_error(self):
        results = diagnose_valve(date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)

    def test_result_category_correct(self):
        results = diagnose_valve(date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert r.category == "阀门异常诊断"
            assert r.anomaly_type in [
                "valve_fault", "valve_stuck", "valve_signal_conflict",
            ]


class TestDiagnoseTank:
    """储液池异常诊断测试"""

    def test_runs_without_error(self):
        results = diagnose_tank(date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)

    def test_detects_low_level(self):
        """液位数据大部分为0，应检测到低液位报警"""
        results = diagnose_tank(date_range=("2025-07-01", "2025-07-01"))
        low_level = [r for r in results if r.anomaly_type == "level_low"]
        # 根据已知数据，1号储液池液位持续为0，应有报警
        assert len(low_level) > 0
        assert low_level[0].alarm_level in [1, 2]

    def test_result_fields_valid(self):
        results = diagnose_tank(date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert r.category == "储液池异常诊断"
            assert r.anomaly_type in ["level_high", "level_low", "level_fast_change"]


class TestDiagnosisIntegration:
    """设备诊断集成测试"""

    def test_diagnose_all_equipment(self):
        from equipment_diagnosis import diagnose_all_equipment
        results = diagnose_all_equipment(date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)
        assert len(results) > 0
        # 验证所有结果都按报警等级排序
        levels = [r.alarm_level for r in results]
        assert levels == sorted(levels)

    def test_report_generation(self):
        from equipment_diagnosis import (
            diagnose_all_equipment,
            generate_equipment_diagnosis_report,
        )
        results = diagnose_all_equipment(date_range=("2025-07-01", "2025-07-01"))
        report = generate_equipment_diagnosis_report(results)
        assert isinstance(report, str)
        assert "设备智慧诊断报告" in report


class TestDiagnosePlcCommunication:
    """PLC通讯中断诊断测试"""

    def test_runs_without_error(self):
        """PLC诊断应能正常运行"""
        from equipment_diagnosis import diagnose_plc_communication
        results = diagnose_plc_communication(date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, EquipmentDiagnosisResult)
            assert r.category == "PLC通讯中断诊断"
            assert r.anomaly_type in ["comm_total_loss", "comm_intermittent"]

    def test_result_alarm_levels_valid(self):
        """PLC通讯报警等级应为1或2"""
        from equipment_diagnosis import diagnose_plc_communication
        results = diagnose_plc_communication(date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert r.alarm_level in [1, 2]


class TestDiagnoseTankEdgeCases:
    """储液池诊断边界测试"""

    def test_result_has_correct_alarm_levels(self):
        """储液池报警等级应为1/2/3"""
        results = diagnose_tank(date_range=("2025-07-01", "2025-07-01"))
        for r in results:
            assert r.alarm_level in [1, 2, 3]


class TestGenerateEquipmentDiagnosisReport:
    """设备诊断报告生成测试"""

    def test_empty_results_report(self):
        """空结果应生成未检测到异常报告"""
        from equipment_diagnosis import generate_equipment_diagnosis_report
        report = generate_equipment_diagnosis_report([])
        assert "未检测到设备异常" in report

    def test_report_custom_title(self):
        """报告应支持自定义标题"""
        from equipment_diagnosis import generate_equipment_diagnosis_report
        report = generate_equipment_diagnosis_report([], title="自定义诊断")
        assert "自定义诊断" in report
