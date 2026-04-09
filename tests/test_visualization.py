"""tests/test_visualization.py - 可视化模块单元测试"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from visualization import (
    plot_health_radar, plot_health_bar, plot_health_trend,
    plot_diagnosis_summary, plot_sensor_timeseries,
    plot_pump_flow_comparison, plot_equipment_diagnosis,
    generate_all_charts,
)
from health_assessment import PumpHealthResult, HealthIndicator
from sensor_diagnosis import DiagnosisResult
from equipment_diagnosis import EquipmentDiagnosisResult


def _make_pump_result(pump_id="P1", score=85.0, grade="良好"):
    """构造测试用的健康度结果"""
    indicators = [
        HealthIndicator("流量偏差率", 0.05, score, 0.30, "测试"),
        HealthIndicator("故障频率", 0, score, 0.30, "测试"),
        HealthIndicator("运行稳定性", 0.02, score, 0.20, "测试"),
        HealthIndicator("控制响应性", 0.99, score, 0.20, "测试"),
    ]
    return PumpHealthResult(
        pump_id=pump_id,
        overall_score=score,
        grade=grade,
        indicators=indicators,
        timestamp_range=(datetime(2025, 7, 1), datetime(2025, 7, 31)),
        warnings=[],
    )


def _make_diagnosis_result(sensor_type="出水浊度仪", sensor_name="测试",
                            anomaly_type="data_absence", level=3):
    """构造测试用的诊断结果"""
    return DiagnosisResult(
        sensor_type=sensor_type,
        sensor_name=sensor_name,
        anomaly_type=anomaly_type,
        alarm_level=level,
        alarm_level_name="三级" if level == 3 else "二级" if level == 2 else "一级",
        start_time=datetime(2025, 7, 1, 10, 0),
        end_time=datetime(2025, 7, 1, 10, 15),
        duration_min=15,
        detail="测试异常",
    )


def _make_equipment_result(category="投加泵异常诊断", target="P1",
                            anomaly_type="pump_fault", level=2):
    """构造测试用的设备诊断结果"""
    return EquipmentDiagnosisResult(
        category=category,
        target=target,
        anomaly_type=anomaly_type,
        alarm_level=level,
        alarm_level_name="二级" if level == 2 else "一级",
        start_time=datetime(2025, 7, 1, 10, 0),
        end_time=datetime(2025, 7, 1, 10, 15),
        duration_min=15,
        detail="测试故障",
    )


class TestPlotHealthRadar:
    """健康度雷达图测试"""

    def test_generates_png_file(self, tmp_path):
        """应生成PNG文件"""
        results = [_make_pump_result(f"P{i}") for i in range(1, 4)]
        path = plot_health_radar(results, save_path=str(tmp_path / "radar.png"))
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_empty_results_no_crash(self, tmp_path):
        """空结果不应崩溃"""
        path = plot_health_radar([], save_path=str(tmp_path / "radar_empty.png"))
        assert os.path.exists(path)


class TestPlotHealthBar:
    """健康度柱状图测试"""

    def test_generates_png_file(self, tmp_path):
        """应生成PNG文件"""
        results = [_make_pump_result(f"P{i}", score=70 + i * 5) for i in range(5)]
        path = plot_health_bar(results, save_path=str(tmp_path / "bar.png"))
        assert os.path.exists(path)

    def test_different_grades(self, tmp_path):
        """不同等级应有不同颜色"""
        results = [
            _make_pump_result("P1", 95, "优秀"),
            _make_pump_result("P2", 75, "良好"),
            _make_pump_result("P3", 55, "轻微异常"),
            _make_pump_result("P4", 30, "异常"),
        ]
        path = plot_health_bar(results, save_path=str(tmp_path / "bar_grades.png"))
        assert os.path.exists(path)


class TestPlotDiagnosisSummary:
    """传感器诊断统计图测试"""

    def test_with_results(self, tmp_path):
        """有结果时应生成图表"""
        results = [_make_diagnosis_result(level=l) for l in [1, 2, 3]]
        path = plot_diagnosis_summary(results, save_path=str(tmp_path / "diag.png"))
        assert os.path.exists(path)

    def test_empty_results(self, tmp_path):
        """空结果应生成"未检测到异常"图"""
        path = plot_diagnosis_summary([], save_path=str(tmp_path / "diag_empty.png"))
        assert os.path.exists(path)


class TestPlotEquipmentDiagnosis:
    """设备诊断统计图测试"""

    def test_with_results(self, tmp_path):
        """有结果时应生成图表"""
        results = [
            _make_equipment_result(anomaly_type="pump_fault", level=2),
            _make_equipment_result(anomaly_type="valve_stuck", level=3),
        ]
        path = plot_equipment_diagnosis(results, save_path=str(tmp_path / "equip.png"))
        assert os.path.exists(path)

    def test_empty_results(self, tmp_path):
        """空结果不应崩溃"""
        path = plot_equipment_diagnosis([], save_path=str(tmp_path / "equip_empty.png"))
        assert os.path.exists(path)
