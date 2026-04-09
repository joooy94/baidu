"""tests/test_config.py - 配置模块单元测试"""

import pytest
from config import (
    CSV_FILES, PUMP_POINTS, SENSOR_POINTS,
    ABSENCE_THRESHOLDS, DRIFT_THRESHOLDS, JUMP_THRESHOLDS,
    HEALTH_WEIGHTS, HEALTH_GRADES, ALARM_LEVELS, INVALID_MARKERS,
)


class TestCsvFiles:
    """CSV文件路径配置测试"""

    def test_has_expected_tagtables(self):
        """验证四个核心数据源都配置了CSV路径"""
        assert "ZHJY" in CSV_FILES
        assert "ShuiZhi" in CSV_FILES
        assert "shuizhi" in CSV_FILES
        assert "flowpress" in CSV_FILES

    def test_all_paths_are_lists(self):
        """每个tagtable的路径必须是列表"""
        for table, paths in CSV_FILES.items():
            assert isinstance(paths, list), f"{table} 路径不是列表"
            assert len(paths) > 0, f"{table} 没有配置CSV文件"


class TestPumpPoints:
    """投矾泵点位映射测试"""

    def test_has_five_pumps(self):
        """必须有P1~P5五台泵"""
        for pid in ["P1", "P2", "P3", "P4", "P5"]:
            assert pid in PUMP_POINTS, f"缺少泵 {pid}"

    def test_each_pump_has_four_metrics(self):
        """每台泵必须有auto/error/flow/remote四个指标"""
        for pid, cfg in PUMP_POINTS.items():
            for metric in ["auto", "error", "flow", "remote"]:
                assert metric in cfg, f"{pid} 缺少指标 {metric}"
                assert "tagindex" in cfg[metric], f"{pid}.{metric} 缺少 tagindex"

    def test_each_pump_has_tagtable(self):
        """每台泵必须指定所属tagtable"""
        for pid, cfg in PUMP_POINTS.items():
            assert "tagtable" in cfg, f"{pid} 缺少 tagtable"
            assert cfg["tagtable"] == "ZHJY"


class TestSensorPoints:
    """传感器点位映射测试"""

    def test_has_expected_sensor_types(self):
        """必须有六类传感器"""
        expected = ["电磁流量计", "出水浊度仪", "进水浊度仪", "储液池液位计", "投加流量计", "阀门"]
        for stype in expected:
            assert stype in SENSOR_POINTS, f"缺少传感器类型: {stype}"

    def test_each_type_has_tagtable_and_sensors(self):
        """每个传感器类型必须有tagtable和sensors"""
        for stype, cfg in SENSOR_POINTS.items():
            assert "tagtable" in cfg, f"{stype} 缺少 tagtable"
            assert "sensors" in cfg, f"{stype} 缺少 sensors"
            assert len(cfg["sensors"]) > 0, f"{stype} 没有传感器点位"


class TestThresholds:
    """诊断阈值配置测试"""

    def test_absence_thresholds_complete(self):
        """数据缺失阈值必须包含1/2/3三个等级"""
        assert 1 in ABSENCE_THRESHOLDS
        assert 2 in ABSENCE_THRESHOLDS
        assert 3 in ABSENCE_THRESHOLDS
        assert ABSENCE_THRESHOLDS[1] > ABSENCE_THRESHOLDS[2] > ABSENCE_THRESHOLDS[3]

    def test_drift_thresholds_for_turbidity(self):
        """浊度仪漂移阈值必须有三级"""
        for stype in ["出水浊度仪", "进水浊度仪"]:
            assert stype in DRIFT_THRESHOLDS
            assert 3 in DRIFT_THRESHOLDS[stype]

    def test_drift_thresholds_for_level(self):
        """液位计漂移阈值必须包含绝对偏差"""
        assert "储液池液位计" in DRIFT_THRESHOLDS

    def test_jump_thresholds_for_turbidity(self):
        """浊度仪异跳阈值必须有配置"""
        assert "出水浊度仪" in JUMP_THRESHOLDS
        assert "rate_per_min" in JUMP_THRESHOLDS["出水浊度仪"]

    def test_jump_thresholds_for_flow(self):
        """流量计异跳阈值使用正常波动倍数"""
        assert "电磁流量计" in JUMP_THRESHOLDS
        assert "normal_multiple" in JUMP_THRESHOLDS["电磁流量计"]


class TestHealthConfig:
    """健康度评估配置测试"""

    def test_weights_sum_to_one(self):
        """四项权重之和必须为1"""
        total = sum(HEALTH_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001, f"权重总和 {total} 不等于1"

    def test_weights_have_four_keys(self):
        """必须有四项权重"""
        expected_keys = {"flow_deviation", "fault_frequency", "stability", "responsiveness"}
        assert set(HEALTH_WEIGHTS.keys()) == expected_keys

    def test_grades_cover_full_range(self):
        """等级区间必须覆盖0~100"""
        ranges = sorted(HEALTH_GRADES.keys(), key=lambda r: r[0])
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 101

    def test_alarm_levels(self):
        """报警等级必须包含1/2/3"""
        assert 1 in ALARM_LEVELS
        assert 2 in ALARM_LEVELS
        assert 3 in ALARM_LEVELS

    def test_invalid_markers(self):
        """异常值标记列表不能为空"""
        assert len(INVALID_MARKERS) > 0
