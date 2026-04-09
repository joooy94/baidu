"""tests/test_health_assessment.py - 健康度评估模块单元测试"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from health_assessment import (
    _calc_flow_deviation, _calc_fault_frequency,
    _calc_stability, _calc_responsiveness,
    assess_pump_health, PumpHealthResult, HealthIndicator,
)


def _make_ts_df(n: int, values: list | None = None, start: str = "2025-07-01") -> pd.DataFrame:
    """生成测试用时间序列 DataFrame"""
    base = datetime.strptime(start, "%Y-%m-%d")
    ts = [base + timedelta(minutes=i) for i in range(n)]
    if values is None:
        values = [1.0] * n
    return pd.DataFrame({
        "timestamp": ts[:len(values)],
        "value": values,
        "is_valid": [True] * len(values),
    })


class TestCalcFlowDeviation:
    """流量偏差率指标测试"""

    def test_stable_flow_high_score(self):
        """稳定流量应得高分"""
        values = [100.0 + np.random.normal(0, 0.5) for _ in range(200)]
        df = _make_ts_df(200, values)
        ind = _calc_flow_deviation(df)
        assert ind.score >= 80
        assert ind.name == "流量偏差率"

    def test_wild_flow_low_score(self):
        """剧烈波动流量应得低分"""
        values = [100.0] * 50 + [300.0] * 50 + [100.0] * 50 + [300.0] * 50
        df = _make_ts_df(200, values)
        ind = _calc_flow_deviation(df)
        assert ind.score < 60

    def test_empty_df_returns_default(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        ind = _calc_flow_deviation(df)
        assert ind.score == 50  # 默认值

    def test_insufficient_data_returns_default(self):
        df = _make_ts_df(5)
        ind = _calc_flow_deviation(df)
        assert ind.score == 50

    def test_returns_health_indicator(self):
        df = _make_ts_df(100, [100.0] * 100)
        ind = _calc_flow_deviation(df)
        assert isinstance(ind, HealthIndicator)
        assert 0 <= ind.score <= 100
        assert ind.weight == 0.30


class TestCalcFaultFrequency:
    """故障频率指标测试"""

    def test_no_faults_high_score(self):
        """零故障应得100分"""
        df = _make_ts_df(100, [0.0] * 100)
        ind = _calc_fault_frequency(df, 100)
        assert ind.score == 100.0

    def test_frequent_faults_low_score(self):
        """频繁故障应得低分"""
        df = _make_ts_df(100, [1.0] * 50 + [0.0] * 50)
        ind = _calc_fault_frequency(df, 24)  # 1天内50次故障
        assert ind.score < 50

    def test_empty_df_returns_default(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        ind = _calc_fault_frequency(df, 0)
        assert ind.score == 50

    def test_result_has_correct_weight(self):
        df = _make_ts_df(100)
        ind = _calc_fault_frequency(df, 100)
        assert ind.weight == 0.30


class TestCalcStability:
    """运行稳定性指标测试"""

    def test_stable_flow_high_score(self):
        """稳定流量(低CV)应得高分"""
        values = [100.0] * 200
        df = _make_ts_df(200, values)
        ind = _calc_stability(df)
        assert ind.score >= 90

    def test_unstable_flow_low_score(self):
        """不稳定流量(高CV)应得低分"""
        values = [10.0, 200.0, 5.0, 300.0] * 50
        df = _make_ts_df(200, values)
        ind = _calc_stability(df)
        assert ind.score < 50

    def test_all_zero_returns_default(self):
        """全零流量无法计算CV"""
        df = _make_ts_df(100, [0.0] * 100)
        ind = _calc_stability(df)
        assert ind.score == 50

    def test_empty_df_returns_default(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        ind = _calc_stability(df)
        assert ind.score == 50


class TestCalcResponsiveness:
    """控制响应性指标测试"""

    def test_perfect_consistency(self):
        """auto和remote完全一致"""
        auto_df = _make_ts_df(100, [1.0] * 100)
        remote_df = _make_ts_df(100, [1.0] * 100)
        ind = _calc_responsiveness(auto_df, remote_df)
        assert ind.score == 100.0

    def test_no_consistency(self):
        """auto和remote完全不一致"""
        auto_df = _make_ts_df(100, [1.0] * 100)
        remote_df = _make_ts_df(100, [0.0] * 100)
        ind = _calc_responsiveness(auto_df, remote_df)
        assert ind.score < 10

    def test_empty_returns_default(self):
        df = pd.DataFrame(columns=["timestamp", "value", "is_valid"])
        ind = _calc_responsiveness(df, df)
        assert ind.score == 50


class TestAssessPumpHealth:
    """综合健康度评估测试"""

    def test_returns_pump_health_result(self):
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(result, PumpHealthResult)

    def test_score_in_range(self):
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        assert 0 <= result.overall_score <= 100

    def test_grade_is_valid(self):
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        assert result.grade in ["优秀", "良好", "轻微异常", "异常"]

    def test_has_four_indicators(self):
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        assert len(result.indicators) == 4
        names = [ind.name for ind in result.indicators]
        assert "流量偏差率" in names
        assert "故障频率" in names
        assert "运行稳定性" in names
        assert "控制响应性" in names

    def test_pump_id_matches(self):
        result = assess_pump_health("P3", date_range=("2025-07-01", "2025-07-01"))
        assert result.pump_id == "P3"

    def test_nonexistent_pump_still_returns(self):
        result = assess_pump_health("P99")
        assert isinstance(result, PumpHealthResult)

    def test_weights_sum_to_one(self):
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        total_weight = sum(ind.weight for ind in result.indicators)
        assert abs(total_weight - 1.0) < 0.01


class TestGenerateHealthReport:
    """健康度报告生成测试"""

    def test_empty_results_report(self):
        """空结果应生成报告"""
        from health_assessment import generate_health_report
        report = generate_health_report([])
        assert "投加设备健康度评估报告" in report

    def test_report_contains_pump_info(self):
        """报告应包含泵编号和分数"""
        from health_assessment import generate_health_report
        result = assess_pump_health("P1", date_range=("2025-07-01", "2025-07-01"))
        report = generate_health_report([result])
        assert "P1" in report
        assert "综合健康度" in report

    def test_report_shows_warnings(self):
        """报告应显示告警信息"""
        from health_assessment import generate_health_report
        results = []
        for pid in ["P1", "P2", "P3", "P4", "P5"]:
            results.append(assess_pump_health(pid, date_range=("2025-07-01", "2025-07-01")))
        report = generate_health_report(results)
        assert "汇总" in report

    def test_report_custom_title(self):
        """报告应支持自定义标题"""
        from health_assessment import generate_health_report
        report = generate_health_report([], title="自定义健康报告")
        assert "自定义健康报告" in report


class TestCalcFlowDeviationEdgeCases:
    """流量偏差率边界测试"""

    def test_zero_rolling_mean(self):
        """滚动均值为0时不应崩溃"""
        values = [0.0] * 200
        df = _make_ts_df(200, values)
        ind = _calc_flow_deviation(df)
        assert isinstance(ind.score, (float, int))

    def test_single_value(self):
        """只有一个值时应返回默认"""
        df = _make_ts_df(1, [100.0])
        ind = _calc_flow_deviation(df)
        assert ind.score == 50


class TestCalcFaultFrequencyEdgeCases:
    """故障频率边界测试"""

    def test_zero_hours(self):
        """总时间为0时应返回默认"""
        df = _make_ts_df(10, [0.0] * 10)
        ind = _calc_fault_frequency(df, 0)
        assert ind.score == 50

    def test_negative_hours(self):
        """负时间应返回默认"""
        df = _make_ts_df(10)
        ind = _calc_fault_frequency(df, -1)
        assert ind.score == 50
