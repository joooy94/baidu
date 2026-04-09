"""tests/test_data_loader.py - 数据加载模块单元测试"""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime

from data_loader import load_tag_mapping, load_csv_data, mark_invalid_values, load_sensor_data, load_pump_data


class TestLoadTagMapping:
    """点位映射加载测试"""

    def test_load_mapping_returns_dict(self):
        mapping = load_tag_mapping()
        assert isinstance(mapping, dict)

    def test_mapping_has_expected_tables(self):
        mapping = load_tag_mapping()
        assert "ZHJY" in mapping
        assert "ShuiZhi" in mapping
        assert "shuizhi" in mapping
        assert "flowpress" in mapping

    def test_zhjy_contains_pump_points(self):
        mapping = load_tag_mapping()
        # 投矾泵P1的流量点位 tagindex=455
        assert 455 in mapping["ZHJY"]
        assert "流量" in mapping["ZHJY"][455]["name"] or "投加" in mapping["ZHJY"][455]["name"]

    def test_shuizhi_contains_turbidity_points(self):
        mapping = load_tag_mapping()
        # 出水浊度 tagindex=4
        assert 4 in mapping["shuizhi"]

    def test_each_entry_has_name_and_device(self):
        mapping = load_tag_mapping()
        for table, indexes in mapping.items():
            for idx, info in indexes.items():
                assert "name" in info, f"{table}/{idx} missing 'name'"
                assert "device" in info, f"{table}/{idx} missing 'device'"


class TestLoadCsvData:
    """CSV数据加载测试"""

    def test_load_zhjy_returns_dataframe(self):
        df = load_csv_data("ZHJY", tagindexes=[455], date_range=("2025-07-01", "2025-07-01"))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_with_date_range(self):
        df = load_csv_data("ZHJY", tagindexes=[455], date_range=("2025-07-01", "2025-07-01"))
        # 应该只有7月1日的数据
        assert all(df["timestamp"].dt.date == pd.Timestamp("2025-07-01").date())

    def test_load_nonexistent_table_returns_empty(self):
        df = load_csv_data("NONEXISTENT")
        assert df.empty

    def test_load_nonexistent_tagindex_returns_empty(self):
        df = load_csv_data("ZHJY", tagindexes=[99999], date_range=("2025-07-01", "2025-07-01"))
        assert df.empty

    def test_has_expected_columns(self):
        df = load_csv_data("ZHJY", tagindexes=[455], date_range=("2025-07-01", "2025-07-01"))
        assert "timestamp" in df.columns
        assert "tagindex" in df.columns
        assert "value" in df.columns

    def test_timestamp_is_datetime(self):
        df = load_csv_data("ZHJY", tagindexes=[455], date_range=("2025-07-01", "2025-07-01"))
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_one_hour_gives_60_records(self):
        # 1小时的T01~T60展开为60条
        df = load_csv_data("ZHJY", tagindexes=[455], date_range=("2025-07-01", "2025-07-01"))
        # 至少应该有数据
        assert len(df) >= 60


class TestMarkInvalidValues:
    """异常值标记测试"""

    def test_marks_known_invalid(self):
        df = pd.DataFrame({"value": [1.0, -4.9991, 3.0, -9999]})
        result = mark_invalid_values(df)
        assert result.loc[1, "is_valid"] == False
        assert result.loc[3, "is_valid"] == False
        assert pd.isna(result.loc[1, "value"])
        assert pd.isna(result.loc[3, "value"])

    def test_keeps_valid_data(self):
        df = pd.DataFrame({"value": [1.0, 2.5, 3.0, 100.0]})
        result = mark_invalid_values(df)
        assert all(result["is_valid"])

    def test_has_is_valid_column(self):
        df = pd.DataFrame({"value": [1.0]})
        result = mark_invalid_values(df)
        assert "is_valid" in result.columns


class TestLoadSensorData:
    """传感器数据加载测试"""

    def test_load_turbidity_sensors(self):
        data = load_sensor_data("出水浊度仪", date_range=("2025-07-01", "2025-07-01"))
        assert len(data) == 4  # 4个出水浊度仪
        for name, df in data.items():
            assert "timestamp" in df.columns
            assert "value" in df.columns
            assert "is_valid" in df.columns
            assert len(df) > 0

    def test_load_specific_sensor(self):
        data = load_sensor_data("出水浊度仪", sensor_name="沉淀池出水_1#浊度", date_range=("2025-07-01", "2025-07-01"))
        assert len(data) == 1
        assert "沉淀池出水_1#浊度" in data

    def test_load_nonexistent_sensor_returns_empty(self):
        data = load_sensor_data("出水浊度仪", sensor_name="不存在的传感器")
        assert len(data) == 0

    def test_load_nonexistent_type_returns_empty(self):
        data = load_sensor_data("不存在的类型")
        assert len(data) == 0


class TestLoadPumpData:
    """投矾泵数据加载测试"""

    def test_load_p1_all_metrics(self):
        data = load_pump_data("P1", date_range=("2025-07-01", "2025-07-01"))
        assert "flow" in data
        assert "error" in data
        assert "auto" in data
        assert "remote" in data

    def test_load_nonexistent_pump(self):
        data = load_pump_data("P99")
        assert len(data) == 0

    def test_flow_data_has_values(self):
        data = load_pump_data("P1", date_range=("2025-07-01", "2025-07-01"))
        assert len(data["flow"]) > 0


class TestExpandRow:
    """_expand_row 行展开测试"""

    def test_expand_one_hour(self):
        """单行展开应生成60条分钟级记录"""
        from data_loader import _expand_row
        row = pd.Series({
            "DateDay": "2025-07-01",
            "DateHour": 10,
            "TagIndex": 100,
            **{f"T{i:02d}": float(i) for i in range(1, 61)},
        })
        result = _expand_row(row)
        assert len(result) == 60
        assert "timestamp" in result.columns
        assert "tagindex" in result.columns
        assert "value" in result.columns

    def test_expand_timestamp_starts_at_hour(self):
        """展开后的时间戳应从整点开始"""
        from data_loader import _expand_row
        row = pd.Series({
            "DateDay": "2025-07-01",
            "DateHour": 14,
            "TagIndex": 100,
            **{f"T{i:02d}": 1.0 for i in range(1, 61)},
        })
        result = _expand_row(row)
        assert result["timestamp"].iloc[0] == datetime(2025, 7, 1, 14, 0)
        assert result["timestamp"].iloc[59] == datetime(2025, 7, 1, 14, 59)

    def test_expand_preserves_tagindex(self):
        """展开后tagindex应保持不变"""
        from data_loader import _expand_row
        row = pd.Series({
            "DateDay": "2025-07-01",
            "DateHour": 0,
            "TagIndex": 455,
            **{f"T{i:02d}": 1.0 for i in range(1, 61)},
        })
        result = _expand_row(row)
        assert all(result["tagindex"] == 455)


class TestLoadTagMappingEdgeCases:
    """点位映射边界测试"""

    def test_mapping_values_are_dicts(self):
        """每个tagtable的值应该是字典"""
        mapping = load_tag_mapping()
        for table, indexes in mapping.items():
            assert isinstance(indexes, dict)

    def test_tagindexes_are_integers(self):
        """所有tagindex应该是整数"""
        mapping = load_tag_mapping()
        for table, indexes in mapping.items():
            for idx in indexes:
                assert isinstance(idx, int)

    def test_p1_flow_in_mapping(self):
        """P1流量点位(tagindex=455)应在映射中"""
        mapping = load_tag_mapping()
        assert 455 in mapping["ZHJY"]


class TestMarkInvalidValuesEdgeCases:
    """异常值标记边界测试"""

    def test_custom_markers(self):
        """支持自定义异常值标记"""
        df = pd.DataFrame({"value": [1.0, -8888.0, 3.0]})
        result = mark_invalid_values(df, custom_markers=[-8888.0])
        assert result.loc[1, "is_valid"] == False
        assert result.loc[0, "is_valid"] == True

    def test_near_match_tolerance(self):
        """-4.999应在容差范围内被标记为无效"""
        df = pd.DataFrame({"value": [1.0, -4.999, 3.0]})
        result = mark_invalid_values(df)
        assert result.loc[1, "is_valid"] == False

    def test_does_not_modify_original(self):
        """标记操作不应修改原始DataFrame"""
        df = pd.DataFrame({"value": [1.0, -4.999]})
        original_vals = df["value"].tolist()
        mark_invalid_values(df)
        assert df["value"].tolist() == original_vals
