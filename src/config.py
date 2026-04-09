"""
投矾智慧控制系统 - 点位映射与常量配置
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "原始数据")
EXCEL_PATH = os.path.join(BASE_DIR, "docs", "投矾系统的智慧诊断点位.xlsx")

# CSV 文件路径映射: tagtable -> [csv文件路径列表]
CSV_FILES = {
    "ZHJY": [os.path.join(DATA_DIR, "ZHJY__202201-202508(20251205).csv")],
    "ShuiZhi": [os.path.join(DATA_DIR, "ShuiZhi_202201-202508(20251205).csv")],
    "shuizhi": sorted([
        os.path.join(DATA_DIR, "ssts", f)
        for f in os.listdir(os.path.join(DATA_DIR, "ssts"))
        if f.endswith(".csv")
    ]),
    "flowpress": sorted([
        os.path.join(DATA_DIR, "流量数据", f)
        for f in os.listdir(os.path.join(DATA_DIR, "流量数据"))
        if f.endswith(".csv")
    ]),
}

# ==================== 点位映射 ====================
# 从 Excel 解析后构建的映射结构
# tagtable -> tagindex -> {name, device, tagname, tagtype}

# 投矾泵点位 (P1~P5)
PUMP_POINTS = {
    "P1": {
        "auto": {"tagindex": 453, "name": "T投矾新_1号数字泵_自动状态"},
        "error": {"tagindex": 454, "name": "T投矾新_1号数字泵_故障输入"},
        "flow": {"tagindex": 455, "name": "T投矾新_1号数字泵_实时投加流量"},
        "remote": {"tagindex": 456, "name": "T投矾新_1号数字泵_远控状态"},
        "tagtable": "ZHJY",
    },
    "P2": {
        "auto": {"tagindex": 457, "name": "T投矾新_2号数字泵_自动状态"},
        "error": {"tagindex": 458, "name": "T投矾新_2号数字泵_故障输入"},
        "flow": {"tagindex": 459, "name": "T投矾新_2号数字泵_实时投加流量"},
        "remote": {"tagindex": 460, "name": "T投矾新_2号数字泵_远控状态"},
        "tagtable": "ZHJY",
    },
    "P3": {
        "auto": {"tagindex": 461, "name": "T投矾新_3号数字泵_自动状态"},
        "error": {"tagindex": 462, "name": "T投矾新_3号数字泵_故障输入"},
        "flow": {"tagindex": 463, "name": "T投矾新_3号数字泵_实时投加流量"},
        "remote": {"tagindex": 464, "name": "T投矾新_3号数字泵_远控状态"},
        "tagtable": "ZHJY",
    },
    "P4": {
        "auto": {"tagindex": 465, "name": "T投矾新_4号数字泵_自动状态"},
        "error": {"tagindex": 466, "name": "T投矾新_4号数字泵_故障输入"},
        "flow": {"tagindex": 467, "name": "T投矾新_4号数字泵_实时投加流量"},
        "remote": {"tagindex": 468, "name": "T投矾新_4号数字泵_远控状态"},
        "tagtable": "ZHJY",
    },
    "P5": {
        "auto": {"tagindex": 469, "name": "T投矾新_5号数字泵_自动状态"},
        "error": {"tagindex": 470, "name": "T投矾新_5号数字泵_故障输入"},
        "flow": {"tagindex": 471, "name": "T投矾新_5号数字泵_实时投加流量"},
        "remote": {"tagindex": 472, "name": "T投矾新_5号数字泵_远控状态"},
        "tagtable": "ZHJY",
    },
}

# 传感器点位
SENSOR_POINTS = {
    # 电磁流量计
    "电磁流量计": {
        "tagtable": "flowpress",
        "sensors": {
            "J沉清叠池A池1#进水": {"tagindex": 6},
            "J沉清叠池A池2#进水": {"tagindex": 10},
            "J沉清叠池B池1#进水": {"tagindex": 244},
            "J沉清叠池B池2#进水": {"tagindex": 248},
        },
    },
    # 出水浊度仪
    "出水浊度仪": {
        "tagtable": "shuizhi",
        "sensors": {
            "沉淀池出水_1#浊度": {"tagindex": 4},
            "沉淀池出水_2#浊度": {"tagindex": 5},
            "沉淀池出水_3#浊度": {"tagindex": 66},
            "沉淀池出水_4#浊度": {"tagindex": 67},
        },
    },
    # 进水浊度仪
    "进水浊度仪": {
        "tagtable": "ShuiZhi",
        "sensors": {
            "1#反应池浊度": {"tagindex": 84},
            "2#反应池浊度": {"tagindex": 85},
            "3#反应池浊度": {"tagindex": 87},
            "4#反应池浊度": {"tagindex": 88},
        },
    },
    # 储液池液位计
    "储液池液位计": {
        "tagtable": "ZHJY",
        "sensors": {
            "T投矾新1号储液池液位": {"tagindex": 431},
            "T投矾新2号储液池液位": {"tagindex": 432},
            "T投矾新3号储液池液位": {"tagindex": 433},
        },
    },
    # 投加流量计（用于管道泄漏诊断）
    "投加流量计": {
        "tagtable": "ZHJY",
        "sensors": {
            "T投矾新1号投加流量计流量": {"tagindex": 434},
            "T投矾新2号投加流量计流量": {"tagindex": 435},
            "T投矾新3号投加流量计流量": {"tagindex": 436},
            "T投矾新4号投加流量计流量": {"tagindex": 437},
            "T投矾新5号投加流量计流量": {"tagindex": 438},
        },
    },
    # 阀门（每个阀门3个信号：关到位/开到位/故障汇总）
    "阀门": {
        "tagtable": "ZHJY",
        "sensors": {
            "T投矾新_1号出液阀_关到位": {"tagindex": 444},
            "T投矾新_1号出液阀_开到位": {"tagindex": 445},
            "T投矾新_1号出液阀_故障汇总": {"tagindex": 446},
            "T投矾新_2号出液阀_关到位": {"tagindex": 447},
            "T投矾新_2号出液阀_开到位": {"tagindex": 448},
            "T投矾新_2号出液阀_故障汇总": {"tagindex": 449},
            "T投矾新_3号出液阀_关到位": {"tagindex": 450},
            "T投矾新_3号出液阀_开到位": {"tagindex": 451},
            "T投矾新_3号出液阀_故障汇总": {"tagindex": 452},
        },
    },
}

# ==================== 诊断阈值配置 ====================

# 数据缺失阈值（分钟）- key=报警等级(1=最高, 3=最低)
ABSENCE_THRESHOLDS = {3: 10, 2: 30, 1: 60}

# 数据漂移阈值 - 按传感器类型分
# 值为与历史均值的相对偏差百分比（液位计除外，用绝对值cm）
DRIFT_THRESHOLDS = {
    "电磁流量计": {
        3: {"threshold": 0.20, "duration_min": 10},  # ±20%持续10min
        2: {"threshold": 0.40, "duration_min": 5},   # ±40%持续5min
        1: {"threshold": 0.50, "duration_min": 3},   # ±50%持续3min
    },
    "出水浊度仪": {
        3: {"threshold": 0.15, "duration_min": 10},
        2: {"threshold": 0.40, "duration_min": 5},
        1: {"threshold": 0.50, "duration_min": 3},
    },
    "进水浊度仪": {
        3: {"threshold": 0.15, "duration_min": 10},
        2: {"threshold": 0.40, "duration_min": 5},
        1: {"threshold": 0.50, "duration_min": 3},
    },
    "储液池液位计": {
        2: {"threshold_abs_cm": 10, "duration_min": 1},  # 绝对偏差>10cm
    },
}

# 数据异跳阈值
JUMP_THRESHOLDS = {
    "出水浊度仪": {"rate_per_min": 50.0},      # >50 NTU/min
    "进水浊度仪": {"normal_multiple": 3.0},     # >正常波动3倍
    "电磁流量计": {"normal_multiple": 3.0},     # >正常波动3倍
    "储液池液位计": {"change_cm_per_10min": 30}, # >30cm/10min
}

# ==================== 健康度评估配置 ====================

HEALTH_WEIGHTS = {
    "flow_deviation": 0.30,   # 流量偏差率
    "fault_frequency": 0.30,  # 故障频率
    "stability": 0.20,        # 运行稳定性
    "responsiveness": 0.20,   # 控制响应性
}

HEALTH_GRADES = {
    (90, 101): "优秀",
    (70, 90): "良好",
    (50, 70): "轻微异常",
    (0, 50): "异常",
}

# 报警等级名称
ALARM_LEVELS = {1: "一级", 2: "二级", 3: "三级"}

# 异常值标记（CSV中这些值代表无效数据）
INVALID_MARKERS = [-4.9991, -4.999, -9999]
