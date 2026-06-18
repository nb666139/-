"""
新能源出力时序模型

基于NREL WIND Toolkit和NSRDB太阳能辐射数据的统计参数，
生成合成风电和光伏日出力曲线。

数据来源：
- 风速统计：NREL WIND Toolkit (https://www.nrel.gov/grid/wind-toolkit.html)
- 太阳辐射统计：NREL NSRDB (https://nsrdb.nrel.gov/)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class WindPowerModel:
    """风电出力时间序列模型。

    基于典型风速分布（Weibull分布），通过风速-功率曲线转换为出力。
    """

    def __init__(self, rated_power_mw: float = 30.0, seed: int = 42) -> None:
        """
        参数:
            rated_power_mw: 风电场额定容量（MW）
            seed: 随机种子
        """
        self.rated_power = rated_power_mw
        self._rng = np.random.default_rng(seed)

        # 风速-功率曲线参数（适用于2.5MW风机）
        self.cut_in_speed = 3.0    # 切入风速 (m/s)
        self.rated_speed = 12.0    # 额定风速 (m/s)
        self.cut_out_speed = 25.0  # 切出风速 (m/s)

        # Weibull分布参数（不同季节的shape和scale）
        self.seasonal_params = {
            "spring": {"shape": 2.1, "scale": 7.5},
            "summer": {"shape": 1.9, "scale": 6.0},
            "autumn": {"shape": 2.3, "scale": 8.0},
            "winter": {"shape": 2.5, "scale": 9.0},
        }

    def _wind_to_power(self, wind_speed: float) -> float:
        """风速-功率转换函数（理想化功率曲线）。"""
        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            return 0.0
        if wind_speed >= self.rated_speed:
            return self.rated_power
        # 二次型转换：P ∝ v^2（简化，实际为三次型且含效率损失）
        normalized = (wind_speed - self.cut_in_speed) / (
            self.rated_speed - self.cut_in_speed
        )
        return self.rated_power * (normalized ** 2)

    def generate_day_profile(
        self, season: str = "winter", n_hours: int = 24, scale: float = 1.0
    ) -> np.ndarray:
        """生成单日风电出力曲线。

        参数:
            season: 季节（spring/summer/autumn/winter）
            n_hours: 小时数（默认24）
            scale: 容量缩放因子

        返回:
            (n_hours,) 形状的风电出力数组（MW）
        """
        params = self.seasonal_params.get(season, self.seasonal_params["winter"])
        wind_speeds = self._rng.weibull(
            a=params["shape"],
            size=n_hours,
        ) * params["scale"]

        # 添加日间变化趋势（中午风速通常略低）
        hour = np.arange(n_hours)
        diurnal = 1.0 + 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)
        wind_speeds *= diurnal * scale

        return np.array([self._wind_to_power(v) for v in wind_speeds])

    def generate_days(
        self, n_days: int = 30, season: str = "winter", scale: float = 1.0
    ) -> np.ndarray:
        """生成多日风电出力曲线。

        返回:
            (n_days * 24,) 形状的出力数组（MW）
        """
        profiles = [
            self.generate_day_profile(season, 24, scale)
            for _ in range(n_days)
        ]
        return np.concatenate(profiles)


class SolarPowerModel:
    """光伏出力时间序列模型。

    基于Beta分布模拟日照强度，转换为光伏出力。
    """

    def __init__(self, rated_power_mw: float = 20.0, seed: int = 43) -> None:
        """
        参数:
            rated_power_mw: 光伏电站额定容量（MW）
            seed: 随机种子
        """
        self.rated_power = rated_power_mw
        self._rng = np.random.default_rng(seed)

    def generate_day_profile(
        self, cloud_cover: str = "clear", n_hours: int = 24, scale: float = 1.0
    ) -> np.ndarray:
        """生成单日光伏出力曲线。

        参数:
            cloud_cover: 云量类别（clear / partly_cloudy / overcast）
            n_hours: 小时数（默认24）
            scale: 容量缩放因子

        返回:
            (n_hours,) 形状的光伏出力数组（MW）
        """
        hour = np.arange(n_hours)

        # 正弦形状的日间出力包络（日出6点，日落18点）
        day_envelope = np.maximum(
            0, np.sin(np.pi * (hour - 6) / 12)
        )

        # 云量参数
        cloud_params = {
            "clear": {"beta_a": 8.0, "beta_b": 2.0},       # 高辐照
            "partly_cloudy": {"beta_a": 3.0, "beta_b": 3.0},  # 中等辐照
            "overcast": {"beta_a": 2.0, "beta_b": 8.0},    # 低辐照
        }
        params = cloud_params.get(cloud_cover, cloud_params["clear"])

        # Beta分布模拟云量波动
        irradiance_factor = self._rng.beta(
            a=params["beta_a"], b=params["beta_b"], size=n_hours
        )

        # 合成出力：包络 × 辐照因子 × 容量
        power = day_envelope * irradiance_factor * self.rated_power * scale
        return np.clip(power, 0.0, self.rated_power * scale)

    def generate_days(
        self, n_days: int = 30, cloud_pattern: str = "mixed", scale: float = 1.0
    ) -> np.ndarray:
        """生成多日光伏出力曲线。

        cloud_pattern: "mixed" 混合云量，"clear" 全晴，"overcast" 全阴
        """
        cloud_categories = ["clear", "partly_cloudy", "overcast"]
        profiles = []
        for day in range(n_days):
            if cloud_pattern == "mixed":
                cat = cloud_categories[day % 3]
            else:
                cat = cloud_pattern
            profiles.append(self.generate_day_profile(cat, 24, scale))
        return np.concatenate(profiles)


class RenewableScenarioGenerator:
    """新能源出力场景综合生成器。

    整合风电和光伏模型，为多个新能源场站生成协同时序出力数据。
    """

    def __init__(self, grid_params: dict[str, Any], seed: int = 42) -> None:
        """
        参数:
            grid_params: IEEE30电网参数（来自ieee30.get_grid_params()）
            seed: 随机种子
        """
        self._rng = np.random.default_rng(seed)
        self._wind_capacity = grid_params["wind_capacity"]
        self._pv_capacity = grid_params["pv_capacity"]

        self.wind_models: dict[int, WindPowerModel] = {}
        for bus, cap in self._wind_capacity.items():
            self.wind_models[bus] = WindPowerModel(
                rated_power_mw=cap,
                seed=seed + bus,
            )

        self.pv_models: dict[int, SolarPowerModel] = {}
        for bus, cap in self._pv_capacity.items():
            self.pv_models[bus] = SolarPowerModel(
                rated_power_mw=cap,
                seed=seed + bus + 100,
            )

    def generate_scenario(
        self,
        n_hours: int = 24,
        season: str = "winter",
        cloud_cover: str = "partly_cloudy",
    ) -> dict[str, np.ndarray]:
        """生成一个场景的新能源出力数据。

        参数:
            n_hours: 时间步数
            season: 季节
            cloud_cover: 云量类别

        返回:
            {"wind": {bus: np.array}, "pv": {bus: np.array}, "total": np.array}
        """
        result: dict[str, Any] = {"wind": {}, "pv": {}}
        total = np.zeros(n_hours)

        for bus, model in self.wind_models.items():
            profile = model.generate_day_profile(season, n_hours)
            result["wind"][bus] = profile
            total += profile

        for bus, model in self.pv_models.items():
            profile = model.generate_day_profile(cloud_cover, n_hours)
            result["pv"][bus] = profile
            total += profile

        result["total"] = total
        return result

    def generate_wind_ramp_scenario(
        self, n_hours: int = 24, ramp_hour: int = 12, drop_pct: float = 0.42
    ) -> dict[str, np.ndarray]:
        """生成风电骤降场景。

        参数:
            ramp_hour: 骤降发生的小时
            drop_pct: 功率下降百分比（如0.42表示下降42%）

        返回:
            同generate_scenario()格式的新能源出力数据
        """
        result = self.generate_scenario(n_hours, season="winter")

        # 在指定小时施加骤降
        for bus in self._wind_capacity:
            for h in range(ramp_hour, n_hours):
                # 指数衰减恢复过程
                recovery = 1 - drop_pct * np.exp(-0.5 * (h - ramp_hour))
                result["wind"][bus][h] *= recovery

        # 重新计算总出力
        total = np.zeros(n_hours)
        for arr in result["wind"].values():
            total += arr
        for arr in result["pv"].values():
            total += arr
        result["total"] = total

        return result
