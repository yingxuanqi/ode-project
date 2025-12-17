# -*- coding: utf-8 -*-
"""
Fuel consumption & range model with Willans-line engine approximation.

Run directly to generate CSV outputs to /mnt/data and example plots.
"""
# 整体说明：这是一个用 Willans 线性模型近似发动机效率的
# 车辆燃油消耗 & 续航里程仿真代码。


import numpy as np      # 数值计算库
import pandas as pd     # 数据表处理库
from dataclasses import dataclass, asdict   # 用 dataclass 简化参数类写法
from typing import Dict  # 类型注解
import matplotlib.pyplot as plt  # 画图


@dataclass
class VehicleParams:
    # 车辆及环境参数的“容器类”（默认值都已经写好）
    CdA= 0.62            # 空气阻力系数 * 正面面积 (m^2)
    Crr= 0.0105          # 滚动阻力系数（无量纲）
    air_density= 1.2     # 空气密度 (kg/m^3)
    g= 9.8               # 重力加速度 (m/s^2)
    mass_kerb= 1450.0    # 整备质量，车本身质量不含油 (kg)
    fuel_density= 0.745  # 燃油密度 (kg/L)
    fuel_L_initial= 20.0 # 初始油量 (L)
    driveline_eff= 0.92  # 传动系统效率（发动机输出到轮上）
    accessory_power= 700.0  # 附件功率，空调、发电机等持续消耗 (W)
    LHV= 43e6            # 燃油低位发热值 (J/kg)
    eta0 = 0.28          # 发动机等效效率（用于 Willans 模型）
    idle_fuel_kgps = 0.00055  # 怠速时燃油消耗率 (kg/s)

    @property
    def alpha(self) -> float:
        # Willans 模型中的斜率：每单位发动机功率对应的燃油流量
        # α = 1 / (效率 * 发热值)
        return 1.0 / (self.eta0 * self.LHV)

    @property
    def beta(self) -> float:
        # Willans 模型中的截距：怠速燃油消耗（即功率为 0 时的流量）
        return self.idle_fuel_kgps

    @property
    def fuel_mass_initial(self) -> float:
        # 初始油质量 = 体积 * 密度 (kg)
        return self.fuel_L_initial * self.fuel_density

    @property
    def mass_initial(self) -> float:
        # 初始总质量 = 车身质量 + 燃油质量 (kg)
        return self.mass_kerb + self.fuel_mass_initial


def road_power(
        v: float,           # 车速 (m/s)
        m: float,           # 当前总质量 (kg)
        p: VehicleParams,   # 参数对象
        theta: float = 0.0, # 道路坡度角（弧度，正表示上坡）
        v_wind: float = 0.0,# 逆风（>0 表示迎风，和车同向加速空气相对速度）
        a: float = 0.0      # 加速度 (m/s^2)，当前模型中通常取 0
    ) -> float:
    # 计算在给定速度下克服空气阻力、滚阻、坡度和加速所需的轮上功率 (W)

    v_rel = max(v + v_wind, 0.0)          # 车速相对空气的速度（不能为负）
    P_aero = 0.5 * p.air_density * p.CdA * (v_rel ** 3)  # 空气阻力功率 ~ v^3
    P_roll = p.Crr * m * p.g * v         # 滚阻功率 ~ Crr * N * v
    P_grade = m * p.g * np.sin(theta) * v # 坡度功率：沿坡度方向的重力分量 * 速度
    P_inert = m * a * v                  # 惯性功率：加速时 F = m a，对应功率 = F v
    return P_aero + P_roll + P_grade + P_inert  # 总轮上功率


def engine_power(P_road: float, p: VehicleParams) -> float:
    # 计算发动机输出功率：
    # 轮上功率 + 附件功率，再除以传动效率
    return (P_road + p.accessory_power) / p.driveline_eff


def fuel_flow_willans(P_e: float, p: VehicleParams) -> float:
    # Willans 线性模型：燃油质量流量 = α * 发动机功率 + β
    # 结果不能为负，因此用 max( ..., 0.0 )
    return max(p.alpha * P_e + p.beta, 0.0)


def simulate_until_empty_constant_speed(
        v_kmh: float,           # 车速 (km/h)
        p: VehicleParams,       # 参数对象
        theta: float = 0.0,     # 坡度
        v_wind: float = 0.0,    # 风速
        dt: float = 0.5,        # 时间步长 (s)
        t_cap_h: float = None   # 可选：时间上限（小时），例如模拟最多开几小时
    ) -> pd.DataFrame:
    # 在给定车速下，进行时间步进仿真，直到油耗尽或者达到时间上限。
    # 返回一个 DataFrame，记录每个时间步的状态，并在 attrs 里写入
    # 总行驶里程、总耗油量和 L/100km。

    v = v_kmh / 3.6                 # km/h → m/s
    F = p.fuel_mass_initial         # 当前油质量 (kg)
    m = p.mass_kerb + F             # 总质量 = 车身 + 燃油
    t = 0.0                         # 当前时间 (s)
    s = 0.0                         # 当前路程 (m)
    rows = []                       # 用于存每一步的结果
    t_cap = (t_cap_h * 3600.0) if (t_cap_h is not None) else np.inf  # 时间上限（秒）

    # 主循环：每一小步更新油量和位置，直到油用完或时间超限
    while F > 0 and t < t_cap:
        # 当前速度和质量下的轮上功率
        P_road = road_power(v=v, m=m, p=p, theta=theta, v_wind=v_wind, a=0.0)
        # 发动机输出功率
        P_e = engine_power(P_road, p)
        # 根据 Willans 模型计算燃油流量 (kg/s)
        mdot = fuel_flow_willans(P_e, p)

        # 这一时间步内油量变化：dF = - mdot * dt
        dF = -mdot * dt
        F_next = F + dF             # 更新后油量

        if F_next < 0:
            # 如果这一小步会把油耗完，就只走一部分时间步
            # frac = 当前油量 / 这一时间步要消耗的油量
            frac = F / (mdot * dt) if mdot > 0 else 1.0
            dt_last = frac * dt     # 实际需要的时间 < dt
            t += dt_last            # 更新时间
            s += v * dt_last        # 路程增加
            F = 0.0                 # 油正式耗尽
            m = p.mass_kerb + F     # 总质量回到空油状态
            # 记录最后一步数据并跳出循环
            rows.append(dict(
                t_s=t,
                s_m=s,
                v_mps=v,
                F_kg=F,
                m_kg=m,
                P_road_W=P_road,
                P_e_W=P_e,
                mdot_kgps=mdot
            ))
            break

        # 如果油还没耗尽，正常推进一步
        t += dt
        s += v * dt
        F = F_next
        m = p.mass_kerb + F

        # 记录这一时间步的状态
        rows.append(dict(
            t_s=t,
            s_m=s,
            v_mps=v,
            F_kg=F,
            m_kg=m,
            P_road_W=P_road,
            P_e_W=P_e,
            mdot_kgps=mdot
        ))

        # 安全限制：如果跑超过 24 小时（理论上不太现实）就强制停止
        if t > 24 * 3600:
            break

    # 把记录转成 DataFrame
    df = pd.DataFrame(rows)
    if not df.empty:
        # 额外加上小时 & 公里单位的列，方便后面分析和画图
        df["t_h"] = df["t_s"] / 3600.0
        df["s_km"] = df["s_m"] / 1000.0

        # 利用 attrs 存储一些汇总信息（总油耗、总里程、L/100km）
        total_fuel_L = (p.fuel_mass_initial - df["F_kg"].iloc[-1]) / p.fuel_density
        total_km = df["s_km"].iloc[-1] if not df["s_km"].empty else 0.0
        df.attrs["total_fuel_L"] = total_fuel_L
        df.attrs["total_km"] = total_km
        df.attrs["L_per_100km"] = (total_fuel_L / max(total_km, 1e-9)) * 100.0

    return df


def sweep_speed_time_and_range(
        p: VehicleParams,
        speeds_kmh: np.ndarray,  # 一组要扫描的速度（km/h）
        theta: float = 0.0,
        v_wind: float = 0.0,
        t_cap_h: float = None
    ) -> pd.DataFrame:
    # 对一系列车速做仿真，收集每个车速下：
    #   - time_to_empty_h：油耗完需要的时间
    #   - range_km：在这一速下还能跑多远
    #   - L_per_100km：油耗
    records = []
    for v_kmh in speeds_kmh:
        df = simulate_until_empty_constant_speed(
            v_kmh, p, theta=theta, v_wind=v_wind, dt=0.5, t_cap_h=t_cap_h
        )
        if df.empty:
            # 如果仿真失败/异常，写 NaN
            records.append(dict(
                speed_kmh=v_kmh,
                time_to_empty_h=np.nan,
                range_km=np.nan,
                L_per_100km=np.nan
            ))
            continue

        # 从 df 的最后一行和 attrs 中取 summary 量
        records.append(dict(
            speed_kmh=v_kmh,
            time_to_empty_h=df["t_h"].iloc[-1],
            range_km=df["s_km"].iloc[-1],
            L_per_100km=df.attrs["L_per_100km"]
        ))

    # 返回一个速度扫描结果表
    return pd.DataFrame.from_records(records)


def scenario_compare(p: VehicleParams, speeds_kmh: np.ndarray) -> Dict[str, pd.DataFrame]:
    # 调用 sweep_speed_time_and_range，分别对三种场景做扫描：
    # 1. 平路（无风）
    # 2. 2% 上坡（theta = arctan(0.02)）
    # 3. 5 m/s 逆风
    # 返回一个 dict：键是场景名字，值是 DataFrame 结果
    return {
        "Flat": sweep_speed_time_and_range(p, speeds_kmh, theta=0.0, v_wind=0.0),
        "Grade 2pct": sweep_speed_time_and_range(p, speeds_kmh, theta=np.arctan(0.02), v_wind=0.0),
        "Headwind 5 mps": sweep_speed_time_and_range(p, speeds_kmh, theta=0.0, v_wind=5.0),
    }


def sensitivity_at_speed(p: VehicleParams, base_speed_kmh: float = 90.0) -> pd.DataFrame:
    # 在某个基准速度（默认 90 km/h）下做参数灵敏度分析：
    # 把 CdA, Crr, 质量, 附件功率 各自 ±10%，看 L/100km 变化多少

    # 先算基准情况的 L/100km（即不改参数）
    base = simulate_until_empty_constant_speed(base_speed_kmh, p, dt=0.5)
    base_L100 = base.attrs["L_per_100km"] if base is not None and not base.empty else np.nan

    sens = []

    # 内部小函数：给定修改后的参数对象，返回对应 L/100km
    def eval_L100(pmod: VehicleParams) -> float:
        df = simulate_until_empty_constant_speed(base_speed_kmh, pmod, dt=0.5)
        return df.attrs["L_per_100km"] if df is not None and not df.empty else np.nan

    # 对 4 个参数逐个做 ±10% 改动
    for name in ["CdA", "Crr", "mass_kerb", "accessory_power"]:
        # 复制一个参数对象（用 asdict 把 dataclass 转成 dict 再新建）
        p_hi = VehicleParams(**asdict(p))
        p_lo = VehicleParams(**asdict(p))

        # 放大 10%、缩小 10%
        setattr(p_hi, name, getattr(p_hi, name) * 1.10)
        setattr(p_lo, name, getattr(p_lo, name) * 0.90)

        # 分别计算 L/100km
        sens.append({"parameter": name + " +10%", "L_per_100km": eval_L100(p_hi)})
        sens.append({"parameter": name + " -10%", "L_per_100km": eval_L100(p_lo)})

    # 把结果整理成 DataFrame，并附上与基准的差值
    df = pd.DataFrame(sens)
    df["delta_vs_base"] = df["L_per_100km"] - base_L100
    df.attrs["base_L100"] = base_L100
    return df


def plot_range_vs_speed(df: pd.DataFrame, title: str):
    # 画“续航里程 vs 速度”的曲线
    plt.figure()
    plt.plot(df["speed_kmh"], df["range_km"], marker="o")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Range to Empty (km)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_time_vs_speed(df: pd.DataFrame, title: str):
    # 画“油耗完时间 vs 速度”的曲线
    plt.figure()
    plt.plot(df["speed_kmh"], df["time_to_empty_h"], marker="o")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Time to Empty (h)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_L100_vs_speed_multi(scenarios: Dict[str, pd.DataFrame]):
    # 在同一张图比较多个场景下的“油耗 vs 速度”
    plt.figure()
    for name, d in scenarios.items():
        plt.plot(d["speed_kmh"], d["L_per_100km"], marker="o", label=name)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Fuel (L/100 km)")
    plt.title("Fuel Consumption vs Speed (Scenario Comparison)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_sensitivity_tornado(df: pd.DataFrame):
    # 按 |delta_vs_base| 从小到大排序，画“龙卷风图”水平条形图
    d = df.sort_values("delta_vs_base", key=lambda x: np.abs(x), ascending=True)
    plt.figure()
    plt.barh(d["parameter"], d["delta_vs_base"])
    plt.xlabel("Δ L/100km vs Base (±10%)")
    plt.title(f"Sensitivity at 90 km/h (Base {df.attrs.get('base_L100', np.nan):.2f} L/100km)")
    plt.tight_layout()


if __name__ == "__main__":
    # 直接运行脚本时的主流程：

    p = VehicleParams()              # 初始化车辆参数
    speeds = np.arange(40, 141, 10)  # 速度从 40 到 140 km/h，每隔 10 km/h

    # 计算三种场景的速度扫描结果
    sc = scenario_compare(p, speeds)

    # 画平路情况下：续航 vs 速度
    plot_range_vs_speed(sc["Flat"], "Range vs Speed (Flat)")
    # 画平路情况下：时间 vs 速度
    plot_time_vs_speed(sc["Flat"], "Time to Empty vs Speed (Flat)")
    # 画三种场景的油耗对比
    plot_L100_vs_speed_multi(sc)

    # 做 90 km/h 的灵敏度分析并画龙卷风图
    sens = sensitivity_at_speed(p, base_speed_kmh=90.0)
    plot_sensitivity_tornado(sens)

    # 显示所有图
    plt.show()
