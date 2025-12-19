# -*- coding: utf-8 -*-
"""
Fuel consumption & range model with Willans-line engine approximation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class VehicleParams:
    def __init__(self):
        #inital parameter and coefficient
        self.CdA = 0.62
        self.Crr = 0.0105
        self.air_density = 1.2
        self.g = 9.8
        self.mass_car = 1450.0 #in kg 

        self.fuel_density = 0.745 #kg/L
        self.fuel_L_initial = 20.0 #L

        #
        self.accessory_power = 700.0 #W

        #value used to calculate fuel consumption rate, based on engine type, no need to change unless want to change type of car
        self.LHV = 43e6 #Lower Heating Value for gasoline fuel
        self.eta0 = 0.28 #Brake Thermal Efficiency
        self.idle_fuel_kgps = 0.00055#idle fuel consumptiuon rate

    #Willans model parameter
    def alpha(self):
        return 1.0 / (self.eta0 * self.LHV)

    def beta(self): #idle fuel consumptiuon rate
        return self.idle_fuel_kgps

    #Initial mass
    def fuel_mass_initial(self):
        return self.fuel_L_initial * self.fuel_density

    def mass_initial(self):
        return self.mass_car + self.fuel_mass_initial()

    #helper function during simulation
    def copy(self):
        p = VehicleParams()
        p.__dict__.update(self.__dict__)
        return p


#Calculate the power required for maintain the speed, 
def road_power(v, m, p, theta=0.0, v_wind=0.0):
    v_rel = max(v + v_wind, 0.0)

    P_aero = 0.5 * p.air_density * p.CdA * v_rel**3
    P_roll = p.Crr * m * p.g * v
    P_grade = m * p.g * np.sin(theta) * v

    return P_aero + P_roll + P_grade 


#Return power required
def engine_power(P_road, p):
    return (P_road + p.accessory_power)

#predicted mdot
def fuel_flow_willans(P_e, p):
    return max(p.alpha() * P_e + p.beta(), 0.0)


#run the code until the all the fuel runs out, if you want to limit the time, change t_cap_h
#using Euler method to simulate the solution
#dF/dt = -mdot  ->  F_{n+1} = F_n - mdot * dt
#where mdot is the fuel mass flow rate given by the Willans model.
#and h is 0.5s, store all the data into list "row" for further graphing
def simulate_until_empty_constant_speed(
    v_kmh, p, theta=0.0, v_wind=0.0, dt=0.5, t_cap_h=None
):
    v = v_kmh / 3.6
    F = p.fuel_mass_initial()
    m = p.mass_car + F

    t = 0.0
    s = 0.0
    rows = []

    t_cap = t_cap_h * 3600 if t_cap_h is not None else np.inf

    while F > 0 and t < t_cap:
        P_road = road_power(v, m, p, theta, v_wind)
        P_e = engine_power(P_road, p)
        mdot = fuel_flow_willans(P_e, p)

        dF = -mdot * dt
        F_next = F + dF

        if F_next < 0:
            frac = F / (mdot * dt) if mdot > 0 else 1.0
            dt_last = frac * dt
            t += dt_last
            s += v * dt_last
            F = 0.0
        else:
            t += dt
            s += v * dt
            F = F_next

        m = p.mass_car + F

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

        if t > 24 * 3600:
            break

    df = pd.DataFrame(rows)

    if not df.empty:
        df["t_h"] = df["t_s"] / 3600
        df["s_km"] = df["s_m"] / 1000

        total_fuel_L = (p.fuel_mass_initial() - df["F_kg"].iloc[-1]) / p.fuel_density
        total_km = df["s_km"].iloc[-1]

        df.attrs["total_fuel_L"] = total_fuel_L
        df.attrs["total_km"] = total_km
        df.attrs["L_per_100km"] = total_fuel_L / max(total_km, 1e-9) * 100

    return df


def sweep_speed_time_and_range(p, speeds_kmh, theta=0.0, v_wind=0.0, t_cap_h=None):
    records = []

    for v in speeds_kmh:
        df = simulate_until_empty_constant_speed(v, p, theta, v_wind, 0.5, t_cap_h)

        if df.empty:
            records.append(dict(speed_kmh=v,
                                time_to_empty_h=np.nan,
                                range_km=np.nan,
                                L_per_100km=np.nan))
        else:
            #find drivingrange, time when the fuel runs out and storage the result
            records.append(dict(
                speed_kmh=v,
                time_to_empty_h=df["t_h"].iloc[-1],
                range_km=df["s_km"].iloc[-1],
                L_per_100km=df.attrs["L_per_100km"]
            ))

    return pd.DataFrame(records)


def scenario_compare(p, speeds_kmh):
    return {
        #basecase: no grade, no wind
        "Basecase": sweep_speed_time_and_range(p, speeds_kmh),
        #constant uphill slope of 1 degree
        "1 degree ": sweep_speed_time_and_range(p, speeds_kmh, theta = np.deg2rad(1.0)),
        #Headwind 5 m/s
        "Headwind 5 m/s": sweep_speed_time_and_range(p, speeds_kmh, v_wind=5.0),
    }


#compute tornado sensitivity analysis, modify each coefficient by 10% and compute the difference between original value
def sensitivity_at_speed(p, base_speed_kmh=80.0):
    base_df = simulate_until_empty_constant_speed(base_speed_kmh, p)
    base_L100 = base_df.attrs["L_per_100km"]

    sens = []

    for name in ["CdA", "Crr", "mass_car", "accessory_power"]:
        p_hi = p.copy()
        p_lo = p.copy()

        setattr(p_hi, name, getattr(p_hi, name) * 1.10)
        setattr(p_lo, name, getattr(p_lo, name) * 0.90)

        for tag, pp in [("+10%", p_hi), ("-10%", p_lo)]:
            df = simulate_until_empty_constant_speed(base_speed_kmh, pp)
            sens.append(dict(
                parameter=f"{name} {tag}",
                L_per_100km=df.attrs["L_per_100km"] - base_L100
            ))

    return pd.DataFrame(sens)


def plot_range_vs_speed(df, title):
    plt.figure()
    plt.plot(df["speed_kmh"], df["range_km"], marker="o")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Range (km)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_time_vs_speed(df, title):
    plt.figure()
    plt.plot(df["speed_kmh"], df["time_to_empty_h"], marker="o")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Time to Empty (h)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()


def plot_L100_vs_speed_multi(scenarios):
    plt.figure()
    for name, df in scenarios.items():
        plt.plot(df["speed_kmh"], df["L_per_100km"], marker="o", label=name)
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Fuel (L/100 km)")
    plt.legend()
    plt.grid(True) 
    plt.tight_layout()


def plot_sensitivity_tornado(df, title="Sensitivity Tornado (±10% at 90 km/h)"):
    # df: columns = ["parameter", "L_per_100km"] where L_per_100km is delta vs base
    d = df.copy()
    d["abs_delta"] = np.abs(d["L_per_100km"])
    d = d.sort_values("abs_delta", ascending=True)

    plt.figure()
    plt.barh(d["parameter"], d["L_per_100km"])
    plt.xlabel("Δ L/100km vs Base")
    plt.title(title)
    plt.grid(True, axis="x")
    plt.tight_layout()


if __name__ == "__main__":
    p = VehicleParams()
    #speed range from 
    speeds = np.arange(40, 141, 10)

    sc = scenario_compare(p, speeds)
    sens = sensitivity_at_speed(p, base_speed_kmh=90.0)
    
    plot_range_vs_speed(sc["Basecase"], "Driving Range versus Cruising Speed under Fixed Fuel Conditions")
    plot_time_vs_speed(sc["Basecase"], "Deiving Time versus Cruising Speed under Fixed Fuel Conditions")
    plot_L100_vs_speed_multi(sc)
    plot_sensitivity_tornado(sens)

    plt.show()
