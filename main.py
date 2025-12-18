import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Vehicle:
    def __init__(self):
        self.mass_kerb = 1450.0          
        self.CdA = 0.62                  
        self.Crr = 0.0105                
        self.driveline_eff = 0.92        
        self.accessory_power = 700.0    
        self.fuel_density = 0.745        
        self.idle_fuel_kgps = 0.00055    
        self.air_density = 1.2           

g = 9.80665
BSFC_CONST = 260.0 

def bsfc_g_per_kwh_from_power(p_kw):
    a, x0, b = 260.0, 25.0, 0.10
    bsfc = a + b * (p_kw - x0) ** 2 / max(x0, 1.0)
    return np.clip(bsfc, 240.0, 420.0)


def wheel_power_constant_v(v, m_total, CdA, Crr, rho, grade=0.0):
    F_aero = 0.5 * rho * CdA * v**2
    N = m_total * g * np.cos(np.arctan(grade))
    F_roll = Crr * N
    F_grade = m_total * g * np.sin(np.arctan(grade))
    return (F_aero + F_roll + F_grade) * v

def fuel_rate_from_Pwheel(P_wheel, veh):
    P_brake = P_wheel / veh.driveline_eff + veh.accessory_power
    p_kw = np.maximum(P_brake, 0.0) / 1000.0

    bsfc = bsfc_g_per_kwh_from_power(p_kw)
    mdot = (p_kw * bsfc) / (3600.0 * 1000.0)
    return np.maximum(mdot, veh.idle_fuel_kgps)



def simulate_timecap_20L(veh, FUEL_L, PAYLOAD_KG, T_HOURS):
    fuel_mass_init = FUEL_L * veh.fuel_density
    avg_total_mass = veh.mass_kerb + PAYLOAD_KG + 0.5 * fuel_mass_init

    speeds_kmh = np.arange(1.0, 161.0, 1.0)
    v = speeds_kmh / 3.6

    P_wheel = wheel_power_constant_v(v, avg_total_mass, veh.CdA, veh.Crr, veh.air_density, 0.0)
    mdot = fuel_rate_from_Pwheel(P_wheel, veh)
    t_empty_h = (fuel_mass_init / mdot) / 3600.0
    range_km = v * (t_empty_h * 3600.0) / 1000.0


    dist_time_cap_km = v * (T_HOURS * 3600.0) / 1000.0
    achievable_km = np.minimum(range_km, dist_time_cap_km)

    df = pd.DataFrame({
        "speed_kmh": speeds_kmh,
        "time_to_empty_h": t_empty_h,
        "range_20L_km": range_km,
        "distance_with_timecap_km": achievable_km,
        "L_per_100km": FUEL_L / np.maximum(range_km, 1e-9) * 100.0
    }).round(3)

    best_idx = int(np.argmax(achievable_km))
    best_speed = speeds_kmh[best_idx]
    best_dist = achievable_km[best_idx]

    return df, best_speed, best_dist, T_HOURS


if __name__ == "__main__":
    veh = Vehicle()
    df, best_speed, best_dist, T_HOURS = simulate_timecap_20L(veh, FUEL_L=200, PAYLOAD_KG=1200000.0, T_HOURS=8.0)


    print(f"best const speed：{best_speed:.1f} km/h")
    print(f"max dis in {T_HOURS:.1f} hrs:{best_dist:.1f} km")
    print(f"20 L fuel can hold approximately {df.loc[df['speed_kmh']==best_speed,'time_to_empty_h'].values[0]:.2f} hrs")


    plt.figure()
    plt.plot(df["speed_kmh"], df["distance_with_timecap_km"])
    plt.scatter([best_speed], [best_dist], color="red")
    plt.xlabel("speed (km/h)")
    plt.ylabel(f"{T_HOURS:.1f} (km)")
    plt.title(f"20L of fuel in 3 hour's best speed")
    plt.show()


    plt.figure()
    plt.plot(df["speed_kmh"], df["time_to_empty_h"])
    plt.axhline(T_HOURS, color="red", linestyle="--")
    plt.xlabel("speed (km/h)")
    plt.ylabel("Fuel exhaustion time (hrs)")
    plt.title("Fuel exhaustion time vs. speed")
    plt.show()
