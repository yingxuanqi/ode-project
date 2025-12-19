import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

#compares the nonlinear bsfc and willans models to ascertain which one would suit analysis better

@dataclass
class VehicleParams: #general parameters being taken
    CdA: float = 0.62 
    Crr: float = 0.0105
    rho: float = 1.2
    g: float = 9.8
    m_vehicle: float = 1450.0 #assumed to be average vehicle mass
    m_payload: float = 0.0 #can be adjusted to see impact of loading
    eta_d: float = 0.92
    fuel_density: float = 0.745
    fuel_L_initial: float = 20.0
    bsfc_const: float = 260.0   #g/kWh
    mdot_idle: float = 0.00055  #kg/s 

    @property
    def fuel_mass_initial(self):
        return self.fuel_L_initial * self.fuel_density

def brake_power(v, m_f, p: VehicleParams):
    m_tot = p.m_vehicle + p.m_payload + m_f
    F_aero = 0.5 * p.rho * p.CdA * v**2
    F_roll = p.Crr * m_tot * p.g
    P_wheel = (F_aero + F_roll) * v 
    P_brake = P_wheel / p.eta_d
    return P_brake #grade power overlooked since we are focusing on a flat ground case 

def mdot_engine_constant_bsfc(m_f, v, p):
    P = brake_power(v, m_f, p)
    return (P * p.bsfc_const) / 3.6e6  #kg/s as mass flow

def bsfc_nonlinear(P):
    return 220 + 0.015 * (P / 1000 - 40)**2 #bsfc here is dependent on power output

def mdot_engine_nonlinear_bsfc(m_f, v, p):
    P = brake_power(v, m_f, p) 
    return (P * bsfc_nonlinear(P)) / 3.6e6 #based on nonlinear BSFC


def time_to_empty_manual(v_kmh, p, nonlinear=False, n_steps=1000): #for the nonlinear bsfc model
    v = v_kmh / 3.6  #m/s
    m0 = p.fuel_mass_initial

    mdot_eng = (lambda m: mdot_engine_nonlinear_bsfc(m, v, p)) if nonlinear \
               else (lambda m: mdot_engine_constant_bsfc(m, v, p))

    m_values = np.linspace(0, m0, n_steps + 1)
    dm = m_values[1] - m_values[0]

    integrand = 1.0 / (np.array([mdot_eng(m) for m in m_values]) + p.mdot_idle)
    t_empty = 0.5 * dm * (integrand[0] + 2 * np.sum(integrand[1:-1]) + integrand[-1])
    return t_empty  #in s

def time_to_empty_willans(v_kmh, p, n_steps=1000): #for willans with a constant bsfc
    v = v_kmh / 3.6
    m0 = p.fuel_mass_initial
    mdot_eng = lambda m: mdot_engine_constant_bsfc(m, v, p)
    m_values = np.linspace(0, m0, n_steps + 1)
    dm = m_values[1] - m_values[0]
    integrand = 1.0 / (np.array([mdot_eng(m) for m in m_values]) + p.mdot_idle)
    t_empty = 0.5 * dm * (integrand[0] + 2 * np.sum(integrand[1:-1]) + integrand[-1])
    return t_empty 


def fuel_consumption_L_per_100km(v_kmh, p, nonlinear=False, n_steps=1000):
    v = v_kmh / 3.6 
    mdot_eng = (lambda m: mdot_engine_nonlinear_bsfc(m, v, p)) if nonlinear \
               else (lambda m: mdot_engine_constant_bsfc(m, v, p))

    m_values = np.linspace(0, p.fuel_mass_initial, n_steps + 1)
    integrand = np.array([mdot_eng(m) for m in m_values]) + p.mdot_idle
    mdot_avg = np.mean(integrand)  

    #100 km at this speed
    t_100km = 100e3 / v  

    fuel_vol_L = (mdot_avg * t_100km) / p.fuel_density
    return fuel_vol_L

if __name__ == "__main__":
    p = VehicleParams()
    speeds = np.arange(40, 141, 10)  #note these are in km/h

    results_compare = []
    for v in speeds:
        #nonlinear BSFC model
        t_nl = time_to_empty_manual(v, p, nonlinear=True)
        rng_nl = (v / 3.6) * t_nl / 1000.0  
        L100_nl = fuel_consumption_L_per_100km(v, p, nonlinear=True)

        #willans model
        t_w = time_to_empty_willans(v, p)
        rng_w = (v / 3.6) * t_w / 1000.0 
        L100_w = fuel_consumption_L_per_100km(v, p, nonlinear=False)

        results_compare.append({
            "speed_kmh": v,
            "time_nl_h": t_nl / 3600.0,
            "range_nl_km": rng_nl,
            "L100_nl": L100_nl,
            "time_w_h": t_w / 3600.0,
            "range_w_km": rng_w,
            "L100_w": L100_w
        })

    speed_vals = [r["speed_kmh"] for r in results_compare]

    plt.figure(figsize=(8,5))
    plt.plot(speed_vals, [r["range_nl_km"] for r in results_compare], marker='o', label="Nonlinear BSFC")
    plt.plot(speed_vals, [r["range_w_km"] for r in results_compare], marker='s', linestyle='--', label="Willans")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Range (km)")
    plt.title("Range vs Speed: Nonlinear BSFC vs Willans Model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(speed_vals, [r["time_nl_h"] for r in results_compare], marker='o', label="Nonlinear BSFC")
    plt.plot(speed_vals, [r["time_w_h"] for r in results_compare], marker='s', linestyle='--', label="Willans")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Time to Empty (hours)")
    plt.title("Time to Empty vs Speed: Nonlinear vs Willans Model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(speed_vals, [r["L100_nl"] for r in results_compare], marker='o', label="Nonlinear BSFC")
    plt.plot(speed_vals, [r["L100_w"] for r in results_compare], marker='s', linestyle='--', label="Willans")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Fuel Consumption (L/100 km)")
    plt.title("Fuel Consumption vs Speed: Nonlinear vs Willans Model")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
