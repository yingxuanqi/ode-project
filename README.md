# ODE-Based Vehicle Fuel Consumption & Range Optimization
This project models vehicle fuel consumption and achievable driving range under fixed fuel conditions using a simplified vehicle longitudinal dynamics model and an ODE-style fuel-mass depletion simulation. It explores:
- The existence of an optimal cruising speed that maximizes driving range (base case)
- The impact of wind and road grade on fuel consumption
- A local (±10%) sensitivity analysis at a selected cruising speed
- A comparison between a Willans-line fuel model and an non-linear BSFC model

## Files
- `main.py`  
  Main simulation using a Willans-line fuel flow approximation (Forward Euler method). Produces:
  - Range vs speed (base case)
  - Time-to-empty vs speed (not used in the report but worth to have)
  - Fuel consumption (L/100 km) vs speed for multiple scenarios (under wind, grade, and combination of wind and grade)
  - Tornado sensitivity bar chart (±10% parameter perturbations at optical cruising speed)

- `bsfc_compare.py`  
  Compares **nonlinear BSFC** and **Willans line model** fuel consumption trends vs speed on flat ground (generates range/time/L100 plots).
  
## Requirements
- Python 3.9+ recommended
- Packages:
  - numpy
  - pandas (used in `main.py`)
  - matplotlib

Install:
```bash
pip install numpy pandas matplotlib
```
## How to Run
- Main simulation
  ```bash
  python main.py
  ```
  This will open matplotlib windosw with several figures
- compare nonlinear BSFC and Villans
  ```bash
  python bsfc_compare.py
  ```
  This will generate three plots:
  - Drive range vs. speed when fuel runs out
  - Time to empty vs. speed
  - Fuel consumption (L/100km) vs speed
## Parameter You may like to change to test the program
-`main.py`
  - In class VehicleParams:
    - fuel_L_initial (default 20L)
    - mass_car (default 1450 kg)
    - CdA (default Cd is 0.62, (assumed reference area is 1 m^2))
    - Cdrr (default 0.01)
    - accessory_power(default 700W, anything between 300-1000 is reasonable)
    - eta0, LHV (default assume gasoline)
  - In __name__ == "__main__":
    - speeds = np.arange(40, 141, 10) (first value is start, second is end, third is step, this is used for testing different range of speed)
  - In sensitivity_at_speed(p, base_speed_kmh=80.0):
    - base_speed_kmh is the optimal speed, can change to other speed to compute local sensitivity analysis
  
