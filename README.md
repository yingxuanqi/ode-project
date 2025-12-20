# ODE-Based Vehicle Fuel Consumption & Range Optimization
This project models vehicle fuel consumption and achievable driving range under fixed fuel conditions using a simplified vehicle longitudinal dynamics model and an ODE-style fuel-mass depletion simulation. It explores:
- The existence of an optimal cruising speed that maximizes driving range (base case)
- The impact of wind and road grade on fuel consumption
- A local (±10%) sensitivity analysis at a selected cruising speed
- A comparison between a Willans-line fuel model and an non-linear BSFC model

## Files
- `main.py`  
  Main simulation using a Willans-line fuel flow approximation (fuel mass depletes over time via forward Euler). Produces:
  - Range vs speed (base case)
  - Time-to-empty vs speed (base case)
  - Fuel consumption (L/100 km) vs speed for multiple scenarios (wind/grade)
  - Tornado sensitivity bar chart (±10% parameter perturbations)

- `bsfc_compare.py`  
  Compares **nonlinear BSFC** and **(constant-BSFC-based) Willans-style** fuel consumption trends vs speed on flat ground (generates range/time/L100 plots).
