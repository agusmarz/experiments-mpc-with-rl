# Eje A: MPC con costo terminal aprendido sobre péndulo invertido

Trabajo Práctico Integrador de Control Predictivo. El eje A pregunta si, poniendo al final del horizonte del MPC una buena estimación del costo que falta (Riccati o una V aprendida con iteración de valor), se puede acortar el horizonte sin perder región de recuperación.

## Estructura

```
simulador/
  pendulo_env.py     Planta no lineal carro-péndulo como env de Gymnasium (PenduloGazebo-v0)
  mpc_no_lineal.py   MPC no lineal en CasADi, RK4 a dt = 0,02 e Ipopt con warm start
mpc_no_lineal.ipynb  Pruebas en lazo cerrado: LQR vs MPC sin V_f vs MPC con x'Px
```

`pendulo_env.py` es el mismo archivo que escribe `planta_pendulo.ipynb`. `mpc_no_lineal.py` sigue la estructura de `inverted_pendulum_control/mpc.py` del repo de ROS 2.

## Uso

```bash
python -m venv .venv
.venv/Scripts/activate      # en Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook mpc_no_lineal.ipynb
```

```python
import sys; sys.path.insert(0, "simulador")
import gymnasium as gym
import pendulo_env
from mpc_no_lineal import MPCNoLineal, riccati, costo_terminal_cuadratico

env = gym.make("PenduloGazebo-v0")
p = env.unwrapped.p
P, K = riccati(p)
mpc = MPCNoLineal(p, N=23, costo_terminal=costo_terminal_cuadratico(P))

estado, _ = env.reset(options={"estado": [0, 0, 0.5, 0]})
F = mpc.solve(estado)
```

`costo_terminal` es cualquier función `x_N -> expresión de CasADi`, que es lo único que cambia entre métodos. `conjunto_terminal=(P, alpha)` agrega `x_N ∈ Ω`.

## Estado

- [x] MPC base no lineal (RK4 + Ipopt, riel blando, costo terminal enchufable)
- [ ] Contraparte clásica: α de Ω
- [ ] Script de la región de recuperación
- [ ] Vψ con iteración de valor
- [ ] Híbrido
- [ ] Validación en Gazebo
