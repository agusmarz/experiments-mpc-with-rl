"""Región de recuperación en el plano (theta0, x_d0) para un MPC de horizonte N
con distintos costos terminales: ninguno, x^T P x y V_psi aprendida.

Un estado inicial se cuenta como recuperado si en `pasos` pasos de 20 ms el
péndulo no se cae ni el carro sale del riel, y termina cerca del equilibrio.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulador"))
from pendulo_env import PenduloEnv, ParametrosPlanta  # noqa: E402
from mpc_no_lineal import MPCNoLineal, costo_terminal_cuadratico, riccati  # noqa: E402


def lazo_cerrado(mpc: MPCNoLineal, env: PenduloEnv, estado0, pasos=250):
    """Devuelve (recuperado, costo acumulado, tiempo medio de solve [s])."""
    estado, _ = env.reset(options={"estado": estado0})
    mpc.reiniciar()
    costo = 0.0
    for _ in range(pasos):
        F = mpc.solve(estado)
        estado, _, terminado, _, info = env.step([F])
        costo += info["costo"]
        if terminado:
            return False, costo, float(np.mean(mpc.tiempos_solve))
    cerca = abs(estado[2]) < 0.05 and abs(estado[0]) < 0.1
    return bool(cerca), costo, float(np.mean(mpc.tiempos_solve))


def barrer(mpc, thetas, velocidades, pasos=250, p=None):
    """Matriz booleana [len(velocidades), len(thetas)] de estados recuperados."""
    env = PenduloEnv(p)
    ok = np.zeros((len(velocidades), len(thetas)), dtype=bool)
    costos = np.full(ok.shape, np.nan)
    tiempos = []
    for i, v in enumerate(velocidades):
        for j, th in enumerate(thetas):
            ok[i, j], costos[i, j], t = lazo_cerrado(mpc, env, [0.0, v, th, 0.0], pasos)
            tiempos.append(t)
    return ok, costos, float(np.mean(tiempos))


if __name__ == "__main__":
    import argparse
    from valor_aprendido import cargar

    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--modelo", required=True, help=".pt de V_psi guardado por valor_aprendido.py")
    ap.add_argument("--puntos", type=int, default=9)
    a = ap.parse_args()

    p = ParametrosPlanta()
    P, _ = riccati(p)
    red, _, _ = cargar(a.modelo)
    thetas = np.linspace(-0.7, 0.7, a.puntos)
    velocidades = np.linspace(-1.5, 1.5, a.puntos)

    for nombre, vf in [("sin V_f", None), ("x'Px", costo_terminal_cuadratico(P)), ("V_psi", red.a_casadi())]:
        mpc = MPCNoLineal(p, N=a.N, costo_terminal=vf)
        ok, costos, t = barrer(mpc, thetas, velocidades, p=p)
        print(f"N={a.N:2d}  {nombre:8s} recuperados {ok.sum():3d}/{ok.size}  "
              f"costo medio (recuperados) {np.nanmean(np.where(ok, costos, np.nan)):9.1f}  solve {1e3 * t:.1f} ms")
        for fila in ok[::-1]:
            print("    " + "".join("#" if b else "." for b in fila))
