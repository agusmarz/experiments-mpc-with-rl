"""Eje A: función de valor V_psi(x) aprendida con iteración de valor ajustada,
usando el MPC como operador de Bellman, para usarla como costo terminal.

    V_psi(x) = x^T P x * exp(phi_theta(x / escala_x))

x^T P x (P de Riccati) es la parte fija: es el valor exacto del problema
linealizado, así que cerca del equilibrio ya es correcta. phi_theta es una MLP
que aprende la corrección no lineal; su última capa arranca en cero, así que
en la iteración 0 V_psi = x^T P x exactamente. (Si toda la red arrancara en
cero, las neuronas ocultas serían simétricas y no aprenderían.)

La corrección es multiplicativa y no aditiva (x^T P x + phi) a propósito: V va
de ~1 cerca del equilibrio a ~1e4 lejos, y un residuo aditivo con error de
cientos de unidades destruye el comportamiento local (probado: con N = 3
recuperaba menos estados que x^T P x). Con exp(phi) el error es relativo,
V_psi > 0 siempre y V_psi(0) = 0.

Iteración de valor ajustada (fitted value iteration)
    0. muestrear una lista fija de estados x_i
    1. con theta_j congelado, target y_i = J*_N(x_i; V_psi_j), el costo óptimo
       del MPC de horizonte N con V_psi_j como costo terminal. Con N = 1 es
       exactamente y_i = l(x_i, u*) + V_psi_j(x_i+1). Con N > 1 es la versión
       de N pasos, que propaga la información N veces más rápido por iteración.
    2. ajustar theta minimizando el MSE entre V_psi(x_i) e y_i
    3. volver a 1
Los targets se saturan en V_max: los estados que el MPC no puede recuperar
no tienen valor finito y sin tope crecerían sin cota iteración tras iteración.
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulador"))
from pendulo_env import ParametrosPlanta  # noqa: E402
from mpc_no_lineal import MPCNoLineal, riccati  # noqa: E402


class RedValor(nn.Module):
    """V_psi(x) = x^T P x * exp(MLP(x / escala_x)), con la última capa en cero."""

    def __init__(self, P, escala_x, ocultas=(64, 64)):
        super().__init__()
        self.register_buffer("P", torch.as_tensor(P, dtype=torch.float64))
        self.register_buffer("escala_x", torch.as_tensor(escala_x, dtype=torch.float64))

        capas, entrada = [], 4
        for h in ocultas:
            capas += [nn.Linear(entrada, h), nn.Tanh()]
            entrada = h
        ultima = nn.Linear(entrada, 1)
        nn.init.zeros_(ultima.weight)
        nn.init.zeros_(ultima.bias)
        self.mlp = nn.Sequential(*capas, ultima).double()

    def cuadratico(self, x):
        return torch.einsum("bi,ij,bj->b", x, self.P, x)

    def residuo(self, x):
        """phi_theta(x) = log(V_psi(x) / x^T P x), es lo que se ajusta."""
        return self.mlp(x / self.escala_x).squeeze(-1)

    def forward(self, x):
        return self.cuadratico(x) * torch.exp(self.residuo(x))

    def valor_numpy(self, x):
        with torch.no_grad():
            return self(torch.as_tensor(np.atleast_2d(x), dtype=torch.float64)).numpy()

    def a_casadi(self):
        """Copia los pesos actuales a una función x_N -> expresión de CasADi,
        lista para pasar como costo_terminal a MPCNoLineal."""
        P = ca.DM(self.P.numpy())
        escala = ca.DM(self.escala_x.numpy())
        lineales = [(ca.DM(c.weight.detach().numpy()), ca.DM(c.bias.detach().numpy()))
                    for c in self.mlp if isinstance(c, nn.Linear)]

        def costo_terminal(x):
            h = x / escala
            for k, (W, b) in enumerate(lineales):
                h = ca.mtimes(W, h) + b
                if k < len(lineales) - 1:
                    h = ca.tanh(h)
            return ca.mtimes([x.T, P, x]) * ca.exp(h)

        return costo_terminal


@dataclass
class ConfigIteracionValor:
    N_bellman: int = 1               # horizonte del MPC que hace de operador de Bellman
    n_estados: int = 2000
    iteraciones: int = 30
    epocas_ajuste: int = 300         # pasos de Adam por iteración (full batch)
    lr: float = 1e-3
    V_max: float = 2.0e4             # tope de los targets
    semilla: int = 0
    # Caja de muestreo de [x, x_d, theta, theta_d]; contiene el plano (theta0, x_d0)
    caja: np.ndarray = field(default_factory=lambda: np.array([0.45, 2.0, 0.8, 4.0]))
    ocultas: tuple = (64, 64)


def muestrear_estados(cfg: ConfigIteracionValor):
    rng = np.random.default_rng(cfg.semilla)
    return rng.uniform(-cfg.caja, cfg.caja, size=(cfg.n_estados, 4))


def targets_bellman(mpc: MPCNoLineal, estados, V_max):
    """y_i = min(J*_N(x_i), V_max); si Ipopt falla, V_max."""
    y = np.empty(len(estados))
    fallas = 0
    for i, x in enumerate(estados):
        mpc.reiniciar()  # cada estado es independiente, sin warm start cruzado
        mpc.solve(x)
        if mpc.ultimo_exito:
            y[i] = min(mpc.ultimo_costo, V_max)
        else:
            y[i] = V_max
            fallas += 1
    return y, fallas


def ajustar(red: RedValor, estados, y, cfg: ConfigIteracionValor, opt):
    """Regresión MSE en escala log: log V_psi(x) - log y = phi(x) - log(y / x^T P x)."""
    x = torch.as_tensor(estados, dtype=torch.float64)
    with torch.no_grad():
        objetivo = torch.log(torch.as_tensor(y).clamp_min(1e-9)) - torch.log(red.cuadratico(x))
    for _ in range(cfg.epocas_ajuste):
        opt.zero_grad()
        perdida = torch.mean((red.residuo(x) - objetivo) ** 2)
        perdida.backward()
        opt.step()
    return perdida.item()


def iteracion_valor(cfg: ConfigIteracionValor, p: ParametrosPlanta | None = None, verbose=True):
    """Entrena V_psi. Devuelve la red y un historial con el costo de aprendizaje."""
    p = p if p is not None else ParametrosPlanta()
    torch.manual_seed(cfg.semilla)
    P, _ = riccati(p)
    red = RedValor(P, escala_x=cfg.caja, ocultas=cfg.ocultas)
    opt = torch.optim.Adam(red.parameters(), lr=cfg.lr)
    estados = muestrear_estados(cfg)

    historial = []
    y_previo = None
    t0 = time.perf_counter()
    for j in range(cfg.iteraciones):
        t_it = time.perf_counter()
        # 1. Operador de Bellman con theta_j congelado (los pesos se copian a CasADi)
        mpc = MPCNoLineal(p, N=cfg.N_bellman, costo_terminal=red.a_casadi())
        y, fallas = targets_bellman(mpc, estados, cfg.V_max)
        t_targets = time.perf_counter() - t_it
        # 2. Ajuste supervisado hacia los targets
        perdida = ajustar(red, estados, y, cfg, opt)

        cambio = np.nan if y_previo is None else float(np.mean(np.abs(y - y_previo)) / np.mean(np.abs(y_previo)))
        y_previo = y
        reg = {
            "iteracion": j,
            "mse_log": perdida,
            "cambio_rel_targets": cambio,
            "frac_saturados": float(np.mean(y >= cfg.V_max)),
            "fallas_ipopt": fallas,
            "solves_mpc": len(estados),
            "t_targets_s": t_targets,
            "t_total_s": time.perf_counter() - t0,
        }
        historial.append(reg)
        if verbose:
            print(f"it {j:3d}  mse_log {perdida:.2e}  dy_rel {cambio:.4f}  "
                  f"sat {reg['frac_saturados']:.3f}  fallas {fallas}  "
                  f"t_targets {t_targets:.1f}s  t {reg['t_total_s']:.0f}s", flush=True)
    return red, historial


def guardar(red: RedValor, cfg: ConfigIteracionValor, historial, ruta):
    torch.save({"estado": red.state_dict(), "cfg": cfg.__dict__, "historial": historial}, ruta)


def cargar(ruta):
    datos = torch.load(ruta, weights_only=False)
    cfg = ConfigIteracionValor(**datos["cfg"])
    P = datos["estado"]["P"].numpy()
    red = RedValor(P, escala_x=cfg.caja, ocultas=cfg.ocultas)
    red.load_state_dict(datos["estado"])
    return red, cfg, datos["historial"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Iteración de valor ajustada con el MPC como operador de Bellman")
    ap.add_argument("--N", type=int, default=1, help="horizonte del operador de Bellman")
    ap.add_argument("--estados", type=int, default=2000)
    ap.add_argument("--iteraciones", type=int, default=30)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()

    cfg = ConfigIteracionValor(N_bellman=a.N, n_estados=a.estados, iteraciones=a.iteraciones, semilla=a.semilla)
    red, hist = iteracion_valor(cfg)
    salida = a.salida or str(Path(__file__).parent / f"V_psi_N{a.N}_s{a.semilla}.pt")
    guardar(red, cfg, hist, salida)
    print("guardado en", salida)
