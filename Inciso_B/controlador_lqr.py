"""Controlador LQR con la misma interfaz que `MPCNoLineal`, para entrenar más rápido.

En el entrenamiento del AG el MPC cumple dos funciones: da la acción de control y
da la predicción x̂[k+1] con la que se calcula el error que ve la red. Este
controlador las separa:

    acción      u = -K (x - [x_ref, 0, 0, 0]), recortada a ±F_max
    predicción  x̂[k+1] = RK4(x[k], u, θ̂), el mismo paso que usa el MPC

Con el costo terminal x'Px de Riccati y sin restricciones activas, el MPC y el LQR
con los mismos Q y R dan casi la misma acción cerca de la vertical, así que las
trayectorias (y la excitación que usa la red para estimar) son parecidas, a una
fracción del costo (~0,5 ms por paso contra ~10 ms del MPC con N = 12).

    adaptativo = False  K fija, calculada una vez con los parámetros nominales
    adaptativo = True   K recalculada con θ̂ en cada paso (DARE sobre la
                        linealización con θ̂), más parecido al MPC adaptativo
"""

import copy
import time

import casadi as ca
import numpy as np
import scipy.linalg as sl

from pendulo_env_incisoB import ParametrosPlanta, derivada
from mpc_no_lineal import _ParametrosSimbolicos, paso_rk4


def _funcion_jacobianos(p: ParametrosPlanta) -> ca.Function:
    """(A, B) continuos en el equilibrio como función de θ = [M, m, L, r], compilada una vez."""
    x = ca.MX.sym("x", 4)
    F = ca.MX.sym("F")
    theta = ca.MX.sym("theta", 4)
    f = ca.vertcat(*derivada(x, F, _ParametrosSimbolicos(p, theta), lib=ca))
    A = ca.substitute(ca.jacobian(f, x), ca.vertcat(x, F), ca.DM.zeros(5))
    B = ca.substitute(ca.jacobian(f, F), ca.vertcat(x, F), ca.DM.zeros(5))
    return ca.Function("jacobianos_theta", [theta], [A, B]).expand()


class ControladorLQR:
    """Misma interfaz que `MPCNoLineal` en lo que usa el evaluador: `N`,
    `ultimo_exito`, `reiniciar(theta_inicial)` y `solve(estado, theta, x_ref)`."""

    N = 0  # sin horizonte: x_ref[0] es la referencia del instante actual

    def __init__(self, p: ParametrosPlanta, adaptativo: bool = False, F_max: float | None = None):
        self.p = copy.deepcopy(p)
        self.adaptativo = bool(adaptativo)
        self.F_max = self.p.fuerza_max if F_max is None else float(F_max)
        self._Q = np.asarray(self.p.Q, dtype=float)
        self._R = np.array([[float(self.p.R)]])
        self._paso = paso_rk4(self.p, self.p.dt)
        self._jacobianos = _funcion_jacobianos(self.p)

        self.theta_nominal = np.array([self.p.M, self.p.m, self.p.L, self.p.r], dtype=float)
        self.K_nominal = self._ganancia(self.theta_nominal)
        if self.K_nominal is None:
            raise ValueError("No se pudo calcular la K del LQR con los parámetros nominales.")

        self.theta_actual = self.theta_nominal.copy()
        self.K = self.K_nominal.copy()
        self.tiempos_solve: list[float] = []
        self.ultimo_exito = True

    def _ganancia(self, theta: np.ndarray) -> np.ndarray | None:
        """K de la DARE sobre la linealización discreta (ZOH) con θ; None si falla."""
        try:
            A, B = (np.array(M, dtype=float) for M in self._jacobianos(theta))
            bloque = sl.expm(np.block([[A, B], [np.zeros((1, 5))]]) * self.p.dt)
            Ad, Bd = bloque[:4, :4], bloque[:4, 4:]
            P = sl.solve_discrete_are(Ad, Bd, self._Q, self._R)
            K = np.linalg.solve(self._R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        except (np.linalg.LinAlgError, ValueError):
            return None
        return K if np.all(np.isfinite(K)) else None

    @staticmethod
    def _theta_valido(theta: np.ndarray) -> bool:
        M, m, L, r = theta
        return bool(np.all(np.isfinite(theta)) and M > 1e-6 and m > 1e-6 and L > 1e-6 and r >= 0.0)

    def actualizar_parametros(self, theta) -> None:
        self.theta_actual = np.asarray(theta, dtype=float).reshape(4).copy()

    def obtener_parametros(self) -> np.ndarray:
        return self.theta_actual.copy()

    def reiniciar(self, theta_inicial=None) -> None:
        self.tiempos_solve = []
        self.ultimo_exito = True
        self.K = self.K_nominal.copy()
        if theta_inicial is not None:
            self.actualizar_parametros(theta_inicial)

    def solve(self, estado, theta=None, x_ref=None) -> tuple[float, np.ndarray]:
        """Devuelve (acción, estado predicho siguiente), como `MPCNoLineal.solve`.

        Si θ̂ es físicamente inválido (masas o largo <= 0, NaN) o la DARE no tiene
        solución, marca `ultimo_exito = False` para que el evaluador lo penalice
        igual que una infactibilidad del MPC.
        """
        inicio = time.perf_counter()
        estado = np.asarray(estado, dtype=float).reshape(4)
        if theta is not None:
            self.actualizar_parametros(theta)
        ref = 0.0 if x_ref is None else float(np.asarray(x_ref, dtype=float).reshape(-1)[0])

        if not self._theta_valido(self.theta_actual):
            self.ultimo_exito = False
            return 0.0, estado.copy()

        if self.adaptativo:
            K = self._ganancia(self.theta_actual)
            if K is None:
                self.ultimo_exito = False
                return 0.0, estado.copy()
            self.K = K

        desvio = estado - np.array([ref, 0.0, 0.0, 0.0])
        u = float(np.clip(-(self.K @ desvio)[0], -self.F_max, self.F_max))
        x_pred = np.asarray(self._paso(estado, u, self.theta_actual), dtype=float).reshape(4)

        self.ultimo_exito = bool(np.all(np.isfinite(x_pred)))
        self.tiempos_solve.append(time.perf_counter() - inicio)
        if not self.ultimo_exito:
            return 0.0, estado.copy()
        return u, x_pred
