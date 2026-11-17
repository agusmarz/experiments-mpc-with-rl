import time

import casadi as ca
import numpy as np
import scipy.linalg as sl

from pendulo_env import ParametrosPlanta, derivada


def costo_etapa(x, u, Q, R):
    return ca.mtimes([x.T, Q, x]) + R * u**2


def costo_terminal_cuadratico(P):
    """V_f(x) = x^T P x, para enchufar como costo terminal del MPC."""
    P = ca.DM(P)
    return lambda x: ca.mtimes([x.T, P, x])


def paso_rk4(p, dt):
    """Función de CasADi x[k+1] = RK4(x[k], F[k]) con F constante durante dt.

    Usa derivada(..., lib=casadi), o sea exactamente la misma dinámica que el env.
    """
    x = ca.MX.sym("x", 4)
    F = ca.MX.sym("F")
    f = lambda estado, fuerza: ca.vertcat(*derivada(estado, fuerza, p, lib=ca))

    k1 = f(x, F)
    k2 = f(x + dt / 2 * k1, F)
    k3 = f(x + dt / 2 * k2, F)
    k4 = f(x + dt * k3, F)
    x_siguiente = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("paso_rk4", [x, F], [x_siguiente])


def linealizar_discreto(p):
    """(Ad, Bd) en el equilibrio con retención de orden cero, para Riccati."""
    x = ca.MX.sym("x", 4)
    F = ca.MX.sym("F")
    f = ca.vertcat(*derivada(x, F, p, lib=ca))
    jac = ca.Function("jacobianos", [x, F], [ca.jacobian(f, x), ca.jacobian(f, F)])
    A, B = (np.array(M) for M in jac(np.zeros(4), 0.0))
    bloque = sl.expm(np.block([[A, B], [np.zeros((1, 5))]]) * p.dt)
    return bloque[:4, :4], bloque[:4, 4:]


def riccati(p):
    """P y K de la DARE sobre la linealización, con u = -K x."""
    Ad, Bd = linealizar_discreto(p)
    R = np.array([[p.R]])
    P = sl.solve_discrete_are(Ad, Bd, p.Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    return P, K


class MPCNoLineal:
    """MPC no lineal sobre el modelo del env, discretizado con RK4.

    El problema de optimización se arma una sola vez; cada llamada a solve()
    solo actualiza el parámetro del estado inicial y lo vuelve a resolver
    con Ipopt, arrancando desde la solución anterior desplazada (warm start).

    Restricciones
        |F_k| <= F_max                 dura
        |x_k| <= x_max - margen_riel + holgura_k
                                       blanda, holgura_k >= 0 muy penalizada,
                                       así el solver nunca da infactible
    margen_riel cubre la diferencia entre RK4 a 20 ms y el Euler semi
    implícito a 1 ms del env: sin margen el MPC apoya el carro justo en
    x_max y la planta real se pasa por décimas de mm.
    El costo terminal V_f(x_N) es una función que se enchufa (recibe x_N de
    CasADi y devuelve una expresión escalar) y es lo único que cambia entre
    métodos. Con None no hay costo terminal.
    """

    def __init__(self, p: ParametrosPlanta, N, costo_terminal=None, F_max=None,
                 peso_holgura=1e4, margen_riel=0.01, conjunto_terminal=None):
        self.p = p
        self.N = N
        self.F_max = p.fuerza_max if F_max is None else F_max
        nx, nu = 4, 1
        Q = ca.DM(p.Q)
        f_discreta = paso_rk4(p, p.dt)

        self.opti = ca.Opti()
        self.x = self.opti.variable(nx, N + 1)
        self.u = self.opti.variable(nu, N)
        self.holgura = self.opti.variable(1, N)  # sobre x_1..x_N, x_0 está fijo
        self.x_inicial = self.opti.parameter(nx, 1)

        costo = 0
        for k in range(N):
            costo += costo_etapa(self.x[:, k], self.u[:, k], Q, p.R)
        if costo_terminal is not None:
            costo += costo_terminal(self.x[:, N])
        # Penalización lineal (exacta) más cuadrática (suave para Ipopt)
        costo += peso_holgura * (ca.sum2(self.holgura) + ca.sumsqr(self.holgura))
        self.opti.minimize(costo)

        self.opti.subject_to(self.x[:, 0] == self.x_inicial)
        for k in range(N):
            self.opti.subject_to(self.x[:, k + 1] == f_discreta(self.x[:, k], self.u[:, k]))
        self.opti.subject_to(self.opti.bounded(-self.F_max, self.u, self.F_max))
        self.opti.subject_to(self.holgura >= 0)
        x_lim = p.x_max - margen_riel
        self.opti.subject_to(self.opti.bounded(-x_lim - self.holgura, self.x[0, 1:], x_lim + self.holgura))

        # Restricción terminal opcional x_N ∈ Ω = {x : x^T P x <= alpha}, con (P, alpha)
        if conjunto_terminal is not None:
            P, alpha = conjunto_terminal
            self.opti.subject_to(ca.mtimes([self.x[:, N].T, ca.DM(P), self.x[:, N]]) <= alpha)

        # expand pasa el grafo de MX a SX, que en Ipopt evalúa mucho más rápido.
        # print_level 0 y sb 'yes' apagan el log y el banner de Ipopt.
        # mu_strategy adaptive aprovecha mejor el warm start: baja de ~7 a ~4
        # iteraciones por solve y el tiempo medio de ~15 a ~11 ms.
        self.opti.solver('ipopt', {
            'expand': True,
            'print_time': False,
            'ipopt': {
                'print_level': 0,
                'sb': 'yes',
                'tol': 1e-6,
                'max_iter': 200,
                'warm_start_init_point': 'yes',
                'mu_strategy': 'adaptive',
            },
        })

        self._x_guess = np.zeros((nx, N + 1))
        self._u_guess = np.zeros((nu, N))
        self._holgura_guess = np.zeros((1, N))

        # Registro para medir contra los 20 ms
        self.tiempos_solve = []
        self.ultimo_exito = True
        self.ultima_holgura = 0.0

        # El primer solve arma el solver (expand, derivadas) y tarda segundos.
        # Lo hago acá para que no ensucie la medición del lazo cerrado.
        self.solve(np.zeros(nx))
        self.reiniciar()

    def reiniciar(self):
        """Borra el warm start y los tiempos, para arrancar un episodio nuevo."""
        self._x_guess[:] = 0.0
        self._u_guess[:] = 0.0
        self._holgura_guess[:] = 0.0
        self.tiempos_solve = []

    def solve(self, estado):
        """Devuelve la primera acción de control óptima para el estado dado."""
        estado = np.asarray(estado, dtype=float).reshape(-1)
        # Si el guess viene de cero (primer paso) arranco la trayectoria en el estado actual
        if not np.any(self._x_guess):
            self._x_guess = np.tile(estado.reshape(-1, 1), (1, self.N + 1))

        self.opti.set_value(self.x_inicial, estado)
        self.opti.set_initial(self.x, self._x_guess)
        self.opti.set_initial(self.u, self._u_guess)
        self.opti.set_initial(self.holgura, self._holgura_guess)

        inicio = time.perf_counter()
        try:
            solucion = self.opti.solve()
            self.ultimo_exito = True
        except RuntimeError:
            # Ipopt no convergió (max_iter, restauración fallida...). Uso el
            # último iterado igual, que suele ser razonable, y lo marco.
            solucion = self.opti.debug
            self.ultimo_exito = False
        self.tiempos_solve.append(time.perf_counter() - inicio)

        x_sol = np.reshape(solucion.value(self.x), self._x_guess.shape)
        u_sol = np.reshape(solucion.value(self.u), self._u_guess.shape)
        holgura_sol = np.reshape(solucion.value(self.holgura), self._holgura_guess.shape)
        self.ultima_holgura = float(np.max(holgura_sol))
        # Costo óptimo J*_N(x) (etapas + terminal + holgura): es el target de la
        # iteración de valor del eje A, (T V)(x) con el MPC como operador de Bellman
        self.ultimo_costo = float(solucion.value(self.opti.f))

        # Warm start: uso el resultado del instante previo para inicializar
        # la nueva búsqueda, eliminando el primer elemento y repitiendo el último
        self._x_guess = np.hstack([x_sol[:, 1:], x_sol[:, -1:]])
        self._u_guess = np.hstack([u_sol[:, 1:], u_sol[:, -1:]])
        self._holgura_guess = np.hstack([holgura_sol[:, 1:], holgura_sol[:, -1:]])

        return float(np.clip(u_sol[0, 0], -self.F_max, self.F_max))
