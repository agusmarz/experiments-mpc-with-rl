import copy
import time

import casadi as ca
import numpy as np
import scipy.linalg as sl


from pendulo_env_incisoB import ParametrosPlanta, derivada


class _ParametrosSimbolicos:
    """Envoltorio simbólico de ParametrosPlanta para usar [M, m, L, r] como parámetros de CasADi."""

    def __init__(self, p_base: ParametrosPlanta, theta_sym: ca.MX):
        self.M = theta_sym[0]
        self.m = theta_sym[1]
        self.L = theta_sym[2]
        self.r = theta_sym[3]

        self.g = p_base.g
        self.b_carro = p_base.b_carro
        self.b_junta = p_base.b_junta
        self.fuerza_constante = p_base.fuerza_constante

    @property
    def l(self):
        return self.L / 2.0

    @property
    def J(self):
        I = (self.m / 12.0) * (3.0 * self.r**2 + self.L**2)
        return I + self.m * self.l**2


def costo_etapa(x, u, Q, R):
    return ca.mtimes([x.T, Q, x]) + R * u**2


def costo_terminal_cuadratico(P):
    """V_f(x) = x^T P x, para enchufar como costo terminal del MPC."""
    P = ca.DM(P)
    return lambda x: ca.mtimes([x.T, P, x])


def paso_rk4(p, dt):
    """Función de CasADi x[k+1] = RK4(x[k], F[k], theta) con F constante durante dt.

    Usa derivada(..., lib=casadi) con theta = [M, m, L, r] como entrada simbólica
    para poder actualizar los parámetros físicos en tiempo real sin recompilar Ipopt.
    """
    x = ca.MX.sym("x", 4)
    F = ca.MX.sym("F")
    theta = ca.MX.sym("theta", 4)

    p_sim = _ParametrosSimbolicos(p, theta)
    f = lambda estado, fuerza: ca.vertcat(*derivada(estado, fuerza, p_sim, lib=ca))

    k1 = f(x, F)
    k2 = f(x + dt / 2 * k1, F)
    k3 = f(x + dt / 2 * k2, F)
    k4 = f(x + dt * k3, F)
    x_siguiente = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function("paso_rk4", [x, F, theta], [x_siguiente])


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

    El problema de optimización se arma una sola vez con [M, m, L, r] como
    parámetros dinámicos de CasADi; cada llamada a solve() actualiza el estado
    inicial (y opcionalmente los parámetros físicos) sin recompilar el solver.

    Restricciones:
        |F_k| <= F_max                 dura
        |x_k| <= x_max - margen_riel + holgura_k
                                       blanda, holgura_k >= 0 muy penalizada
    """

    def __init__(self, p: ParametrosPlanta, N, costo_terminal=None, F_max=None,
                 peso_holgura=1e4, margen_riel=0.01, conjunto_terminal=None):
        self.p = copy.deepcopy(p)
        self.N = N
        self.F_max = p.fuerza_max if F_max is None else F_max
        nx, nu, n_theta = 4, 1, 4
        Q = ca.DM(self.p.Q)
        f_discreta = paso_rk4(self.p, self.p.dt)

        self.opti = ca.Opti()
        self.x = self.opti.variable(nx, N + 1)
        self.u = self.opti.variable(nu, N)
        self.holgura = self.opti.variable(1, N)
        self.x_inicial = self.opti.parameter(nx, 1)
        self.theta_params = self.opti.parameter(n_theta, 1)  # [M, m, L, r]
        # Referencia de posición del carro a lo largo del horizonte, x_ref[0..N].
        # El equilibrio [x_ref, 0, 0, 0] es válido para cualquier x_ref (la dinámica
        # no depende de x), así que alcanza con desplazar el costo.
        self.ref_x = self.opti.parameter(1, N + 1)

        def desvio(k):
            return self.x[:, k] - ca.vertcat(self.ref_x[0, k], 0, 0, 0)

        costo = 0
        for k in range(N):
            costo += costo_etapa(desvio(k), self.u[:, k], Q, self.p.R)
        if costo_terminal is not None:
            costo += costo_terminal(desvio(N))
        costo += peso_holgura * (ca.sum2(self.holgura) + ca.sumsqr(self.holgura))
        self.opti.minimize(costo)

        self.opti.subject_to(self.x[:, 0] == self.x_inicial)
        for k in range(N):
            self.opti.subject_to(
                self.x[:, k + 1] == f_discreta(self.x[:, k], self.u[:, k], self.theta_params)
            )
        self.opti.subject_to(self.opti.bounded(-self.F_max, self.u, self.F_max))
        self.opti.subject_to(self.holgura >= 0)
        x_lim = self.p.x_max - margen_riel
        self.opti.subject_to(self.opti.bounded(-x_lim - self.holgura, self.x[0, 1:], x_lim + self.holgura))

        if conjunto_terminal is not None:
            P, alpha = conjunto_terminal
            self.opti.subject_to(ca.mtimes([desvio(N).T, ca.DM(P), desvio(N)]) <= alpha)

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

        self.theta_actual = np.array([self.p.M, self.p.m, self.p.L, self.p.r], dtype=float)
        self.actualizar_parametros(self.theta_actual)
        self.opti.set_value(self.ref_x, np.zeros((1, N + 1)))

        self.tiempos_solve = []
        self.ultimo_exito = True
        self.ultima_holgura = 0.0
        self.ultima_trayectoria_x = self._x_guess.copy()
        self.ultima_trayectoria_u = self._u_guess.copy()

        # Compilación inicial del grafo en Ipopt
        self.solve(np.zeros(nx))
        self.reiniciar()

    def actualizar_parametros(self, theta) -> None:
        """Actualiza [M, m, L, r] en el solver sin recompilar el problema NLP."""
        theta_arr = np.asarray(theta, dtype=float).reshape(4)
        self.theta_actual = theta_arr.copy()
        self.p.M = float(theta_arr[0])
        self.p.m = float(theta_arr[1])
        self.p.L = float(theta_arr[2])
        self.p.r = float(theta_arr[3])
        self.opti.set_value(self.theta_params, self.theta_actual)

    def obtener_parametros(self) -> np.ndarray:
        """Devuelve una copia del vector actual [M, m, L, r] usado por el MPC."""
        return self.theta_actual.copy()

    def reiniciar(self, theta_inicial=None) -> None:
        """Borra el warm start y los tiempos para arrancar un episodio nuevo."""
        self._x_guess[:] = 0.0
        self._u_guess[:] = 0.0
        self._holgura_guess[:] = 0.0
        self.tiempos_solve = []
        self.ultimo_exito = True
        self.ultima_holgura = 0.0
        if theta_inicial is not None:
            self.actualizar_parametros(theta_inicial)

    def solve(self, estado, theta=None, x_ref=None) -> tuple[float, np.ndarray]:
        """Resuelve el MPC y devuelve (accion_optima, estado_predicho_siguiente).

        Si `theta` se pasa como argumento, actualiza [M, m, L, r] antes de resolver.
        `x_ref` es la posición de referencia del carro: un escalar (constante en todo
        el horizonte) o un vector de N + 1 valores para los instantes k..k+N. Con
        None se regula al centro (x_ref = 0).
        Si los parámetros propuestos vuelven el problema infactible o numéricamente
        inválido, marca `self.ultimo_exito = False` para que el evaluador penalice.
        """
        estado = np.asarray(estado, dtype=float).reshape(-1)
        if theta is not None:
            self.actualizar_parametros(theta)
        ref = np.zeros(self.N + 1) if x_ref is None else np.broadcast_to(
            np.asarray(x_ref, dtype=float).reshape(-1), (self.N + 1,))
        self.opti.set_value(self.ref_x, ref.reshape(1, -1))

        # Chequeo rápido de degeneración física (ej. masas o largo <= 0)
        M, m, L, r = self.theta_actual
        if not np.all(np.isfinite(self.theta_actual)) or M <= 1e-6 or m <= 1e-6 or L <= 1e-6 or r < 0.0:
            self.ultimo_exito = False
            self.ultima_holgura = np.inf
            return 0.0, estado.copy()

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
            solucion = self.opti.debug
            self.ultimo_exito = False
        self.tiempos_solve.append(time.perf_counter() - inicio)

        x_sol = np.reshape(solucion.value(self.x), self._x_guess.shape)
        u_sol = np.reshape(solucion.value(self.u), self._u_guess.shape)
        holgura_sol = np.reshape(solucion.value(self.holgura), self._holgura_guess.shape)

        # Resguardo ante NaNs si Ipopt divergió por parámetros extremos
        if not (np.all(np.isfinite(x_sol)) and np.all(np.isfinite(u_sol))):
            self.ultimo_exito = False
            self.ultima_holgura = np.inf
            self._x_guess[:] = 0.0
            self._u_guess[:] = 0.0
            self._holgura_guess[:] = 0.0
            return 0.0, estado.copy()

        self.ultima_holgura = float(np.max(holgura_sol))
        self.ultima_trayectoria_x = x_sol.copy()
        self.ultima_trayectoria_u = u_sol.copy()

        # Desplazamiento para warm start del siguiente paso
        self._x_guess = np.hstack([x_sol[:, 1:], x_sol[:, -1:]])
        self._u_guess = np.hstack([u_sol[:, 1:], u_sol[:, -1:]])
        self._holgura_guess = np.hstack([holgura_sol[:, 1:], holgura_sol[:, -1:]])

        u_opt = float(np.clip(u_sol[0, 0], -self.F_max, self.F_max))
        x_pred_siguiente = np.asarray(x_sol[:, 1], dtype=float).reshape(-1)
        return u_opt, x_pred_siguiente