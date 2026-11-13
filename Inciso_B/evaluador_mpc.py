from typing import Sequence
import numpy as np

from pendulo_env_incisoB import ParametrosPlanta, PenduloEnv
from mpc_no_lineal import MPCNoLineal, costo_terminal_cuadratico, riccati
from Algoritmo_Genetico import RedEstimadora


class EvaluadorSimulacionMPC:
    """Evalúa una `RedEstimadora` en lazo cerrado con `MPCNoLineal` y `PenduloEnv`
    sobre múltiples escenarios aleatorios por generación.
    """

    def __init__(
        self,
        params_iniciales_mpc: Sequence[float] = (3.0, 1.0, 0.5, 0.01),
        rangos_params_min: Sequence[float] = (1.5, 0.4, 0.30, 0.005),
        rangos_params_max: Sequence[float] = (5.5, 2.0, 0.90, 0.030),
        rangos_estado_min: Sequence[float] = (-0.15, -0.20, -0.24, -0.20),
        rangos_estado_max: Sequence[float] = (0.15, 0.20, 0.24, 0.20),
        theta_inicial_min_abs: float = 0.06,
        horizonte_mpc: int = 20,
        usar_costo_terminal: bool = True,
        escala_delta: float | Sequence[float] = (0.04, 0.015, 0.015, 0.0005),
        bonus_erguido: float = 2.0,
        peso_error_pred: float = 1500.0,
        peso_suavizado_delta: float = 0.2,
        penalizacion_infactible: float = 1000.0,
        seed_env: int = 42,
    ):
        self.params_iniciales_mpc = np.asarray(params_iniciales_mpc, dtype=np.float64).reshape(4)
        self.rangos_params_min = np.asarray(rangos_params_min, dtype=np.float64).reshape(4)
        self.rangos_params_max = np.asarray(rangos_params_max, dtype=np.float64).reshape(4)
        self.rangos_estado_min = np.asarray(rangos_estado_min, dtype=np.float64).reshape(4)
        self.rangos_estado_max = np.asarray(rangos_estado_max, dtype=np.float64).reshape(4)
        self.theta_inicial_min_abs = float(theta_inicial_min_abs)

        self.escala_delta = np.asarray(escala_delta, dtype=np.float64)
        self.bonus_erguido = float(bonus_erguido)
        self.peso_error_pred = float(peso_error_pred)
        self.peso_suavizado_delta = float(peso_suavizado_delta)
        self.penalizacion_infactible = float(penalizacion_infactible)
        self.seed_env = seed_env

        # Ponderación del error de predicción [x, x_dot, th, th_dot]
        self.pesos_pred = np.array([10.0, 1.0, 10.0, 1.0], dtype=np.float64)

        # Entorno de simulación (se pasa ParametrosPlanta() explícitamente)
        self.env = PenduloEnv(parametros=ParametrosPlanta())

        # Solver NMPC
        p_mpc = ParametrosPlanta()
        p_mpc.M, p_mpc.m, p_mpc.L, p_mpc.r = self.params_iniciales_mpc
        costo_term = costo_terminal_cuadratico(riccati(p_mpc)[0]) if usar_costo_terminal else None
        self.mpc = MPCNoLineal(p=p_mpc, N=horizonte_mpc, costo_terminal=costo_term)

    def generar_escenarios(self, num_casos: int, rng: np.random.Generator) -> list[dict]:
        """Genera `num_casos` plantas reales y estados iniciales aleatorios."""
        escenarios: list[dict] = []
        for _ in range(num_casos):
            params_reales = rng.uniform(self.rangos_params_min, self.rangos_params_max)
            x0 = rng.uniform(self.rangos_estado_min, self.rangos_estado_max)

            # Garantizar una inclinación inicial mínima para excitar la dinámica
            if abs(x0[2]) < self.theta_inicial_min_abs:
                signo = -1.0 if rng.random() < 0.5 else 1.0
                max_th = max(abs(self.rangos_estado_min[2]), abs(self.rangos_estado_max[2]))
                x0[2] = signo * rng.uniform(self.theta_inicial_min_abs, max_th)

            escenarios.append({
                "reales": params_reales,
                "x0": x0,
            })
        return escenarios

    def _puntuar_paso(self, estado: np.ndarray, error_pred: np.ndarray, delta_theta: np.ndarray) -> float:
        """Puntúa inclinación correcta + precisión de predicción del modelo del MPC."""
        _, _, th, _ = estado

        # 1. Premio por mantenerse cerca de la vertical (máximo cuando th = 0)
        factor_vertical = max(0.0, 1.0 - abs(th) / self.env.p.theta_corte)
        premio_inclinacion = self.bonus_erguido * factor_vertical

        # 2. Penalización por discrepancia entre planta real y predicción del MPC
        err_pred_norm = float(np.sum((error_pred * self.pesos_pred) ** 2))
        penalizacion_modelo = self.peso_error_pred * err_pred_norm

        # 3. Penalización suave para evitar cambios innecesarios cuando ya convergió
        penalizacion_movimiento = self.peso_suavizado_delta * float(np.sum(delta_theta**2))

        return premio_inclinacion - penalizacion_modelo - penalizacion_movimiento

    def _correr_un_escenario(self, red: RedEstimadora, pasos_episodio: int, esc: dict) -> float:
        """Ejecuta un episodio completo sobre un escenario dado."""
        self.env.p.M, self.env.p.m, self.env.p.L, self.env.p.r = esc["reales"]
        x0 = np.asarray(esc["x0"], dtype=np.float64)
        estado_real, _ = self.env.reset(seed=self.seed_env, options={"estado": x0})

        theta_mpc = self.params_iniciales_mpc.copy()
        self.mpc.reiniciar(theta_inicial=theta_mpc)

        u_opt, x_pred_siguiente = self.mpc.solve(estado_real)
        if not self.mpc.ultimo_exito:
            return -self.penalizacion_infactible

        estado_real, _, terminated, truncated, _ = self.env.step([u_opt])
        score = self._puntuar_paso(estado_real, np.zeros(4), np.zeros(4))

        if terminated or truncated:
            return score - self.env.p.penalizacion_falla

        for _ in range(1, pasos_episodio):
            error_pred = estado_real - x_pred_siguiente
            delta_theta = red.predecir(theta_mpc, estado_real, error_pred)

            theta_mpc = theta_mpc + self.escala_delta * delta_theta
            u_opt, x_pred_siguiente = self.mpc.solve(estado_real, theta=theta_mpc)

            if not self.mpc.ultimo_exito:
                return score - self.penalizacion_infactible

            estado_real, _, terminated, truncated, _ = self.env.step([u_opt])
            score += self._puntuar_paso(estado_real, error_pred, delta_theta)

            if terminated or truncated:
                score -= self.env.p.penalizacion_falla
                break

        return score

    def evaluar(self, red: RedEstimadora, pasos_episodio: int, escenarios: Sequence[dict]) -> float:
        """Evalúa la red en todos los escenarios de la generación y devuelve el score promedio."""
        scores = [self._correr_un_escenario(red, pasos_episodio, esc) for esc in escenarios]
        return float(np.mean(scores))

    def close(self):
        self.env.close()