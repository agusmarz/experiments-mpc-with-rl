from typing import Sequence
import numpy as np

from pendulo_env_incisoB import ParametrosPlanta, PenduloEnv
from mpc_no_lineal import MPCNoLineal, costo_terminal_cuadratico, riccati
from controlador_lqr import ControladorLQR
from Algoritmo_Genetico import RedEstimadora


def referencia_carro(ref: dict | None, pasos, dt: float, pasos_por_tramo: int = 100,
                     secuencia_escalones: Sequence[float] = (0.0, -1.0, 1.0)) -> np.ndarray:
    """Posición de referencia del carro en los pasos indicados (array de enteros).

    ref = None                                         -> centro (0)
    ref = {"tipo": "escalones", "amplitud": A}         -> A * secuencia, un valor por
          tramo de `pasos_por_tramo` pasos; la secuencia se repite si el episodio sigue
    ref = {"tipo": "senoidal", "amplitud": A, "frecuencia": f, "fase": phi}
                                                       -> A * sin(2π f k dt + phi)
    """
    k = np.asarray(pasos, dtype=np.float64)
    if ref is None:
        return np.zeros_like(k)
    if ref["tipo"] == "escalones":
        secuencia = np.asarray(secuencia_escalones, dtype=np.float64)
        idx = (k // pasos_por_tramo).astype(int) % len(secuencia)
        return ref["amplitud"] * secuencia[idx]
    if ref["tipo"] == "senoidal":
        return ref["amplitud"] * np.sin(2.0 * np.pi * ref["frecuencia"] * k * dt + ref.get("fase", 0.0))
    raise ValueError(f"Tipo de referencia desconocido: {ref['tipo']}")


class EvaluadorSimulacionMPC:
    """Evalúa una `RedEstimadora` en lazo cerrado con `MPCNoLineal` y `PenduloEnv`
    sobre múltiples escenarios aleatorios por generación.

    El fitness mide qué tan cerca terminan los parámetros estimados θ̂ de los
    reales θ*, relativo al error de partida (0 = perfecto, -1 = igual que no
    adaptar). Caerse o volver infactible el MPC se penaliza aparte.

    Cada escenario trae además una referencia de posición del carro que se mueve
    durante el episodio (escalones o senoidal), para excitar la dinámica y que
    el error de predicción tenga información sobre M y m.

    `controlador` elige quién da la acción y la predicción x̂[k+1]:
        "mpc"             MPCNoLineal con Ipopt (lo original, ~10 ms por paso)
        "lqr"             ControladorLQR con K nominal fija (~0,5 ms por paso)
        "lqr_adaptativo"  ControladorLQR con K recalculada con θ̂ en cada paso
    En los tres la predicción es un paso de RK4 del modelo con θ̂.
    """

    def __init__(
        self,
        params_iniciales_mpc: Sequence[float] = (3.0, 1.0, 0.5, 0.01),
        rangos_params_min: Sequence[float] = (1.5, 0.4, 0.30, 0.005),
        rangos_params_max: Sequence[float] = (5.5, 2.0, 0.90, 0.030),
        rangos_estado_min: Sequence[float] = (-0.15, -0.20, -0.24, -0.20),
        rangos_estado_max: Sequence[float] = (0.15, 0.20, 0.24, 0.20),
        theta_inicial_min_abs: float = 0.06,
        controlador: str = "mpc",
        horizonte_mpc: int = 20,
        usar_costo_terminal: bool = True,
        escala_delta: float | Sequence[float] = (0.04, 0.015, 0.015, 0.0005),
        # Fitness
        pesos_params: Sequence[float] = (1.0, 1.0, 1.0, 0.1),
        fraccion_final: float = 0.25,
        penalizacion_caida: float = 2.0,
        peso_suavizado_delta: float = 0.01,
        penalizacion_infactible: float = 2.0,
        # Referencia de posición
        tipos_referencia: Sequence[str] = ("escalones", "senoidal"),
        rango_amplitud_ref: tuple[float, float] = (0.10, 0.25),
        secuencia_escalones: Sequence[float] = (0.0, -1.0, 1.0),
        pasos_por_tramo: int = 100,
        rango_frecuencia_ref: tuple[float, float] = (0.1, 0.4),
        seed_env: int = 42,
    ):
        self.params_iniciales_mpc = np.asarray(params_iniciales_mpc, dtype=np.float64).reshape(4)
        self.rangos_params_min = np.asarray(rangos_params_min, dtype=np.float64).reshape(4)
        self.rangos_params_max = np.asarray(rangos_params_max, dtype=np.float64).reshape(4)
        self.rango_params = self.rangos_params_max - self.rangos_params_min
        self.rangos_estado_min = np.asarray(rangos_estado_min, dtype=np.float64).reshape(4)
        self.rangos_estado_max = np.asarray(rangos_estado_max, dtype=np.float64).reshape(4)
        self.theta_inicial_min_abs = float(theta_inicial_min_abs)

        self.escala_delta = np.asarray(escala_delta, dtype=np.float64)
        self.pesos_params = np.asarray(pesos_params, dtype=np.float64).reshape(4)
        self.fraccion_final = float(fraccion_final)
        self.penalizacion_caida = float(penalizacion_caida)
        self.peso_suavizado_delta = float(peso_suavizado_delta)
        self.penalizacion_infactible = float(penalizacion_infactible)

        self.tipos_referencia = tuple(tipos_referencia)
        self.rango_amplitud_ref = tuple(rango_amplitud_ref)
        self.secuencia_escalones = np.asarray(secuencia_escalones, dtype=np.float64)
        self.pasos_por_tramo = int(pasos_por_tramo)
        self.rango_frecuencia_ref = tuple(rango_frecuencia_ref)
        self.seed_env = seed_env

        # Por escenario de la última llamada a evaluar(): True si el MPC no pudo
        # resolver el paso 0. Ese paso no depende de la red, así que el AG lo usa
        # para detectar problemas del entorno (por ejemplo, falta de memoria).
        self.fallos_inicio: list[bool] = []
        self._fallo_inicio = False

        # Entorno de simulación (se pasa ParametrosPlanta() explícitamente)
        self.env = PenduloEnv(parametros=ParametrosPlanta())
        self.dt = self.env.p.dt

        # Controlador (el atributo se sigue llamando `mpc` aunque sea un LQR:
        # los dos exponen N, ultimo_exito, reiniciar() y solve())
        p_mpc = ParametrosPlanta()
        p_mpc.M, p_mpc.m, p_mpc.L, p_mpc.r = self.params_iniciales_mpc
        self.controlador = controlador
        if controlador == "mpc":
            costo_term = costo_terminal_cuadratico(riccati(p_mpc)[0]) if usar_costo_terminal else None
            self.mpc = MPCNoLineal(p=p_mpc, N=horizonte_mpc, costo_terminal=costo_term)
        elif controlador in ("lqr", "lqr_adaptativo"):
            self.mpc = ControladorLQR(p=p_mpc, adaptativo=(controlador == "lqr_adaptativo"))
        else:
            raise ValueError(f"controlador debe ser 'mpc', 'lqr' o 'lqr_adaptativo', no {controlador!r}")

    # ------------------------------------------------------------------ escenarios

    def _sortear_referencia(self, rng: np.random.Generator) -> dict:
        tipo = self.tipos_referencia[rng.integers(len(self.tipos_referencia))]
        amplitud = float(rng.uniform(*self.rango_amplitud_ref))
        if tipo == "escalones":
            return {"tipo": "escalones", "amplitud": amplitud}
        if tipo == "senoidal":
            return {
                "tipo": "senoidal",
                "amplitud": amplitud,
                "frecuencia": float(rng.uniform(*self.rango_frecuencia_ref)),
                "fase": float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        raise ValueError(f"Tipo de referencia desconocido: {tipo}")

    def generar_escenarios(self, num_casos: int, rng: np.random.Generator) -> list[dict]:
        """Genera `num_casos` plantas reales, estados iniciales y referencias aleatorias."""
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
                "ref": self._sortear_referencia(rng),
            })
        return escenarios

    def referencia(self, ref: dict | None, pasos) -> np.ndarray:
        """Posición de referencia del carro en los pasos indicados (array de enteros).

        escalones: amplitud * secuencia_escalones, un valor por tramo de
                   `pasos_por_tramo` pasos; la secuencia se repite si el episodio sigue.
        senoidal:  amplitud * sin(2π f k dt + fase).
        """
        return referencia_carro(ref, pasos, self.dt, self.pasos_por_tramo, self.secuencia_escalones)

    def _ref_horizonte(self, ref: dict | None, k: int) -> np.ndarray:
        """Referencia para los instantes k..k+N que ve el MPC (preview completo)."""
        return self.referencia(ref, np.arange(k, k + self.mpc.N + 1))

    # ------------------------------------------------------------------ fitness

    def _error_parametrico(self, theta: np.ndarray, reales: np.ndarray) -> float:
        """Error cuadrático ponderado, con cada parámetro normalizado por su rango."""
        e = (theta - reales) / self.rango_params
        return float(np.sum(self.pesos_params * e**2) / np.sum(self.pesos_params))

    def _puntaje(self, errores, err_inicial, pasos_hechos, pasos_episodio, suavizado, fallo) -> float:
        """0 = θ̂ terminó en θ*, -1 = igual que el nominal, < -1 = empeoró."""
        n = max(1, int(round(self.fraccion_final * len(errores)))) if errores else 0
        err_final = float(np.mean(errores[-n:])) if n else err_inicial
        score = -err_final / max(err_inicial, 1e-3)
        score -= self.peso_suavizado_delta * suavizado / max(pasos_hechos, 1)
        if fallo:
            score -= self.penalizacion_caida * (1.0 - pasos_hechos / pasos_episodio)
        return score

    def _correr_un_escenario(self, red: RedEstimadora, pasos_episodio: int, esc: dict) -> float:
        """Ejecuta un episodio completo sobre un escenario dado."""
        reales = np.asarray(esc["reales"], dtype=np.float64)
        ref = esc.get("ref")
        self.env.p.M, self.env.p.m, self.env.p.L, self.env.p.r = reales
        x0 = np.asarray(esc["x0"], dtype=np.float64)
        estado_real, _ = self.env.reset(seed=self.seed_env, options={"estado": x0})

        theta_mpc = self.params_iniciales_mpc.copy()
        self.mpc.reiniciar(theta_inicial=theta_mpc)
        err_inicial = self._error_parametrico(theta_mpc, reales)
        errores: list[float] = []
        suavizado = 0.0

        u_opt, x_pred_siguiente = self.mpc.solve(estado_real, x_ref=self._ref_horizonte(ref, 0))
        if not self.mpc.ultimo_exito:
            self._fallo_inicio = True
            return -1.0 - self.penalizacion_infactible

        estado_real, _, terminated, truncated, _ = self.env.step([u_opt])
        if terminated or truncated:
            return self._puntaje(errores, err_inicial, 1, pasos_episodio, 0.0, fallo=True)

        for k in range(1, pasos_episodio):
            error_pred = estado_real - x_pred_siguiente
            delta_theta = red.predecir(theta_mpc, estado_real, error_pred)
            theta_mpc = theta_mpc + self.escala_delta * delta_theta
            suavizado += float(np.sum(delta_theta**2))
            errores.append(self._error_parametrico(theta_mpc, reales))

            u_opt, x_pred_siguiente = self.mpc.solve(
                estado_real, theta=theta_mpc, x_ref=self._ref_horizonte(ref, k)
            )
            if not self.mpc.ultimo_exito:
                return (self._puntaje(errores, err_inicial, k, pasos_episodio, suavizado, fallo=True)
                        - self.penalizacion_infactible)

            estado_real, _, terminated, truncated, _ = self.env.step([u_opt])
            if terminated or truncated:
                return self._puntaje(errores, err_inicial, k + 1, pasos_episodio, suavizado, fallo=True)

        return self._puntaje(errores, err_inicial, pasos_episodio, pasos_episodio, suavizado, fallo=False)

    def evaluar(self, red: RedEstimadora, pasos_episodio: int, escenarios: Sequence[dict]) -> float:
        """Evalúa la red en todos los escenarios de la generación y devuelve el score promedio."""
        scores = []
        self.fallos_inicio = []
        for esc in escenarios:
            self._fallo_inicio = False
            scores.append(self._correr_un_escenario(red, pasos_episodio, esc))
            self.fallos_inicio.append(self._fallo_inicio)
        return float(np.mean(scores))

    def close(self):
        self.env.close()
