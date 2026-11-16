import math
from pathlib import Path
from typing import Sequence, Tuple
import numpy as np

try:
    from scipy.linalg import solve_continuous_are
    TIENE_SCIPY = True
except ImportError:
    TIENE_SCIPY = False


# ==============================================================================
# CONSTANTES FÍSICAS Y RANGOS DE PARÁMETROS [M, m, L, r]
# ==============================================================================
DT_SIMULACION = 0.02  # 20 ms (unificado con evaluador_mpc y controlador_lqr)
GRAVEDAD = 9.81

THETA_NOMINAL = np.array([3.0, 1.0, 0.50, 0.01], dtype=np.float32)
THETA_MIN = np.array([1.5, 0.4, 0.30, 0.01], dtype=np.float32)
THETA_MAX = np.array([5.5, 2.0, 0.90, 0.01], dtype=np.float32)

THETA_CENTRO = np.array([3.5, 1.2, 0.60, 0.0175], dtype=np.float32)
THETA_SEMI_RANGO = np.array([2.0, 0.8, 0.30, 0.0125], dtype=np.float32)


# ==============================================================================
# DINÁMICA DE LA PLANTA E INTEGRADOR RK4
# ==============================================================================
def dinamica_pendulo(estado: np.ndarray, fuerza: float, params: Sequence[float]) -> np.ndarray:
    """
    Ecuaciones no lineales del carro-péndulo invertido (theta = 0 vertical arriba).
    estado = [x, x_dot, theta, theta_dot]
    params = [M, m, L, r]
    """
    M, m, L, r = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    x, x_dot, theta, theta_dot = (
        float(estado[0]),
        float(estado[1]),
        float(estado[2]),
        float(estado[3]),
    )

    l = 0.5 * L
    J = m * ((r ** 2) / 4.0 + (L ** 2) / 3.0)

    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    det = (M + m) * J - (m * l * cos_t) ** 2
    rhs1 = fuerza + m * l * (theta_dot ** 2) * sin_t
    rhs2 = m * GRAVEDAD * l * sin_t

    x_ddot = (J * rhs1 - m * l * cos_t * rhs2) / det
    theta_ddot = ((M + m) * rhs2 - m * l * cos_t * rhs1) / det

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=np.float64)


def paso_rk4(
    estado: np.ndarray,
    fuerza: float,
    params: Sequence[float],
    dt: float = DT_SIMULACION,
) -> np.ndarray:
    """Un paso de integración Runge-Kutta de 4to orden."""
    s = np.asarray(estado, dtype=np.float64)
    k1 = dinamica_pendulo(s, fuerza, params)
    k2 = dinamica_pendulo(s + 0.5 * dt * k1, fuerza, params)
    k3 = dinamica_pendulo(s + 0.5 * dt * k2, fuerza, params)
    k4 = dinamica_pendulo(s + dt * k3, fuerza, params)
    return s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def calcular_ganancia_lqr_nominal(params: Sequence[float] = THETA_NOMINAL) -> np.ndarray:
    """Calcula una ganancia LQR estabilizante alrededor de theta = 0 para generar trayectorias."""
    if not TIENE_SCIPY:
        return np.array([-3.16, -5.80, -48.50, -11.20], dtype=np.float64)

    M, m, L, r = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    l = 0.5 * L
    J = m * ((r ** 2) / 4.0 + (L ** 2) / 3.0)
    det0 = (M + m) * J - (m * l) ** 2

    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -(m ** 2 * GRAVEDAD * l ** 2) / det0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, ((M + m) * m * GRAVEDAD * l) / det0, 0.0],
        ],
        dtype=np.float64,
    )
    B = np.array([[0.0], [J / det0], [0.0], [-(m * l) / det0]], dtype=np.float64)

    Q = np.diag([12.0, 1.5, 60.0, 6.0])
    R = np.array([[0.15]], dtype=np.float64)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P).ravel()
    return K


# ==============================================================================
# RED NEURONAL ESTIMADORA DE PARÁMETROS
# ==============================================================================
class RedEstimadora:
    """
    MLP compacto con activaciones tanh que estima incrementos delta_theta = [dM, dm, dL, dr]
    a partir de la acción de control, el estado medido, el error de predicción a 1 paso
    y la estimación paramétrica actual.
    """

    ESCALA_U = 30.0
    ESCALA_ESTADO = np.array([2.0, 3.0, 0.5, 3.0], dtype=np.float32)
    ESCALA_ERROR = np.array([0.02, 0.15, 0.02, 0.25], dtype=np.float32)
    ESCALA_DELTA = np.array([0.10, 0.04, 0.03, 0.0], dtype=np.float32)  # dr = 0 (r inobservable)

    def __init__(
        self,
        dim_entrada: int = 13,
        capas_ocultas: Tuple[int, ...] = (32, 32),
        dim_salida: int = 4,
        escala_inicial: float = 0.25,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dim_entrada = dim_entrada
        self.capas_ocultas = tuple(capas_ocultas)
        self.dim_salida = dim_salida

        if rng is None:
            rng = np.random.default_rng()

        arquitectura = [dim_entrada, *self.capas_ocultas, dim_salida]
        self.pesos: list[np.ndarray] = []
        self.sesgos: list[np.ndarray] = []

        for fan_in, fan_out in zip(arquitectura[:-1], arquitectura[1:]):
            std = escala_inicial / math.sqrt(fan_in)
            W = rng.normal(0.0, std, size=(fan_in, fan_out)).astype(np.float32)
            b = np.zeros(fan_out, dtype=np.float32)
            self.pesos.append(W)
            self.sesgos.append(b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32, copy=False)
        for W, b in zip(self.pesos, self.sesgos):
            h = np.tanh(h @ W + b)
        return h

    def predecir(
        self,
        accion_control: float,
        estado_real: np.ndarray,
        error_estados: np.ndarray,
        theta_estimado: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula el ajuste delta_theta para el paso actual.

        Args:
            accion_control: Fuerza u_k aplicada en el paso anterior [N].
            estado_real: Estado medido x_real_{k+1} = [x, x_dot, theta, theta_dot].
            error_estados: x_real_{k+1} - x_pred_{k+1}.
            theta_estimado: Estimación actual [M_hat, m_hat, L_hat, r_hat].

        Returns:
            delta_theta: Vector float32 de tamaño 4 con [dM, dm, dL, 0.0].
        """
        u_norm = np.array(
            [np.clip(float(accion_control) / self.ESCALA_U, -2.5, 2.5)],
            dtype=np.float32,
        )
        x_norm = np.clip(
            np.asarray(estado_real, dtype=np.float32) / self.ESCALA_ESTADO,
            -3.0,
            3.0,
        )
        e_norm = np.clip(
            np.asarray(error_estados, dtype=np.float32) / self.ESCALA_ERROR,
            -4.0,
            4.0,
        )
        theta_norm = (
            np.asarray(theta_estimado, dtype=np.float32) - THETA_CENTRO
        ) / THETA_SEMI_RANGO

        entrada = np.concatenate([u_norm, x_norm, e_norm, theta_norm], axis=0)
        salida_cruda = self.forward(entrada)

        norma_err = float(np.linalg.norm(e_norm))
        factor_freno = math.tanh(5.0 * norma_err)

        return salida_cruda * self.ESCALA_DELTA * factor_freno

    def num_parametros(self) -> int:
        return sum(W.size + b.size for W, b in zip(self.pesos, self.sesgos))

    def obtener_vector_pesos(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.sesgos):
            partes.append(W.ravel())
            partes.append(b.ravel())
        return np.concatenate(partes).astype(np.float32)

    def cargar_vector_pesos(self, vector: np.ndarray) -> None:
        idx = 0
        for i, (W, b) in enumerate(zip(self.pesos, self.sesgos)):
            tam_w = W.size
            self.pesos[i] = vector[idx : idx + tam_w].reshape(W.shape).astype(np.float32)
            idx += tam_w

            tam_b = b.size
            self.sesgos[i] = vector[idx : idx + tam_b].reshape(b.shape).astype(np.float32)
            idx += tam_b

    def clonar(self) -> "RedEstimadora":
        copia = RedEstimadora(
            dim_entrada=self.dim_entrada,
            capas_ocultas=self.capas_ocultas,
            dim_salida=self.dim_salida,
        )
        copia.cargar_vector_pesos(self.obtener_vector_pesos().copy())
        return copia

    def guardar(self, ruta: str | Path) -> None:
        np.savez(
            ruta,
            pesos=self.obtener_vector_pesos(),
            dim_entrada=np.array([self.dim_entrada], dtype=np.int32),
            capas_ocultas=np.array(self.capas_ocultas, dtype=np.int32),
            dim_salida=np.array([self.dim_salida], dtype=np.int32),
        )

    @classmethod
    def cargar(cls, ruta: str | Path) -> "RedEstimadora":
        datos = np.load(ruta)
        red = cls(
            dim_entrada=int(datos["dim_entrada"][0]),
            capas_ocultas=tuple(int(v) for v in datos["capas_ocultas"]),
            dim_salida=int(datos["dim_salida"][0]),
        )
        red.cargar_vector_pesos(datos["pesos"])
        return red


# ==============================================================================
# EVALUACIÓN DE EPISODIOS CON EXCITACIÓN PERSISTENTE
# ==============================================================================
def generar_escenarios_entrenamiento(
    n_escenarios: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Genera plantas reales aleatorias y condiciones iniciales variadas."""
    escenarios = []
    for i in range(n_escenarios):
        M_real = float(rng.uniform(1.8, 5.2))
        m_real = float(rng.uniform(0.5, 1.8))
        L_real = float(rng.uniform(0.35, 0.85))
        r_real = float(rng.uniform(0.006, 0.025))  # Variación real desconocida para robustez

        theta_init_est = THETA_NOMINAL.copy()
        if i % 2 == 1:
            # Perturbar también la estimación inicial para evitar memorizar un único punto de partida
            theta_init_est[0] = float(rng.uniform(2.2, 4.2))
            theta_init_est[1] = float(rng.uniform(0.6, 1.5))
            theta_init_est[2] = float(rng.uniform(0.38, 0.72))

        estado_init = np.array(
            [
                float(rng.uniform(-0.15, 0.15)),
                0.0,
                float(rng.uniform(-0.12, 0.12)),
                0.0,
            ],
            dtype=np.float64,
        )
        fase = float(rng.uniform(0.0, 2.0 * math.pi))

        escenarios.append(
            {
                "theta_real": np.array([M_real, m_real, L_real, r_real], dtype=np.float32),
                "theta_init_est": theta_init_est,
                "estado_init": estado_init,
                "fase": fase,
            }
        )
    return escenarios


def referencia_excitacion(t: float, fase: float) -> float:
    """
    Referencia de posición x_ref(t) con escalones y senoidales para garantizar
    aceleraciones sostenidas que permitan desacoplar M y m.
    """
    if t < 1.5:
        escalon = 0.45
    elif t < 3.2:
        escalon = -0.45
    elif t < 4.8:
        escalon = 0.35
    else:
        escalon = -0.25

    onda = 0.20 * math.sin(2.2 * t + fase) + 0.10 * math.sin(5.1 * t - fase)
    return escalon + onda


def evaluar_individuo(
    red: RedEstimadora,
    escenarios: Sequence[dict],
    K_lqr: np.ndarray,
    pasos: int = 300,
    peso_suavizado_delta: float = 0.02,
) -> Tuple[float, np.ndarray]:
    """
    Evalúa la red sobre un conjunto de escenarios.
    Retorna (fitness, mae_final_por_parametro [M, m, L]).
    """
    rango_param = THETA_MAX[:3] - THETA_MIN[:3]
    costos_escenarios = []
    errores_finales = []

    # Pesos temporales crecientes: prioriza converger rápido y sostener el valor correcto
    pesos_tiempo = np.linspace(0.4, 1.6, pasos, dtype=np.float64)
    pesos_tiempo /= np.sum(pesos_tiempo)

    for esc in escenarios:
        theta_real = esc["theta_real"]
        theta_est = esc["theta_init_est"].copy()
        estado_real = esc["estado_init"].copy()
        fase = esc["fase"]

        err_param_pasos = np.zeros(pasos, dtype=np.float64)
        err_pred_pasos = np.zeros(pasos, dtype=np.float64)
        deltas_hist = np.zeros((pasos, 3), dtype=np.float64)

        pasos_completados = pasos
        for k in range(pasos):
            t = k * DT_SIMULACION
            x_ref = referencia_excitacion(t, fase)
            estado_ref = np.array([x_ref, 0.0, 0.0, 0.0], dtype=np.float64)

            # Control estabilizante + leve señal de sondeo en fuerza
            sondeo = 1.2 * math.sin(7.3 * t + fase)
            u = float(-np.dot(K_lqr, estado_real - estado_ref) + sondeo)
            u = float(np.clip(u, -35.0, 35.0))

            # Predicción a 1 paso con los parámetros estimados actuales y avance de planta real
            estado_pred = paso_rk4(estado_real, u, theta_est, DT_SIMULACION)
            estado_siguiente = paso_rk4(estado_real, u, theta_real, DT_SIMULACION)

            # Chequeo de caída del péndulo
            if abs(estado_siguiente[2]) > 1.25 or abs(estado_siguiente[0]) > 4.5:
                pasos_completados = k
                break

            error_estados = estado_siguiente - estado_pred
            delta = red.predecir(u, estado_siguiente, error_estados, theta_est)

            theta_est = np.clip(theta_est + delta, THETA_MIN, THETA_MAX)
            theta_est[3] = THETA_NOMINAL[3]  # Mantener r fijo

            # Métricas de costo en este paso
            err_norm_param = (theta_est[:3] - theta_real[:3]) / rango_param
            err_param_pasos[k] = float(np.mean(err_norm_param ** 2))
            err_pred_pasos[k] = float(
                np.mean((error_estados / RedEstimadora.ESCALA_ERROR) ** 2)
            )
            deltas_hist[k] = delta[:3]

            estado_real = estado_siguiente

        if pasos_completados < pasos:
            # Penalización proporcional si el episodio terminó antes
            costo_esc = 5.0 + 10.0 * (1.0 - pasos_completados / pasos)
            errores_finales.append(np.abs(theta_est[:3] - theta_real[:3]))
        else:
            costo_param = float(np.dot(pesos_tiempo, err_param_pasos))
            costo_pred = float(np.mean(err_pred_pasos))
            diff_deltas = np.diff(deltas_hist, axis=0) / RedEstimadora.ESCALA_DELTA[:3]
            costo_suavizado = float(np.mean(np.sum(diff_deltas ** 2, axis=1)))

            costo_esc = costo_param + 0.05 * costo_pred + peso_suavizado_delta * costo_suavizado

            # Error medio absoluto en el último tercio del episodio (estado estacionario)
            errores_finales.append(np.abs(theta_est[:3] - theta_real[:3]))

        costos_escenarios.append(costo_esc)

    costo_total = float(np.mean(costos_escenarios))
    fitness = -costo_total
    mae_final = np.mean(errores_finales, axis=0)
    return fitness, mae_final


# ==============================================================================
# ALGORITMO GENÉTICO
# ==============================================================================
class AlgoritmoGenetico:
    def __init__(
        self,
        tamano_poblacion: int = 60,
        generaciones: int = 120,
        n_elites: int = 4,
        escala_inicial: float = 0.25,
        sigma_inicial: float = 0.15,
        sigma_final: float = 0.01,
        tasa_genes_ruido: float = 0.30,
        tasa_cruce: float = 0.80,
        n_escenarios_por_gen: int = 6,
        pasos_episodio: int = 300,
        peso_suavizado_delta: float = 0.02,
        semilla: int = 42,
    ) -> None:
        self.tamano_poblacion = tamano_poblacion
        self.generaciones = generaciones
        self.n_elites = n_elites
        self.escala_inicial = escala_inicial
        self.sigma_inicial = sigma_inicial
        self.sigma_final = sigma_final
        self.tasa_genes_ruido = tasa_genes_ruido
        self.tasa_cruce = tasa_cruce
        self.n_escenarios_por_gen = n_escenarios_por_gen
        self.pasos_episodio = pasos_episodio
        self.peso_suavizado_delta = peso_suavizado_delta

        self.rng = np.random.default_rng(semilla)
        self.K_lqr = calcular_ganancia_lqr_nominal(THETA_NOMINAL)

        self.poblacion = [
            RedEstimadora(escala_inicial=self.escala_inicial, rng=self.rng)
            for _ in range(self.tamano_poblacion)
        ]
        self.mejor_red: RedEstimadora | None = None
        self.mejor_fitness = -float("inf")
        self.historial: list[dict] = []

    def _seleccion_torneo(self, fitnesses: np.ndarray, k: int = 3) -> int:
        candidatos = self.rng.choice(self.tamano_poblacion, size=k, replace=False)
        return int(candidatos[np.argmax(fitnesses[candidatos])])

    def _cruzar(self, gen_a: np.ndarray, gen_b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.tasa_cruce:
            return gen_a.copy()
        # Cruce aritmético aleatorio por gen (BLX-0.25)
        alpha = self.rng.uniform(-0.25, 1.25, size=gen_a.shape).astype(np.float32)
        return alpha * gen_a + (1.0 - alpha) * gen_b

    def _mutar(self, genoma: np.ndarray, sigma: float) -> np.ndarray:
        mascara = self.rng.random(size=genoma.shape) < self.tasa_genes_ruido
        ruido = self.rng.normal(0.0, sigma, size=genoma.shape).astype(np.float32)
        return genoma + mascara * ruido

    def entrenar(
        self,
        ruta_guardado: str | Path = "mejor_red_estimadora.npz",
        verbose: bool = True,
    ) -> RedEstimadora:
        for gen in range(self.generaciones):
            progreso = gen / max(1, self.generaciones - 1)
            sigma = self.sigma_inicial * ((self.sigma_final / self.sigma_inicial) ** progreso)

            # Common Random Numbers: todos los individuos enfrentan los mismos escenarios en esta gen
            escenarios = generar_escenarios_entrenamiento(self.n_escenarios_por_gen, self.rng)

            fitnesses = np.zeros(self.tamano_poblacion, dtype=np.float64)
            maes = np.zeros((self.tamano_poblacion, 3), dtype=np.float64)

            for i, ind in enumerate(self.poblacion):
                fit, mae = evaluar_individuo(
                    ind,
                    escenarios,
                    self.K_lqr,
                    pasos=self.pasos_episodio,
                    peso_suavizado_delta=self.peso_suavizado_delta,
                )
                fitnesses[i] = fit
                maes[i] = mae

            orden = np.argsort(fitnesses)[::-1]
            idx_mejor = int(orden[0])

            # Actualizar mejor global
            self.mejor_red = self.poblacion[idx_mejor].clonar()
            self.mejor_fitness = float(fitnesses[idx_mejor])

            registro = {
                "generacion": gen + 1,
                "mejor_fitness": float(fitnesses[idx_mejor]),
                "fitness_medio": float(np.mean(fitnesses)),
                "mae_M": float(maes[idx_mejor, 0]),
                "mae_m": float(maes[idx_mejor, 1]),
                "mae_L": float(maes[idx_mejor, 2]),
                "sigma": float(sigma),
            }
            self.historial.append(registro)

            if verbose and (gen == 0 or (gen + 1) % 5 == 0 or gen == self.generaciones - 1):
                print(
                    f"Gen {gen + 1:3d}/{self.generaciones:3d} | "
                    f"Fit: {registro['mejor_fitness']:+.5f} (med: {registro['fitness_medio']:+.5f}) | "
                    f"MAE -> M: {registro['mae_M']:.3f} kg, "
                    f"m: {registro['mae_m']:.3f} kg, "
                    f"L: {registro['mae_L']:.3f} m | "
                    f"sigma: {sigma:.4f}"
                )

            # Construir siguiente generación
            nueva_poblacion: list[RedEstimadora] = []
            genomas = [ind.obtener_vector_pesos() for ind in self.poblacion]

            # Elitismo puro
            for idx_elite in orden[: self.n_elites]:
                nueva_poblacion.append(self.poblacion[int(idx_elite)].clonar())

            # Cruce + mutación
            while len(nueva_poblacion) < self.tamano_poblacion:
                p1 = self._seleccion_torneo(fitnesses)
                p2 = self._seleccion_torneo(fitnesses)
                hijo_gen = self._cruzar(genomas[p1], genomas[p2])
                hijo_gen = self._mutar(hijo_gen, sigma)

                hijo = RedEstimadora()
                hijo.cargar_vector_pesos(hijo_gen)
                nueva_poblacion.append(hijo)

            self.poblacion = nueva_poblacion

        assert self.mejor_red is not None
        self.mejor_red.guardar(ruta_guardado)
        if verbose:
            print(f"\nEntrenamiento finalizado. Mejor red guardada en: {ruta_guardado}")

        return self.mejor_red


if __name__ == "__main__":
    ag = AlgoritmoGenetico(
        tamano_poblacion=60,
        generaciones=120,
        escala_inicial=0.25,
        sigma_inicial=0.15,
        sigma_final=0.01,
        tasa_genes_ruido=0.30,
        n_escenarios_por_gen=6,
        pasos_episodio=300,
        peso_suavizado_delta=0.02,
        semilla=42,
    )
    ag.entrenar(ruta_guardado="mejor_red_estimadora.npz", verbose=True)