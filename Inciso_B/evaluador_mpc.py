import math
from pathlib import Path
from typing import Callable, Sequence
import numpy as np

try:
    from scipy.linalg import expm, solve_continuous_are
    from scipy.optimize import minimize
    TIENE_SCIPY = True
except ImportError:
    TIENE_SCIPY = False

from Algoritmo_Genetico import (
    DT_SIMULACION,
    GRAVEDAD,
    THETA_MAX,
    THETA_MIN,
    THETA_NOMINAL,
    RedEstimadora,
    paso_rk4,
)


def referencia_escalon_simple(t: float) -> np.ndarray:
    x_ref = 0.5 if t >= 0.4 else 0.0
    return np.array([x_ref, 0.0, 0.0, 0.0], dtype=np.float64)


def referencia_excitacion_persistente(t: float) -> np.ndarray:
    if t < 1.5:
        escalon = 0.45
    elif t < 3.2:
        escalon = -0.45
    elif t < 4.8:
        escalon = 0.35
    else:
        escalon = -0.25

    onda = 0.20 * math.sin(2.2 * t) + 0.10 * math.sin(5.1 * t)
    return np.array([escalon + onda, 0.0, 0.0, 0.0], dtype=np.float64)


def referencia_origen(t: float) -> np.ndarray:
    return np.zeros(4, dtype=np.float64)


class ControladorMPC:
    def __init__(
        self,
        horizonte: int = 25,
        dt: float = DT_SIMULACION,
        Q: Sequence[float] = (18.0, 1.5, 80.0, 8.0),
        R: float = 0.08,
        R_delta: float = 0.04,
        u_max: float = 35.0,
        tol_actualizacion: float = 1e-4,
    ) -> None:
        self.N = int(horizonte)
        self.dt = float(dt)
        self.Q = np.diag(np.asarray(Q, dtype=np.float64))
        self.R = float(R)
        self.R_delta = float(R_delta)
        self.u_max = float(u_max)
        self.tol_actualizacion = float(tol_actualizacion)

        self._theta_cache: np.ndarray | None = None
        self._u_prev: float = 0.0
        self._u_warm: np.ndarray = np.zeros(self.N, dtype=np.float64)

        self._Sx: np.ndarray | None = None
        self._Su: np.ndarray | None = None
        self._Q_bar: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._H: np.ndarray | None = None

    def reiniciar(self) -> None:
        self._u_prev = 0.0
        self._u_warm = np.zeros(self.N, dtype=np.float64)

    @staticmethod
    def _linealizar_continuo(params: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        M, m, L, r = float(params[0]), float(params[1]), float(params[2]), float(params[3])
        l = 0.5 * L
        J = m * ((r ** 2) / 4.0 + (L ** 2) / 3.0)
        det0 = (M + m) * J - (m * l) ** 2

        A_c = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -(m ** 2 * GRAVEDAD * l ** 2) / det0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, ((M + m) * m * GRAVEDAD * l) / det0, 0.0],
            ],
            dtype=np.float64,
        )
        B_c = np.array([[0.0], [J / det0], [0.0], [-(m * l) / det0]], dtype=np.float64)
        return A_c, B_c

    def _discretizar_zoh(self, A_c: np.ndarray, B_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if TIENE_SCIPY:
            M_aug = np.zeros((5, 5), dtype=np.float64)
            M_aug[:4, :4] = A_c * self.dt
            M_aug[:4, 4:5] = B_c * self.dt
            exp_M = expm(M_aug)
            return exp_M[:4, :4], exp_M[:4, 4:5]

        I4 = np.eye(4, dtype=np.float64)
        Adt = A_c * self.dt
        Adt2 = Adt @ Adt
        Adt3 = Adt2 @ Adt
        Adt4 = Adt3 @ Adt
        A_d = I4 + Adt + Adt2 / 2.0 + Adt3 / 6.0 + Adt4 / 24.0
        Phi = I4 * self.dt + (A_c * (self.dt ** 2)) / 2.0 + (Adt2 * self.dt) / 6.0
        B_d = Phi @ B_c
        return A_d, B_d

    def _calcular_costo_terminal(self, A_c: np.ndarray, B_c: np.ndarray) -> np.ndarray:
        if TIENE_SCIPY:
            try:
                return solve_continuous_are(A_c, B_c, self.Q, np.array([[self.R]]))
            except Exception:
                pass
        return 4.0 * self.Q

    def _actualizar_modelo_interno(self, theta: np.ndarray) -> None:
        if (
            self._theta_cache is not None
            and np.max(np.abs(theta - self._theta_cache)) < self.tol_actualizacion
        ):
            return

        A_c, B_c = self._linealizar_continuo(theta)
        A_d, B_d = self._discretizar_zoh(A_c, B_c)
        P_term = self._calcular_costo_terminal(A_c, B_c)

        N = self.N
        Sx = np.zeros((4 * N, 4), dtype=np.float64)
        Su = np.zeros((4 * N, N), dtype=np.float64)

        potencias_A = [np.eye(4, dtype=np.float64)]
        for _ in range(N):
            potencias_A.append(potencias_A[-1] @ A_d)

        for i in range(N):
            Sx[4 * i : 4 * (i + 1), :] = potencias_A[i + 1]
            for j in range(i + 1):
                col = (potencias_A[i - j] @ B_d).ravel()
                Su[4 * i : 4 * (i + 1), j] = col

        Q_bar = np.zeros((4 * N, 4 * N), dtype=np.float64)
        for i in range(N - 1):
            Q_bar[4 * i : 4 * (i + 1), 4 * i : 4 * (i + 1)] = self.Q
        Q_bar[4 * (N - 1) : 4 * N, 4 * (N - 1) : 4 * N] = P_term

        D = np.eye(N, dtype=np.float64)
        for i in range(1, N):
            D[i, i - 1] = -1.0

        R_bar = self.R * np.eye(N, dtype=np.float64)
        Rd_bar = self.R_delta * np.eye(N, dtype=np.float64)

        H = 2.0 * (Su.T @ Q_bar @ Su + R_bar + D.T @ Rd_bar @ D)
        H = 0.5 * (H + H.T)

        self._Sx = Sx
        self._Su = Su
        self._Q_bar = Q_bar
        self._D = D
        self._H = H
        self._theta_cache = theta.copy()

    def calcular_accion(
        self,
        estado_actual: np.ndarray,
        referencia_horizonte: np.ndarray,
        theta_modelo: np.ndarray,
    ) -> float:
        self._actualizar_modelo_interno(np.asarray(theta_modelo, dtype=np.float64))
        assert self._Sx is not None and self._Su is not None
        assert self._Q_bar is not None and self._D is not None and self._H is not None

        x0 = np.asarray(estado_actual, dtype=np.float64)
        ref = np.asarray(referencia_horizonte, dtype=np.float64)
        if ref.ndim == 1:
            X_ref = np.tile(ref, self.N)
        else:
            X_ref = ref.reshape(-1)

        d0 = np.zeros(self.N, dtype=np.float64)
        d0[0] = self._u_prev

        err_libre = self._Sx @ x0 - X_ref
        f = 2.0 * (self._Su.T @ (self._Q_bar @ err_libre) - self.R_delta * (self._D.T @ d0))

        H = self._H

        if TIENE_SCIPY:
            bounds = [(-self.u_max, self.u_max)] * self.N
            res = minimize(
                fun=lambda u: 0.5 * float(u @ (H @ u)) + float(f @ u),
                x0=self._u_warm,
                jac=lambda u: H @ u + f,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 60, "ftol": 1e-9},
            )
            u_opt = res.x
        else:
            L_lip = float(np.linalg.norm(H, 2)) + 1e-8
            alpha = 1.0 / L_lip
            u_opt = self._u_warm.copy()
            for _ in range(80):
                u_opt = np.clip(u_opt - alpha * (H @ u_opt + f), -self.u_max, self.u_max)

        u0 = float(np.clip(u_opt[0], -self.u_max, self.u_max))
        self._u_prev = u0
        self._u_warm = np.concatenate([u_opt[1:], u_opt[-1:]])
        return u0


class EvaluadorSimulacionMPC:
    def __init__(
        self,
        red_estimadora: RedEstimadora | str | Path | None = None,
        controlador_mpc: ControladorMPC | None = None,
        dt: float = DT_SIMULACION,
    ) -> None:
        if isinstance(red_estimadora, (str, Path)):
            self.red: RedEstimadora | None = RedEstimadora.cargar(red_estimadora)
        else:
            self.red = red_estimadora

        self.mpc = controlador_mpc if controlador_mpc is not None else ControladorMPC(dt=dt)
        self.dt = float(dt)

    def simular(
        self,
        theta_real: Sequence[float],
        modo: str = "adaptativo",
        theta_inicial: Sequence[float] = THETA_NOMINAL,
        estado_inicial: Sequence[float] = (0.0, 0.0, 0.08, 0.0),
        pasos: int = 300,
        fn_referencia: Callable[[float], np.ndarray] = referencia_excitacion_persistente,
        ruido_medicion_std: Sequence[float] | None = None,
        semilla: int = 123,
    ) -> dict:
        modo = modo.lower().strip()
        if modo not in {"adaptativo", "nominal", "oraculo"}:
            raise ValueError("El modo debe ser 'adaptativo', 'nominal' u 'oraculo'.")
        if modo == "adaptativo" and self.red is None:
            raise ValueError("Se requiere una RedEstimadora cargada para el modo 'adaptativo'.")

        rng = np.random.default_rng(semilla)
        self.mpc.reiniciar()

        theta_real_arr = np.asarray(theta_real, dtype=np.float32)
        if modo == "oraculo":
            theta_est = theta_real_arr.copy()
        else:
            theta_est = np.asarray(theta_inicial, dtype=np.float32).copy()
            theta_est[3] = THETA_NOMINAL[3]

        estado_real = np.asarray(estado_inicial, dtype=np.float64).copy()
        std_ruido = (
            np.asarray(ruido_medicion_std, dtype=np.float64)
            if ruido_medicion_std is not None
            else None
        )

        tiempos = np.arange(pasos, dtype=np.float64) * self.dt
        hist_real = np.zeros((pasos, 4), dtype=np.float64)
        hist_pred = np.zeros((pasos, 4), dtype=np.float64)
        hist_ref = np.zeros((pasos, 4), dtype=np.float64)
        hist_u = np.zeros(pasos, dtype=np.float64)
        hist_theta = np.zeros((pasos, 4), dtype=np.float32)
        hist_err_pred = np.zeros((pasos, 4), dtype=np.float64)
        hist_deltas = np.zeros((pasos, 4), dtype=np.float32)

        estable = True
        pasos_validos = pasos

        for k in range(pasos):
            t = tiempos[k]

            if std_ruido is not None:
                estado_medido = estado_real + rng.normal(0.0, std_ruido)
            else:
                estado_medido = estado_real.copy()

            ref_horizonte = np.array(
                [fn_referencia(t + j * self.dt) for j in range(self.mpc.N)],
                dtype=np.float64,
            )
            ref_actual = ref_horizonte[0]

            u = self.mpc.calcular_accion(estado_medido, ref_horizonte, theta_est)

            estado_pred = paso_rk4(estado_medido, u, theta_est, self.dt)
            estado_siguiente = paso_rk4(estado_real, u, theta_real_arr, self.dt)

            if std_ruido is not None:
                estado_siguiente_medido = estado_siguiente + rng.normal(0.0, std_ruido)
            else:
                estado_siguiente_medido = estado_siguiente

            error_pred = estado_siguiente_medido - estado_pred

            if modo == "adaptativo":
                assert self.red is not None
                delta = self.red.predecir(u, estado_siguiente_medido, error_pred, theta_est)
                theta_est = np.clip(theta_est + delta, THETA_MIN, THETA_MAX)
                theta_est[3] = THETA_NOMINAL[3]
            else:
                delta = np.zeros(4, dtype=np.float32)

            hist_real[k] = estado_real
            hist_pred[k] = estado_pred
            hist_ref[k] = ref_actual
            hist_u[k] = u
            hist_theta[k] = theta_est
            hist_err_pred[k] = error_pred
            hist_deltas[k] = delta

            estado_real = estado_siguiente

            if abs(estado_real[2]) > 1.3 or abs(estado_real[0]) > 5.0:
                estable = False
                pasos_validos = k + 1
                break

        tiempos = tiempos[:pasos_validos]
        hist_real = hist_real[:pasos_validos]
        hist_pred = hist_pred[:pasos_validos]
        hist_ref = hist_ref[:pasos_validos]
        hist_u = hist_u[:pasos_validos]
        hist_theta = hist_theta[:pasos_validos]
        hist_err_pred = hist_err_pred[:pasos_validos]
        hist_deltas = hist_deltas[:pasos_validos]

        tramo_final = max(1, pasos_validos // 4)
        theta_final_media = np.mean(hist_theta[-tramo_final:, :3], axis=0)
        err_abs_final = np.abs(theta_final_media - theta_real_arr[:3])
        err_rel_final_pct = 100.0 * err_abs_final / theta_real_arr[:3]

        metricas = {
            "modo": modo,
            "estable": estable,
            "pasos_completados": pasos_validos,
            "rmse_x": float(np.sqrt(np.mean((hist_real[:, 0] - hist_ref[:, 0]) ** 2))),
            "rmse_theta_deg": float(np.degrees(np.sqrt(np.mean(hist_real[:, 2] ** 2)))),
            "rmse_pred_total": float(np.sqrt(np.mean(hist_err_pred ** 2))),
            "energia_u": float(np.sum(hist_u ** 2) * self.dt),
            "theta_final_estimado": theta_final_media,
            "mae_M": float(err_abs_final[0]),
            "mae_m": float(err_abs_final[1]),
            "mae_L": float(err_abs_final[2]),
            "err_rel_pct_M": float(err_rel_final_pct[0]),
            "err_rel_pct_m": float(err_rel_final_pct[1]),
            "err_rel_pct_L": float(err_rel_final_pct[2]),
        }

        return {
            "t": tiempos,
            "estados_reales": hist_real,
            "estados_predichos": hist_pred,
            "referencias": hist_ref,
            "acciones_u": hist_u,
            "theta_estimado": hist_theta,
            "theta_real": theta_real_arr,
            "errores_prediccion": hist_err_pred,
            "deltas_theta": hist_deltas,
            "metricas": metricas,
        }

    def comparar_modos(
        self,
        theta_real: Sequence[float],
        theta_inicial: Sequence[float] = THETA_NOMINAL,
        estado_inicial: Sequence[float] = (0.0, 0.0, 0.08, 0.0),
        pasos: int = 300,
        fn_referencia: Callable[[float], np.ndarray] = referencia_excitacion_persistente,
        ruido_medicion_std: Sequence[float] | None = None,
    ) -> dict[str, dict]:
        modos = ["nominal", "adaptativo", "oraculo"] if self.red is not None else ["nominal", "oraculo"]
        resultados = {}
        for m in modos:
            resultados[m] = self.simular(
                theta_real=theta_real,
                modo=m,
                theta_inicial=theta_inicial,
                estado_inicial=estado_inicial,
                pasos=pasos,
                fn_referencia=fn_referencia,
                ruido_medicion_std=ruido_medicion_std,
            )
        return resultados
