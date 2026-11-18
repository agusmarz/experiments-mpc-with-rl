"""Eje C: política que elige el ajuste del MPC según el estado, aprendida con
Fitted Q Iteration (FQI) y Extra-Trees como regresor.

Formulación (ver el planteo del informe)
    estado    s = [x, x_d, theta, theta_d], medido cada M pasos de control
    acción    a en {0, ..., K-1}, el índice de un ajuste (Q_mpc, R_mpc, N)
    recompensa r = -min(costo_intervalo / c_barra, 1), con el costo medido
              por el env con su métrica FIJA p.Q, p.R (no la del ajuste)
    falla     el péndulo pasa theta_corte, el carro sale del riel o Ipopt
              falla; la transición es terminal con r = -1 / (1 - gamma)

FQI
    0. simular UNA vez y guardar transiciones (s, a, r, s', d)
    1. Q_0 = 0, entonces el primer objetivo es y = r
    2. ajustar Q_{k+1} por regresión de Q(s_i, a_i) sobre y_i
    3. y_i = r_i + gamma (1 - d_i) max_a' Q_{k+1}(s'_i, a'), volver a 2
        donde d_i = 1 si la transición es terminal, 0 si no lo es.

"""

import dataclasses
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulador"))
from pendulo_env import PenduloEnv, ParametrosPlanta  # noqa: E402
from mpc_no_lineal import MPCNoLineal, costo_terminal_cuadratico, riccati  # noqa: E402

# El espacio de acciones es discreto, viene dado por estos 4 ajustes del MPC
# El agente tiene que elegir cada M pasos, cuál de estos ajustes usar.
AJUSTES = [
    {"nombre": "base",        "Q": [10.0, 1.0, 100.0, 1.0], "R": 0.1, "N": 10},
    {"nombre": "agresivo",    "Q": [10.0, 1.0, 400.0, 4.0], "R": 0.1, "N": 10},
    {"nombre": "conservador", "Q": [10.0, 1.0, 100.0, 1.0], "R": 1.0, "N": 10},
    {"nombre": "horizonte",   "Q": [10.0, 1.0, 100.0, 1.0], "R": 0.1, "N": 20},
]


@dataclass
class ConfigFQI:
    M: int = 10        # pasos de control entre decisiones del agente               
    gamma: float = 0.95 
    n_estados: int = 2000             
    n_episodios: int = 200           
    decisiones_por_episodio: int = 25
    iteraciones: int = 60
    congelar_desde: int = 20          # a partir de acá los umbrales de los árboles se congelan y solo se recalculan los valores de las hojas
    n_arboles: int = 50
    min_muestras_corte: int = 4
    percentil_c_barra: float = 99.0
    semilla: int = 0
    caja: np.ndarray = field(default_factory=lambda: np.array([0.45, 2.0, 0.8, 4.0]))


# Ajustes del MPC

def crear_mpcs(p: ParametrosPlanta, ajustes=AJUSTES):
    """Un MPC compilado por ajuste. Se usa una COPIA de p con otros pesos:
    el env sigue usando p.Q y p.R, que son la métrica fija de evaluación.
    Cada ajuste lleva su propio costo terminal x^T P x, con P de Riccati para
    SUS pesos; sin costo terminal un horizonte de 10 pasos (0,2 s) es miope y
    el péndulo se cae incluso desde theta0 = 0,1 rad."""
    mpcs = []
    for a in ajustes:
        p_ajuste = dataclasses.replace(p, Q=np.diag(a["Q"]), R=a["R"])
        P, _ = riccati(p_ajuste)
        mpcs.append(MPCNoLineal(p_ajuste, N=a["N"], costo_terminal=costo_terminal_cuadratico(P)))
    return mpcs


# Simulación

def simular_intervalo(mpc, env, estado, M, reiniciar=True):
    """Aplica un ajuste durante M pasos desde estado.
    Devuelve (costo con la métrica fija, estado siguiente, falla, pasos simulados)."""
    x, _ = env.reset(options={"estado": estado})
    if reiniciar:
        mpc.reiniciar()
    costo = 0.0
    for paso in range(M):
        F = mpc.solve(x)
        x, _, terminado, _, info = env.step([F])
        costo += info["costo"]
        if terminado or not mpc.ultimo_exito:
            return costo, x, True, paso + 1
    return costo, x, False, M


def recolectar_sincronico(mpcs, env, cfg, rng):
    """Cada estado de la caja se prueba con los K ajustes (como el eje A, pero con acciones)."""
    estados = rng.uniform(-cfg.caja, cfg.caja, size=(cfg.n_estados, 4))
    datos, pasos = [], 0
    for x0 in estados:
        for a, mpc in enumerate(mpcs):
            costo, x_sig, falla, n = simular_intervalo(mpc, env, x0, cfg.M)
            datos.append((x0, a, costo, x_sig, falla))
            pasos += n
    return datos, pasos


def recolectar_episodios(mpcs, env, cfg, rng):
    """Episodios con un ajuste al azar en cada decisión, para cubrir los estados
    por donde pasan las trayectorias de recuperación (donde se evalúa max Q(s', .))."""
    datos, pasos = [], 0
    for _ in range(cfg.n_episodios):
        x = rng.uniform(-cfg.caja, cfg.caja)
        a_previa = None
        for _ in range(cfg.decisiones_por_episodio):
            a = int(rng.integers(len(mpcs)))
            # Mismo ajuste que antes, se conserva el warm start; si cambia, se reinicia
            costo, x_sig, falla, n = simular_intervalo(mpcs[a], env, x, cfg.M, reiniciar=(a != a_previa))
            datos.append((x, a, costo, x_sig, falla))
            pasos += n
            if falla:
                break
            x, a_previa = x_sig, a
    return datos, pasos


def a_arreglos(datos):
    S = np.array([d[0] for d in datos])
    A = np.array([d[1] for d in datos], dtype=int)
    C = np.array([d[2] for d in datos])
    S2 = np.array([d[3] for d in datos])
    D = np.array([d[4] for d in datos], dtype=float)
    return S, A, C, S2, D


def espejar(S, A, C, S2, D):
    """El sistema es impar,
    así que (s, a, c, s', d) tiene su espejo (-s, a, c, -s', d)."""
    return (np.vstack([S, -S]), np.concatenate([A, A]), np.concatenate([C, C]),
            np.vstack([S2, -S2]), np.concatenate([D, D]))


def recompensas(C, A, D, cfg, ajustes=AJUSTES):
    """r en [-1, 0] por intervalo y r = -1/(1 - gamma) en la falla.
    c_barra se toma como un percentil alto del costo de los intervalos sin falla."""
    costo = C 
    c_barra = np.percentile(costo[D == 0], cfg.percentil_c_barra)
    R = -np.minimum(costo / c_barra, 1.0)
    R[D == 1] = -1.0 / (1.0 - cfg.gamma)
    return R, c_barra


# FQI

class BosqueCongelado:
    """Estructura FIJA de un ExtraTreesRegressor ya ajustado; solo se recalculan
    los valores de las hojas (promedio de los objetivos que caen en cada una).
    Con la estructura fija los pesos k(x, x_i) dependen solo de las entradas,
    el operador de FQI es una contracción de módulo gamma y la iteración converge.
    Si se reconstruye el bosque en cada iteración los cortes dependen de y, y esa
    garantía formal se pierde"""

    def __init__(self, bosque, X):
        self.bosque = bosque
        self.hojas_X = bosque.apply(X)                      # (n, M) hoja de cada muestra en cada árbol
        self.n_nodos = [arbol.tree_.node_count for arbol in bosque.estimators_]
        self.valores = None

    def ajustar(self, y):
        self.valores = []
        for m, n_nodos in enumerate(self.n_nodos):
            hojas = self.hojas_X[:, m]
            suma = np.bincount(hojas, weights=y, minlength=n_nodos)
            cuenta = np.bincount(hojas, minlength=n_nodos)
            self.valores.append(suma / np.maximum(cuenta, 1))
        return self

    def predict(self, X):
        hojas = self.bosque.apply(X)
        return np.mean([v[hojas[:, m]] for m, v in enumerate(self.valores)], axis=0)


def entrada(S, A, K):
    return np.hstack([S, np.eye(K)[A]])


def fqi(S, A, R, S2, D, K, cfg, verbose=True):
    X = entrada(S, A, K)
    X_sig = [entrada(S2, np.full(len(S2), b), K) for b in range(K)]
    y = R.copy()                                   # Q_0 = 0
    historial = []
    congelado = None
    for k in range(cfg.iteraciones):
        if k < cfg.congelar_desde:
            # Bosque nuevo desde cero: fit descarta todo lo anterior
            modelo = ExtraTreesRegressor(n_estimators=cfg.n_arboles,
                                         min_samples_split=cfg.min_muestras_corte,
                                         n_jobs=-1, random_state=cfg.semilla + k)
            modelo.fit(X, y)                       # Q_{k+1}, con objetivos congelados
        else:
            if congelado is None:
                congelado = BosqueCongelado(modelo, X)
            modelo = congelado.ajustar(y)          # mismos cortes, hojas con los objetivos nuevos
        q_sig = np.column_stack([modelo.predict(x) for x in X_sig])
        y_nuevo = R + cfg.gamma * (1.0 - D) * q_sig.max(axis=1)
        cambio = float(np.max(np.abs(y_nuevo - y)))
        y = y_nuevo
        historial.append({"iteracion": k, "cambio_max_objetivos": cambio, "congelado": congelado is not None})
        if verbose:
            print(f"it {k:3d}  max|y_k+1 - y_k| {cambio:.4f}", flush=True)
    return modelo, historial


def elegir_ajuste(modelo, estado, K):
    x = np.repeat(np.atleast_2d(estado), K, axis=0)
    return int(np.argmax(modelo.predict(entrada(x, np.arange(K), K))))


# Evaluación

def lazo_cerrado(politica, mpcs, env, estado0, M, pasos=250):
    """politica(estado) -> índice de ajuste, consultada cada M pasos.
    Devuelve (recuperado, costo acumulado, cantidad de cambios de ajuste)."""
    x, _ = env.reset(options={"estado": estado0})
    costo, cambios, a_previa = 0.0, 0, None
    for t in range(pasos):
        if t % M == 0:
            a = politica(x)
            if a != a_previa:
                mpcs[a].reiniciar()
                cambios += a_previa is not None
            a_previa = a
        F = mpcs[a].solve(x)
        x, _, terminado, _, info = env.step([F])
        costo += info["costo"]
        if terminado:
            return False, costo, cambios
    cerca = abs(x[2]) < 0.05 and abs(x[0]) < 0.1
    return bool(cerca), costo, cambios


def mapa_politica(modelo, K, thetas, velocidades):
    """Ajuste elegido en el plano (theta0, x_d0), el mismo del eje A."""
    mapa = np.zeros((len(velocidades), len(thetas)), dtype=int)
    for i, v in enumerate(velocidades):
        for j, th in enumerate(thetas):
            mapa[i, j] = elegir_ajuste(modelo, [0.0, v, th, 0.0], K)
    return mapa


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="FQI para elegir el ajuste del MPC (eje C)")
    ap.add_argument("--estados", type=int, default=2000)
    ap.add_argument("--episodios", type=int, default=200)
    ap.add_argument("--iteraciones", type=int, default=60)
    ap.add_argument("--congelar_desde", type=int, default=20)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()

    cfg = ConfigFQI(n_estados=a.estados, n_episodios=a.episodios,
                    iteraciones=a.iteraciones, congelar_desde=a.congelar_desde,
                    semilla=a.semilla)
    rng = np.random.default_rng(cfg.semilla)
    p = ParametrosPlanta()
    env = PenduloEnv(p)
    mpcs = crear_mpcs(p)
    K = len(mpcs)

    t0 = time.perf_counter()
    datos_s, pasos_s = recolectar_sincronico(mpcs, env, cfg, rng)
    datos_e, pasos_e = recolectar_episodios(mpcs, env, cfg, rng)
    t_datos = time.perf_counter() - t0
    S, A, C, S2, D = espejar(*a_arreglos(datos_s + datos_e))
    R, c_barra = recompensas(C, A, D, cfg)
    print(f"transiciones {len(S)} (con espejo)  terminales {D.mean():.3f}  "
          f"pasos simulados {pasos_s + pasos_e}  c_barra {c_barra:.1f}  t {t_datos:.0f}s")

    # fqi desde el módulo importado, para que el pickle referencie
    # fqi_ajuste.BosqueCongelado y no __main__.BosqueCongelado
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import fqi_ajuste
    modelo, historial = fqi_ajuste.fqi(S, A, R, S2, D, K, cfg)

    thetas = np.linspace(-0.7, 0.7, 9)
    velocidades = np.linspace(-1.5, 1.5, 9)
    mapa = mapa_politica(modelo, K, thetas, velocidades)
    print("ajuste elegido en (theta0, x_d0):", ", ".join(f"{i}={aj['nombre']}" for i, aj in enumerate(AJUSTES)))
    for fila in mapa[::-1]:
        print("    " + "".join(str(v) for v in fila))

    salida = a.salida or str(Path(__file__).parent / f"fqi_s{cfg.semilla}.pkl")
    with open(salida, "wb") as f:
        pickle.dump({"modelo": modelo, "cfg": dataclasses.asdict(cfg), "ajustes": AJUSTES, "historial": historial,
                     "c_barra": c_barra, "pasos_simulados": pasos_s + pasos_e,
                     "t_datos_s": t_datos}, f)
    print("guardado en", salida)
