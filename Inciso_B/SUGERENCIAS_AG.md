# Sugerencias para los operadores del algoritmo genético

Análisis de cómo `Algoritmo_Genetico.py` mezcla y muta las redes `RedEstimadora`
(MLP 12 → 10 → 10 → 4 con tanh, 284 pesos), y qué cambiaría. Nada de esto está
implementado todavía: son propuestas para probar contra la versión actual.

## Cómo se genera hoy cada hijo

1. **Selección:** torneo de 3 para cada padre.
2. **Cruce uniforme gen a gen, siempre:** cada uno de los 284 pesos sale al azar
   del padre 1 o del padre 2.
3. **Mutación por permutación** (`prob_permutacion = 0.3`): intercambia 5 pares
   de pesos elegidos al azar en todo el vector.
4. **Mutación gaussiana** (`prob_mutacion_ruido = 0.8`): suma N(0, 0.05²) al 25 %
   de los pesos, unos 71 por hijo.

Los 2 élites pasan sin cambios y se reevalúan en cada generación.

## Problema 1: el cruce gen a gen rompe neuronas

En una red, la unidad con sentido es la **neurona**: sus pesos de entrada, su bias
y sus pesos de salida hacia la capa siguiente funcionan juntos. Además, las
neuronas ocultas no tienen un orden fijo. La neurona 3 de un padre puede cumplir
el mismo papel que la neurona 7 del otro (el problema de las *convenciones en
competencia*). Y con tanh, cambiar el signo de todos los pesos de una neurona da
la misma función.

El cruce uniforme ignora todo eso. El hijo puede terminar con la mitad de las
entradas de una neurona del padre A, la otra mitad del padre B y la salida de
cualquiera de los dos. Esa neurona en general no se parece a ninguna de las
originales.

Hoy no es catastrófico: con población 30 y elitismo, los padres suelen descender
de ancestros comunes y están parcialmente alineados. Pero muchos hijos terminan
siendo, en la práctica, mutaciones grandes al azar. Eso coincide con lo que se ve
en los logs, donde el promedio de la población queda muy por debajo del mejor.

**Propuesta: cruce por neurona.** Para cada neurona oculta, el hijo hereda el
bloque completo de un padre o del otro: fila de pesos de entrada, bias y columna
de pesos de salida. La capa de salida se trata igual, neurona por neurona.

**Propuesta: cruzar solo con probabilidad `prob_cruce` (~0.5).** Si no se cruza,
el hijo es una copia de un padre más la mutación. Así hay hijos que exploran
cerca de soluciones que ya funcionan.

## Problema 2: el intercambio de genes es una mutación sin sentido

Intercambiar dos pesos al azar puede llevar, por ejemplo, un peso de la entrada
`e_pred` de la primera capa a un bias de la capa de salida. Tienen escalas y
funciones distintas, así que es un salto aleatorio grande en el 30 % de los hijos.
No es una permutación de neuronas, que sí dejaría la función intacta. El ruido
gaussiano ya cumple el papel de explorar, y de forma controlada.

**Propuesta: sacarlo** (`prob_permutacion = 0`).

## Problema 3: el ruido gaussiano tiene σ fijo

Con σ = 0.05 sobre ~71 pesos por hijo, al principio la búsqueda puede quedarse
corta y al final ser demasiado brusca para ajustes finos.

**Propuestas, de más simple a más elaborada:**

- **σ que decrece con las generaciones**, por ejemplo de 0.1 a 0.01.
- **Regla del 1/5:** si más de 1/5 de los hijos supera a sus padres, subir σ; si
  menos, bajarlo.
- **σ autoadaptativo:** cada individuo lleva su propio σ como un gen más, que
  también muta y se hereda.

## Lo que ya está bien

- Todos los individuos de una generación se evalúan en **los mismos escenarios**,
  así que las comparaciones dentro de la generación son justas.
- Los élites se **reevalúan** cada generación, así que no arrastran un puntaje
  viejo que tuvo suerte.
- El torneo de 3 da una presión de selección moderada, razonable con una fitness
  ruidosa.
- La validación sobre escenarios fijos decide qué red se entrega, no el último
  ganador ruidoso.

## Alternativa más fuerte: CMA-ES

Para unos cientos de parámetros continuos con fitness ruidosa, **CMA-ES** suele
superar claramente a un AG con cruce. Sigue siendo evolutivo y sin gradiente: no
cruza individuos, sino que aprende la forma de la distribución de la que muestrea
(con 284 parámetros, una covarianza de 284 × 284 es manejable). Población típica:
4 + 3·ln(284) ≈ 21.

Si la consigna exige específicamente un *algoritmo genético*, conviene quedarse
en el AG con los cambios de arriba y, como mucho, usar CMA-ES de referencia.

## Cómo decidir: medir, no suponer

1. Agregar las variantes como opciones del constructor, sin cambiar el
   comportamiento por defecto: `tipo_cruce="uniforme" | "neurona"`, `prob_cruce`,
   `prob_permutacion` y un esquema para σ.
2. Correr la versión actual y la variante (cruce por neurona, sin permutación,
   σ decreciente) con el **mismo presupuesto de evaluaciones** y 2 a 3 semillas
   cada una.
3. Comparar las curvas de **validación**, que usan siempre los mismos 10
   escenarios y por eso son comparables entre corridas.
4. Quedarse con la variante que llegue más alto en validación. Si las diferencias
   quedan dentro de la dispersión entre semillas, no hay evidencia para cambiar.
