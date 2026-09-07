# Experimentos del backend Qoala

> Backend: **Qoala (simulación NetSquid)**. Requiere el entorno conda `qoala`
> (Python 3.11, netqasm ≥2.0). No corre en el entorno `squidasm`.

Experimentos en este directorio:

1. **Propagación de parámetros de hardware** (este archivo) — valida que el ruido
   del qdevice (T1/T2, tiempos, despolarización de puerta) se propaga fielmente a
   través de la traducción de NetQMPI, comparando contra un control Qoala nativo.
2. **Fidelidad del EPR en la teleportación** →
   [README_epr_fidelity.md](README_epr_fidelity.md) — con qdevice perfecto, barre
   la fidelidad del par EPR de `qsend`/`qrecv` y la compara con el enlace perfecto.
3. **Scheduling / multitarea** →
   [README_scheduling.md](README_scheduling.md) — dos programas NetQMPI
   compartiendo un nodo; el scheduler de Qoala reduce el makespan (−18.9%) sin
   degradar la fidelidad (con Gantt CPS/QPS del intercalado).

---

# Experimento 1: propagación de parámetros de hardware Qoala vía NetQMPI

## Objetivo

Validar que los parámetros de hardware configurables del qdevice de
NetSquid/Qoala (T1, T2, tiempos de puerta/init/medida y probabilidades de
despolarización) **se propagan correctamente** cuando un programa se ejecuta a
través del backend de NetQMPI, y que la traducción del `QoalaCircuitAdapter` **no
los pierde ni los simplifica**.

La métrica es la **fidelidad de la traducción**: se ejecuta el mismo circuito
(a) vía NetQMPI→Qoala y (b) como programa Qoala nativo hecho a mano, con el
**mismo qdevice** y un **enlace de entrelazamiento perfecto**, y se compara la
fidelidad observada en ambos caminos. Si el adaptador es fiel, ambas curvas
deben coincidir dentro del error estadístico del número de shots.

## Sonda de fidelidad (circuito)

Teleportación de superposición distribuida con **lectura en base X**
([apps/dist_superposition_xbasis.py](apps/dist_superposition_xbasis.py)):

1. Rank 0 prepara `|+⟩` con `H`.
2. Lo teleporta a Rank 1 con `qsend`/`qrecv` (EPR + BSM + correcciones).
3. Rank 1 aplica `H` (rota a base X) y mide.

Sin ruido, `|+⟩` teleportado → `H` → `|0⟩` → resultado **0 determinista**. La
fidelidad se estima como `F = P(outcome = 0)`.

> **Por qué base X y no el `1_send_recv` original.** Medir `|+⟩` directamente en Z
> da 50/50 *incluso sin ruido*, y la despolarización y el dephasing (T2) dejan
> esa proporción intacta → curvas planas, inútiles como fidelidad. Midiendo en la
> base propia del estado (X), la fidelidad va de 1.0 (sin ruido) a 0.5
> (totalmente despolarizado), sensible a T1, T2 y despolarización. Esto solo
> cambia el `app.py` del experimento (una `H` extra); **no toca el SDK**.

## Control nativo

[native/dist_superposition_alice.iqoala](native/dist_superposition_alice.iqoala)
y [native/dist_superposition_bob.iqoala](native/dist_superposition_bob.iqoala)
son una implementación **independiente** del mismo circuito, escrita a mano en
`.iqoala` y ejecutada con la API propia de qoala-sim
(`run_n_node_app`). Ambos caminos usan **exactamente la misma `TopologyConfig`**
(construida con `QoalaExecutorAdapter._build_topology`), de modo que cualquier
diferencia de fidelidad proviene solo de la traducción del adaptador, no del
hardware.

## Parámetros barridos

Enlace **perfecto** en todo el experimento (aísla el ruido al qdevice). Dos
barridos 1-D:

| Barrido | Parámetro | Valores | Resto |
|---|---|---|---|
| Memoria | `T1 = T2` (ns) | 1e12, 3e11, 1e11, 3e10, 1e10, 3e9, 1e9, 3e8, 1e8 | puertas perfectas (depolar 0) |
| Puerta | `single_qubit_gate_depolar_prob` | 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5 | memoria perfecta (T1=T2=0) |

Shots por punto: **1000** (por defecto). Tiempos de puerta por defecto:
1q = 5000 ns, 2q = 200000 ns, init = measure = 5000 ns.

## Cómo se pasan los parámetros de hardware

Sin tocar el SDK agnóstico. Solo por el Runtime del backend Qoala, mediante el
YAML unificado de `--config` (bloque `qoala.hardware`):

- **CLI:** `netqmpi -n 2 app.py --qoala --config config.yaml`
- **Programático:** `QoalaRunConfig(hw_config=QoalaQDeviceConfig(...))`

```yaml
# config.yaml
qoala:
  hardware:
    t1: 0
    t2: 0
    single_qubit_gate_time: 5000
    two_qubit_gate_time: 200000
    init_time: 5000
    measure_time: 5000
    single_qubit_gate_depolar_prob: 0.0
    two_qubit_gate_depolar_prob: 0.0
```

Ejemplo completo en [configs/qoala_example.yaml](configs/qoala_example.yaml).

## Ejecutar

```sh
conda run -n qoala python scripts/experiments/qoala_hw_sweep.py --shots 1000 --seed 7
# salida: results/qoala_hw_sweep.csv, results/fidelity_vs_t1t2.png, results/fidelity_vs_depolar.png
conda run -n qoala python scripts/experiments/qoala_hw_sweep.py --quick   # smoke run rápido
```

## Qué se esperaba observar

- **Depolar:** `F` desciende ~linealmente de 1.0 (prob 0) a 0.5 (prob 0.5,
  estado maximalmente mezclado).
- **T1/T2:** `F` ≈ 1.0 para T grande (poca decoherencia) y desciende hacia 0.5
  al reducir T. La rodilla cae cerca de la **duración total del protocolo**
  (~1e9 ns, dominada por la latencia clásica de las correcciones de
  teleportación), no de los tiempos de puerta.
- **NetQMPI vs nativo:** ambas curvas deben solaparse dentro del error
  estadístico (para 1000 shots, error estándar de una proporción ≤ 0.016; banda
  3σ combinada ≈ 0.067).

## Qué se observó

Ejecución con **1000 shots/punto**, seed NetQMPI = 7 (el path nativo re-siembra
internamente). Datos completos en [results/qoala_hw_sweep.csv](results/qoala_hw_sweep.csv);
gráficas en [results/fidelity_vs_t1t2.png](results/fidelity_vs_t1t2.png) y
[results/fidelity_vs_depolar.png](results/fidelity_vs_depolar.png).

**Barrido T1/T2** (puertas perfectas): decaimiento sigmoidal monótono, rodilla en
T ≈ 1e9 ns (la duración del protocolo), como se predijo.

| T1=T2 (ns) | F NetQMPI | F nativo | \|Δ\| |
|---|---|---|---|
| 1e12 | 0.998 | 1.000 | 0.002 |
| 1e11 | 0.987 | 0.992 | 0.005 |
| 1e10 | 0.908 | 0.915 | 0.007 |
| 3e9  | 0.763 | 0.762 | 0.001 |
| 1e9  | 0.591 | 0.578 | 0.013 |
| 3e8  | 0.530 | 0.506 | 0.024 |
| 1e8  | 0.530 | 0.493 | 0.037 |

**Barrido depolarización 1q** (memoria perfecta): decaimiento monótono de 1.0 a ~0.5.

| depolar 1q | F NetQMPI | F nativo | \|Δ\| |
|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.000 |
| 0.05 | 0.905 | 0.903 | 0.002 |
| 0.10 | 0.829 | 0.833 | 0.004 |
| 0.20 | 0.705 | 0.716 | 0.011 |
| 0.30 | 0.638 | 0.620 | 0.018 |
| 0.40 | 0.597 | 0.565 | 0.032 |
| 0.50 | 0.561 | 0.519 | 0.042 |

**Veredicto:** `max |F_netqmpi − F_native| = 0.042` (en depolar 1q = 0.5), frente a
una banda estadística 3σ ≈ **0.067** en ese punto. La discrepancia máxima está
**DENTRO** del error estadístico esperado para 1000 shots.

**Conclusión:** los parámetros de hardware del qdevice (T1, T2, tiempos,
despolarización) **se propagan correctamente** a través del backend de NetQMPI: las
curvas de fidelidad NetQMPI→Qoala y Qoala-nativo coinciden dentro del error de
muestreo en toda la rejilla. La traducción del `QoalaCircuitAdapter` no pierde ni
simplifica el ruido de hardware.

> Nota: las fidelidades NetQMPI salen ligeramente por encima de las nativas en el
> extremo de mucho ruido (p.ej. 0.561 vs 0.519 a depolar=0.5); la diferencia sigue
> por debajo de 3σ y es compatible con ruido de muestreo (los dos caminos usan
> semillas independientes). Ambas implementaciones ejecutan el mismo número de
> puertas ruidosas, así que no hay diferencia sistemática esperada.

## Limitaciones documentadas

- **Qoala es solo simulador** (NetSquid); no hay hardware real.
- `init_time` y `measure_time` no son configurables por separado en los factory
  uniformes de Qoala (comparten duración con las puertas de 1 qubit). Aquí se
  exponen por separado construyendo la `TopologyConfig` por-instrucción
  (`QoalaExecutorAdapter._build_topology`).
- El ruido de despolarización se aplica a las **puertas** de 1 y 2 qubits;
  `INSTR_INIT`/`INSTR_MEASURE` llevan su duración pero sin error de
  despolarización (no hay modelo de error de lectura tipo readout-flip separado).
- La rodilla del barrido T1/T2 depende de la latencia clásica del protocolo
  (fijada a 1e9 ns, igual en ambos caminos), no solo del qdevice.
