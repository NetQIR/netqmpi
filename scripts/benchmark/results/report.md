## Portability matrix

| app | aer | cunqa | qoala | netqasm |
|---|---|---|---|---|
| cascade | OK | OK | OK | OK |
| ghz | OK | OK | error | OK |
| qft | OK | OK | n/i | n/i |
| qft_telegate | OK | OK | n/i | n/i |

`OK` = echo exact on every configuration tried. `n/i` = the adapter raises `NotImplementedError`. `WRONG` = completed but returned the wrong answer on a backend that has no noise model. A bare `F=` range is decoherence on a backend that models hardware.

## Overhead by backend

| backend | configs | cold total | warm total | warm NetQMPI | warm share | median ops |
|---|---|---|---|---|---|---|
| aer | 60 | 0.309 s | 0.0148 s | 1.38 ms | 9.22% | 57 |
| cunqa | 24 | 4.795 s | 4.6838 s | 3.42 ms | 0.07% | 32 |
| qoala | 3 | 2.840 s | 2.3249 s | 0.16 ms | 0.01% | 9 |
| netqasm | 4 | 26.080 s | 26.0800 s | 0.31 ms | 0.00% | 14 |

## Fitted cost model  t_netqmpi = alpha + beta * G

| backend | alpha (ms) | beta (us/op) | R^2 | configs |
|---|---|---|---|---|
| aer | 0.414 | 16.16 | 0.965 | 60 |
| cunqa | -1.589 | 135.45 | 0.767 | 24 |
| qoala | 0.025 | 15.49 | 1.000 † | 3 |
| netqasm | 0.184 | 9.48 | 0.897 † | 4 |
| pooled | 1.373 | 18.18 | 0.250 | 87 |

† fitted on fewer than 5 configurations against 2 free parameters: the coefficients are indicative, the R^2 is not evidence.

Splitting local gates from communication primitives, `t = alpha + beta*G_local + gamma*G_comm`:

| backend | alpha (ms) | beta (us/local op) | gamma (us/comm op) | gamma/beta | R^2 |
|---|---|---|---|---|---|
| aer | 0.409 | 15.07 | 17.49 | 1.2x | 0.965 |
| cunqa | -1.207 | 400.49 | -119.38 | -0.3x | 0.849 |
| netqasm | 0.202 | 3.43 | 13.59 | 4.0x | 0.902 |
| pooled | 1.259 | -4.20 | 45.64 | -10.9x | 0.259 |

Pooled crossover: NetQMPI falls below 1% of the run at about 153 recorded operations.

