## Portability matrix

| app | aer | cunqa | qoala | netqasm |
|---|---|---|---|---|
| cascade | OK | OK | OK | error |
| ghz | OK | OK | n/i | – |
| qft | OK | OK | n/i | – |
| qft_telegate | n/i | OK | n/i | – |

`OK` = echo exact on every configuration tried. `n/i` = the adapter raises `NotImplementedError`. `WRONG` = completed but returned the wrong answer on a backend that has no noise model. A bare `F=` range is decoherence on a backend that models hardware.

## Overhead by backend

| backend | configs | cold total | warm total | warm NetQMPI | warm share | median ops |
|---|---|---|---|---|---|---|
| aer | 45 | 0.317 s | 0.0148 s | 1.23 ms | 9.41% | 47 |
| cunqa | 24 | 4.795 s | 4.6838 s | 3.42 ms | 0.07% | 32 |
| qoala | 3 | 3.137 s | 2.3921 s | 0.16 ms | 0.01% | 9 |

## Fitted cost model  t_netqmpi = alpha + beta * G

| backend | alpha (ms) | beta (us/op) | R^2 | configs |
|---|---|---|---|---|
| aer | 0.370 | 15.33 | 0.983 | 45 |
| cunqa | -1.589 | 135.45 | 0.767 | 24 |
| qoala | 0.018 | 16.12 | 1.000 † | 3 |
| pooled | 1.449 | 19.63 | 0.217 | 72 |

† fitted on fewer than 5 configurations against 2 free parameters: the coefficients are indicative, the R^2 is not evidence.

Splitting local gates from communication primitives, `t = alpha + beta*G_local + gamma*G_comm`:

| backend | alpha (ms) | beta (us/local op) | gamma (us/comm op) | gamma/beta | R^2 |
|---|---|---|---|---|---|
| aer | 0.366 | 11.62 | 19.54 | 1.7x | 0.984 |
| cunqa | -1.207 | 400.49 | -119.38 | -0.3x | 0.849 |
| pooled | 1.381 | -6.24 | 48.83 | -7.8x | 0.224 |
