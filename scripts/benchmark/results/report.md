## Portability matrix

| app | aer | cunqa | qoala | netqasm |
|---|---|---|---|---|
| cascade | OK | OK | OK | OK |
| ghz | OK | OK | n/i | OK |
| qft | OK | OK | n/i | n/i |
| qft_telegate | OK | OK | n/i | n/i |

`OK` = echo exact on every configuration tried. `n/i` = the adapter raises `NotImplementedError`. `WRONG` = completed but returned the wrong answer on a backend that has no noise model. A bare `F=` range is decoherence on a backend that models hardware.

## Overhead by backend

| backend | configs | cold total | warm total | warm NetQMPI | warm share | median ops |
|---|---|---|---|---|---|---|
| aer | 60 | 0.311 s | 0.0132 s | 1.19 ms | 12.69% | 57 |
| cunqa | 24 | 4.795 s | 4.6838 s | 3.42 ms | 0.07% | 32 |
| qoala | 3 | 3.137 s | 2.3921 s | 0.16 ms | 0.01% | 9 |
| netqasm | 4 | 29.416 s | 29.4159 s | 0.32 ms | 0.00% | 14 |

## Fitted cost model  t_netqmpi = alpha + beta * G

| backend | alpha (ms) | beta (us/op) | R^2 | configs |
|---|---|---|---|---|
| aer | 0.423 | 13.40 | 0.975 | 60 |
| cunqa | -1.589 | 135.45 | 0.767 | 24 |
| qoala | 0.018 | 16.12 | 1.000 † | 3 |
| netqasm | 0.154 | 10.66 | 0.879 † | 4 |
| pooled | 1.418 | 15.44 | 0.188 | 87 |

† fitted on fewer than 5 configurations against 2 free parameters: the coefficients are indicative, the R^2 is not evidence.

Splitting local gates from communication primitives, `t = alpha + beta*G_local + gamma*G_comm`:

| backend | alpha (ms) | beta (us/local op) | gamma (us/comm op) | gamma/beta | R^2 |
|---|---|---|---|---|---|
| aer | 0.389 | 5.73 | 22.86 | 4.0x | 0.983 |
| cunqa | -1.207 | 400.49 | -119.38 | -0.3x | 0.849 |
| netqasm | 0.181 | 1.28 | 17.04 | 13.3x | 0.887 |
| pooled | 1.269 | -13.76 | 51.28 | -3.7x | 0.205 |
