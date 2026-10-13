| dataset | arm | split | n | served% | EM | F1 | coverage | sel_risk P(wrong\|ans) | marg_risk P(ans&wrong) | AURC | ECE | rounds | $ | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | fixed | calibrate | 150 | 100.0 | 77/150 | 0.670 | 1.000 | 0.487 | 0.487 | n/a | n/a | 450 | 0.1122 | — |
| hotpotqa | fixed | report | 98 | 100.0 | 40/98 | 0.575 | 1.000 | 0.592 | 0.592 | n/a | n/a | 294 | 0.0734 | partial run: report n=98 < calibrate n=150 |
| hotpotqa | gated | calibrate | 150 | 81.3 | 74/150 | 0.661 | 0.813 | 0.426 | 0.347 | 0.340 | 0.308 | 239 | 0.0696 | — |
| hotpotqa | gated | report | 98 | 68.4 | 39/98 | 0.561 | 0.684 | 0.478 | 0.327 | 0.491 | 0.334 | 177 | 0.0519 | partial run: report n=98 < calibrate n=150 |
| hotpotqa | gated_crc | report | 98 | 9.2 | 39/98 | 0.572 | 0.092 | 0.556 | 0.051 | 0.505 | 0.327 | 267 | 0.0772 | threshold rebuilt from calibrate rows (run halted before trace) |
| musique | fixed | calibrate | 150 | 100.0 | 39/150 | 0.355 | 1.000 | 0.740 | 0.740 | n/a | n/a | 450 | 0.0649 | — |
| musique | fixed | report | 111 | 100.0 | 38/111 | 0.407 | 1.000 | 0.658 | 0.658 | n/a | n/a | 333 | 0.0474 | partial run: report n=111 < calibrate n=150 |
| musique | gated | calibrate | 150 | 58.7 | 39/150 | 0.345 | 0.587 | 0.648 | 0.380 | 0.580 | 0.350 | 371 | 0.0679 | — |
| musique | gated | report | 111 | 63.1 | 40/111 | 0.424 | 0.631 | 0.471 | 0.297 | 0.382 | 0.266 | 259 | 0.0465 | partial run: report n=111 < calibrate n=150 |
| musique | gated_crc | report | 111 | 0.0 | 42/111 | 0.447 | 0.000 | n/a | n/a | 0.409 | 0.280 | 325 | 0.0580 | threshold rebuilt from calibrate rows (run halted before trace) |
| twowiki | fixed | calibrate | 150 | 100.0 | 99/150 | 0.679 | 1.000 | 0.340 | 0.340 | n/a | n/a | 450 | 0.0346 | — |
| twowiki | fixed | report | 150 | 100.0 | 99/150 | 0.690 | 1.000 | 0.340 | 0.340 | n/a | n/a | 450 | 0.0343 | — |
| twowiki | gated | calibrate | 150 | 61.3 | 99/150 | 0.683 | 0.613 | 0.207 | 0.127 | 0.139 | 0.168 | 323 | 0.0395 | — |
| twowiki | gated | report | 150 | 67.3 | 101/150 | 0.711 | 0.673 | 0.297 | 0.200 | 0.189 | 0.263 | 316 | 0.0382 | — |
| twowiki | gated_crc | report | 150 | 94.0 | 99/150 | 0.696 | 0.940 | 0.326 | 0.307 | 0.216 | 0.257 | 311 | 0.0377 | live decision at recorded threshold |
