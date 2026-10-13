# CRC shift test (offline, no Jev calls): twowiki -> hotpotqa, musique
# Single-source shift across closed-pool multi-hop datasets (shift in difficulty more than in task type).
# Source calibration: outputs/twowiki_n150.jsonl n_cal_gated=150 alpha=0.3 spec_sha=697b55bc2332...
#   rule sufficiency: threshold=0.520001 threshold_score=0.520000 k_hat=95 n_cal=150
#   rule answer_abstain: threshold=0.090000 threshold_score=0.090000 k_hat=137 n_cal=150
# Note: the sufficiency row is a signal-shift diagnostic only (live stopping would change the rows); only answer_abstain maps to an operating characteristic.
| dataset | n_report | rule | coverage | sel_risk P(wrong\|ans) | marg_risk P(ans&wrong) | alpha | notes |
|---|---|---|---|---|---|---|---|
| hotpotqa | 98 | shift:sufficiency (twowiki-calibrated) | 0.714 | 0.543 | 0.388 | 0.300 | — |
| hotpotqa | 98 | shift:answer_abstain (twowiki-calibrated) | 0.929 | 0.593 | 0.551 | 0.300 | — |
| hotpotqa | 98 | gated_crc-infile (own calibration) | 0.092 | 0.556 | 0.051 | 0.300 | 9/98 served; threshold rebuilt from calibrate rows (run halted before trace) |
| musique | 111 | shift:sufficiency (twowiki-calibrated) | 0.459 | 0.373 | 0.171 | 0.300 | — |
| musique | 111 | shift:answer_abstain (twowiki-calibrated) | 0.937 | 0.625 | 0.586 | 0.300 | — |
| musique | 111 | gated_crc-infile (own calibration) | 0.000 | n/a | n/a | 0.300 | 0/111 served; threshold rebuilt from calibrate rows (run halted before trace) |
