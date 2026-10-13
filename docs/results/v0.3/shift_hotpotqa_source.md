# CRC shift test (offline, no Jev calls): hotpotqa -> musique, twowiki
# Single-source shift across closed-pool multi-hop datasets (shift in difficulty more than in task type).
# Source calibration: outputs/hotpotqa_halted_n98.jsonl n_cal_gated=150 alpha=0.3 spec_sha=b28902e593b0...
#   rule sufficiency: threshold=0.960001 threshold_score=0.960000 k_hat=28 n_cal=150
#   rule answer_abstain: threshold=0.980001 threshold_score=0.980000 k_hat=12 n_cal=150
# Note: the sufficiency row is a signal-shift diagnostic only (live stopping would change the rows); only answer_abstain maps to an operating characteristic.
| dataset | n_report | rule | coverage | sel_risk P(wrong\|ans) | marg_risk P(ans&wrong) | alpha | notes |
|---|---|---|---|---|---|---|---|
| musique | 111 | shift:sufficiency (hotpotqa-calibrated) | 0.036 | 0.250 | 0.009 | 0.300 | — |
| musique | 111 | shift:answer_abstain (hotpotqa-calibrated) | 0.027 | 0.000 | 0.000 | 0.300 | — |
| musique | 111 | gated_crc-infile (own calibration) | 0.000 | n/a | n/a | 0.300 | 0/111 served; threshold rebuilt from calibrate rows (run halted before trace) |
| twowiki | 150 | shift:sufficiency (hotpotqa-calibrated) | 0.253 | 0.079 | 0.020 | 0.300 | — |
| twowiki | 150 | shift:answer_abstain (hotpotqa-calibrated) | 0.080 | 0.000 | 0.000 | 0.300 | — |
| twowiki | 150 | gated_crc-infile (own calibration) | 0.940 | 0.326 | 0.307 | 0.300 | 141/150 served; live decision at recorded threshold |
