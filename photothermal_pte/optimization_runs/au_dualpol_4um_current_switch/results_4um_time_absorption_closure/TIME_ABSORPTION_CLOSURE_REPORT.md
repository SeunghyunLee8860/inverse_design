# Time-window and absorption-closure diagnostic

Status: `BLOCKED_TIME_OR_ABSORPTION_CLOSURE`

Shared-linear Au is evaluated on the historical factor-8 partial-z
eta=0.35 Eb case that had the worst Q/closed-flux closure. No incident
rescaling or downstream current is used.

| total periods | window periods | Q window power | Q spatial | target/discrete Q | target Q/TD flux | discrete Q/TD flux | TD/phasor flux | all gates |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 24 | 4 | 0.1436% | 1.6273% | 1.1172% | 1.9094% | 0.7835% | 0.3039% | False |
| 32 | 4 | 21.6934% | 45.6132% | 1.1394% | 171.3449% | 170.5412% | 327.8769% | False |
| 40 | 4 | 854.0150% | 97.1305% | 1.1136% | 100.7121% | 100.7042% | 9806.8252% | False |
| 40 | 8 | 128.5751% | 98.8911% | 1.1372% | 100.1969% | 100.1947% | 13132.3495% | False |

This is not a mesh certificate: the device contract is unconfirmed
and only the historical partial material-z factor-8 grid is tested.
