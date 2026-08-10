# Run 014 — adaptive E||a current maximization

This run restarts from uniform `rho=0.5` using the validated contact-anchored
TaIrTe4 optical/thermal/electrical contract.  It replaces the fixed 36-evaluation
Run012 continuation with measured objective-plateau advancement.

- beta: `1,2,3,4,6,8,12,16,24,32,48,64`
- no fixed total-iteration completion rule
- low-beta morphology weight: zero, then gradual ramp
- exact thresholded feature counts: diagnostic during continuation
- requested feature audit: five 100 nm pixel supports = 500 nm
- initial density: uniform 0.5; no imposed symmetry or S-shaped seed
- polarization: `E||a` (`Lumerical y=a`, polarization angle 90 deg)

Every solver evaluation writes a numbered plot.  Completion means the measured
plateau or projected move-floor stationarity gate passed at every beta; it does
not mean a global optimum.
