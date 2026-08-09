# Run 008: corrected E||a objective-first constrained optimization

- Fresh restart from the corrected-axis uniform initial density.
- Solver axes: `x=b`, `y=a`, `z=c`; illumination: `E||a` (`90 deg`).
- Fixed-sign objective maximizes the magnitude of the initial negative PTE current.
- Beta 2 starts smooth-constraint feasible with 20% cap slack.
- Beta 2 and 4 prioritize FOM/topology exploration; exact 500 nm counts are diagnostic.
- Phase-wise morphology nonincrease begins at beta 8; exact-count nonincrease begins at beta 32.
- A beta stage advances only after the registered four-update FOM/density plateau gates pass.
- Stage budgets are fail-closed watchdogs, not authority to force an unconverged beta transition.
- Final promotion still requires exact binary density, zero exact 500 nm solid/void violations, and a fresh GPU Maxwell/CUDA thermal evaluation.
