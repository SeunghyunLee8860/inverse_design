# Run 003 exact-disk constraint recovery

Status: `VALIDATED_OFFLINE_EXACT_DISK_CONSTRAINT_RECOVERY`; passed: `True`.

Preserved checkpoint: `f36b5d1121b090e18fb44d48efa87e60863f02278e84d81bc5c609c6e11998bf`.

Baseline exact bad cells were `385` solid and `521` void. The largest offline diagnostic step reduced them to `371` / `415`. This diagnostic density was not accepted, was not sent to Maxwell or thermal solvers, and did not replace the checkpoint.

Maximum centered-FD relative error: `9.001601e-08`.
