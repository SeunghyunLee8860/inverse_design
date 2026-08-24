# Portable Lumerical `runres`

This bundle holds a FlexNet `PROJECT` reservation for the complete lifetime of
a Lumerical job. It prevents the nine `lum_fdtd_solve` licences required by a
GPU solve from being taken by another job between optimization evaluations.

## Files

- `runres`: owns the reservation, starts the child, forwards termination
  signals, and releases the reservation after the child exits.
- `lum_reserve.py`: edits marker-scoped reservation blocks, calls `lmreread`,
  exports `LM_PROJECT`, verifies the reservation, and removes stale blocks.
- `runres.env.example`: target-server configuration template.

The sources were vendored from the production installation on `aigpu1123` on
2026-08-24. Original SHA-256 values before the portable path adjustment:

```text
runres         c38cdd72fa663d22b669b62bdeb6a84bdc00455a7126234e2a94720f4d05bbeb
lum_reserve.py b0c2df30f27951fc244a52e7324c0f705163c9ba34a8b4321eb896a56509a947
```

## Site Requirements

1. The target must use a FlexNet/Ansys license daemon that supports `RESERVE`
   by `PROJECT` and propagates `LM_PROJECT` to Lumerical.
2. `MSOPT_RESERVE_OPT_FILE` must name the options file read by the actual
   license daemon. A private copy on a compute node has no effect.
3. All participating users must share the same lock file and have controlled
   write access to the options file. Prefer a dedicated group. Do not embed a
   sudo password or grant broad passwordless sudo.
4. `MSOPT_RESERVE_SERVER` must use the real license host, not `localhost`,
   unless the daemon actually runs on the compute node.
5. Reservations consume shared global capacity. Coordinate the reservation
   count with the license administrator.

## Installation

```bash
install -d "$HOME/.local/lib/lumerical-runres" "$HOME/.local/bin"
install -m 644 lum_reserve.py "$HOME/.local/lib/lumerical-runres/"
install -m 755 runres "$HOME/.local/bin/"
export LUM_RESERVE_MODULE_DIR="$HOME/.local/lib/lumerical-runres"
```

Copy `runres.env.example`, replace every site-specific path, and source it.

## Fail-Closed Validation

First perform read-only discovery. Do not edit the options file yet.

```bash
"$MSOPT_RESERVE_LMUTIL" lmstat -c "$MSOPT_RESERVE_SERVER" -a
"$MSOPT_RESERVE_LMUTIL" lmstat -c "$MSOPT_RESERVE_SERVER" \
  -i "$MSOPT_RESERVE_FEATURE"
python "$LUM_RESERVE_MODULE_DIR/lum_reserve.py" status
```

Confirm that `status` identifies the correct feature, inventory, options file,
and write method. Then use a short smoke process while watching `lmstat` from a
second shell:

```bash
MSOPT_RUN_CMD=/usr/bin/env runres \
  --reserve-wait 60 \
  --reserve-count 9 \
  --reserve-tag smoke \
  python3 -c 'import os,time; print(os.environ["LM_PROJECT"]); time.sleep(60)'
```

The reservation must appear during the sleep and disappear after normal exit.
Verify SIGTERM cleanup as a separate test before launching production.

## Production Without a Site `run` Wrapper

With `MSOPT_RUN_CMD=/usr/bin/env`, arguments after the reservation flags form
the child command directly:

```bash
runres \
  --reserve-wait 86400 \
  --reserve-count 9 \
  --reserve-tag inverse_design_gpu3 \
  CUDA_VISIBLE_DEVICES=3 python3 -u optimize.py
```

The child and every Lumerical process it creates inherit `LM_PROJECT`.

## Recovery

Normal exit, SIGINT, SIGTERM, and SIGHUP release the block. After a launcher
SIGKILL or host crash, run the sweep on the same host that owned the marker:

```bash
python "$LUM_RESERVE_MODULE_DIR/lum_reserve.py" sweep
```

`sweep` removes only blocks marked `msopt-reserve` whose recorded local PID no
longer exists. It deliberately does not remove another host's block.
