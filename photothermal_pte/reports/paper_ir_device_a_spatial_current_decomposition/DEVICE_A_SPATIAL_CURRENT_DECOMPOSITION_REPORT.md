# Device-A spatial current decomposition

Status: `COMPLETED_DEVICE_A_SPATIAL_CURRENT_DECOMPOSITION`

This is a read-only/offline decomposition of the immutable registered
Maxwell -> explicit-3D thermal -> PTE fields. No new Maxwell, thermal, or
weighting solve was run.

## Literal current equation

`x=b`, `y=a`, and

```text
I = integral[-sigma_b S_b (d_b T)(d_b psi)
             -sigma_a S_a (d_a T)(d_a psi)] dV.
```

The coefficients are `-2.970000`
and `2.946000 A/(m K)`. They are
nearly equal in magnitude and opposite in sign; neither term was omitted.

## Same-position polarization difference

Positive values below mean that the saved `E||a` field produces more current
than the saved `E||b` field at the same registered beam position.

| d (um) | total a-b (nA) | x=b term (nA) | y=a term (nA) | positive-cell difference (nA) | negative-cell difference (nA) | largest spatial region |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 2.566641 | 1.260697 | 1.305944 | 1.639590 | 0.927051 | free_edge_within_1um |
| 3 | 2.236814 | 1.039417 | 1.197397 | 1.632258 | 0.604556 | free_edge_within_1um |
| 5 | 1.908797 | 0.735975 | 1.172822 | 1.293913 | 0.614884 | free_edge_within_1um |


The negative-cell column is the signed change in cancellation. A positive
value means the `a` case is less negative (less cancelled) than `b`.

Both crystallographic derivative terms contribute. At `d=1 um` the excess
is `1.260697 nA` from the `x=b` term and `1.305944 nA` from the `y=a` term;
at `d=5 um` they are `0.735975 nA` and `1.172822 nA`. The result is therefore
not attributable to one omitted derivative or one swapped Seebeck term.

## Where the excess current occurs

The free-edge band contributes `+4.216371`, `+3.519862`, and `+2.561187 nA`
to `a-b` at `d=1,3,5 um`. The flake interior contributes `-1.528097`,
`-1.468526`, and `-1.190489 nA`, respectively. Thus the interior actually
favors `b`; the simulated `a>b` trend is created by the free-edge response.

The beam-centred partition independently localizes the `d=1 um` excess:
`r<0.5 w0` contributes `+4.200080 nA`, while all larger-radius bins together
contribute `-1.633439 nA`. The weighting-potential partition places the
largest positive difference in `psi=0.2--0.4` for all three positions. These
are independent partitions of the same closed volume integral, not fitted
current corrections.

## Spatial contract

The flake cells are partitioned exactly once into top-contact within 2 um,
bottom-contact within 2 um, remaining free-edge within 1 um, and flake
interior. Independent radial and weighting-potential-bin decompositions are
also stored in JSON/CSV. Region sums, `x+y`, and the published current all
close below `1e-12`; maximum observed error is `3.019e-16`.

This checkpoint diagnoses where the existing result is generated. It does
not establish that the Maxwell Q, approximate Figure-3H registration, exact
contact CAD, or paper beam radius is correct. Raw NPZ fields remain outside
Git and are path/size/SHA-256 pinned in the manifest.
