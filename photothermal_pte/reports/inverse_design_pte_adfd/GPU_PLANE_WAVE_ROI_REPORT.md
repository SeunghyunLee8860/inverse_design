# GPU-only plane-wave protection of the central 2 µm ROI

Status: `BLOCKED_GPU_ONLY_SIX_PML_IDEAL_PLANE_WAVE`

No device, thermal solve, adjoint, finite difference, or optimization was
executed after this source gate failed.

## Question being tested

The protected optical/design ROI is exactly `x,y=[-1,1] µm`. The requested
illumination is a normal-incidence, x-polarized ideal plane wave with
`3–6 µm` source support and analysis at `4 µm`. The finite structure must have
PML on all six outer faces, no transverse periodic/Bloch boundary, and the
FDTD time stepping must run on a GPU.

These constraints were tested directly in Ansys Lumerical 2026 R1.2
(`v261`, solver 8.35.4522). The CPU FDTD resource was inactive for every
completed solve. Lumerical still uses CPU code for scripting, meshing, and
post-processing; that is an unavoidable documented part of a GPU run and is
not a CPU FDTD solve.

## Documentation and installed-engine audit

The Ansys documentation distinguishes three relevant source models:

1. `Bloch/periodic` plane wave is the ideal infinite plane wave and must use
   periodic or Bloch boundaries transverse to propagation.
2. `Diffracting` plane wave is a finite rectangular aperture and is allowed
   with PML in all directions, but it deliberately diffracts.
3. TFSF is the supported plane-wave illumination for a finite non-periodic
   scatterer with PML, but the FDTD GPU engine does not support TFSF.

Primary references:

- https://optics.ansys.com/hc/en-us/articles/360034382854-Plane-wave-and-beam-source-Simulation-object
- https://optics.ansys.com/hc/en-us/articles/360034382874-Understanding-field-truncation-issues-with-finite-sized-plane-wave-sources
- https://optics.ansys.com/hc/en-us/articles/360034382914-Understanding-the-diffracting-option-of-the-plane-wave-source
- https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU

The installed v261 engine was also probed rather than trusting documentation
alone. A GPU TFSF job terminated fail-closed with:

`Error: GPU simulation does not support the use of TFSF sources.`

BFAST is likewise listed as unsupported on the GPU and is intended for
periodic fixed-angle problems, not this isolated structure.

## Corrections made during the audit

- `plane wave type` is now set explicitly and read back before and after save;
  the GUI/API default is never trusted.
- The incident monitor was moved to the actual device-top plane rather than
  being placed inside the future 600 nm design volume.
- The first controls had a 100 nm mesh override ending exactly at the protected
  ROI boundary. That artificial mesh transition contaminated the control.
  The corrected controls use a lateral sampling mesh spanning the full domain.
- Source profile readback confirms that the ordinary source injects only Ex
  with unit amplitude at its source plane.
- A 1 ps to 5 ps check showed unchanged frequency-domain ROI metrics, so early
  termination is not the reported error.
- Standard and x/y steep-angle PML were compared.
- Source-to-monitor gap, lateral domain, finite aperture, z-domain depth, and
  PML layers were varied independently.

## Representative GPU results

All values below are empty-air controls evaluated only in the exact 2 µm ROI.
No normalization, fitted gain, clipping, smoothing, or spatial rescaling was
used.

| Source | Domain / aperture | PML | gap | mean E2 error | intensity RMS | peak-to-peak | max phase | Ey/Ex | Ez/Ex |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bloch/periodic source crossing x/y PML | 6 / 6 µm | 24 | 0.15 µm | 17.248% | 0.491% | 2.075% | 8.983° | 1.598% | 16.311% |
| Official Diffracting | 6 / 5 µm | 24 | 0.15 µm | 41.090% | 8.746% | 32.931% | 6.745° | 1.782% | 16.165% |
| Official Diffracting | 24 / 20 µm | 24 | 0.15 µm | 13.000% | 6.077% | 18.615% | 0.729° | 0.467% | 6.608% |
| Official Diffracting, 20 nm gap | 24 / 20 µm | 24 | 0.02 µm | 11.488% | 4.995% | 15.242% | 2.011° | 0.470% | 6.465% |
| Same, deep z and stronger PML | 24 / 20 µm | 32 | 0.02 µm | 11.450% | 4.974% | 15.184% | 2.027° | 0.470% | 6.466% |

The 6 µm corrected ordinary-source RMS happens to be below 0.5%, but it fails
the peak-to-peak, phase, component-purity, and mean-field gates. Results are
non-monotonic with lateral domain because a Bloch/periodic plane-wave source
crossing a transverse PML is not a valid matched source/boundary pair. It is
not promoted.

An otherwise identical 1 ps/5 ps pair (using the earlier ROI-edge mesh) gave
unchanged frequency-domain metrics to the displayed precision; simulation
duration was therefore not the source of the reported distortion.

Increasing the finite Diffracting aperture to 20 µm reduces phase and
longitudinal-field errors but does not protect the 2 µm ROI to the required
tolerance. Increasing the z clearance, PML from 24 to 32 layers, and simulation
time did not remove the remaining error. This is the finite-aperture field,
not a missing API setting.

An imported raised-cosine flat-top E/H source was also implemented as a
diagnostic. It is a finite beam rather than an ideal plane wave and did not
pass the ROI field-purity gates, so it is not a hidden production fallback.

## Gate and decision

Required simultaneously:

- mean intensity error from the unit source `<0.5%`;
- intensity RMS `<0.5%`;
- intensity peak-to-peak `<1%`;
- maximum phase deviation `<1 degree`;
- `Ey/Ex` and `Ez/Ex` L2 ratios `<0.1%`.

No GPU-only six-PML case passed. Therefore AD–FD remains fail-closed.

Lumerical can solve the physical problem if one constraint changes:

- allow CPU FDTD: use TFSF with six PML;
- allow transverse periodic/Bloch boundaries: use the ideal plane-wave source,
  while explicitly accepting a periodic supercell;
- retain GPU and six PML: use a physically finite beam and certify that beam,
  rather than calling it an ideal plane wave.

The current contract permits none of these substitutions without user
approval.

## Raw artifacts

Raw FSP/HDF5/JSON files remain outside Git. Key cases are:

- `/home/seunghyun/tairte4_artifacts/gpu_bloch_roi_d6_final_gate_20260726_1/`
- `/home/seunghyun/tairte4_artifacts/gpu_bloch_roi_d16_ap17_gap015_pml24_20260726_1/`
- `/home/seunghyun/tairte4_artifacts/gpu_diff_roi_d6_ap5_fullmesh_gap015_pml24_20260726_1/`
- `/home/seunghyun/tairte4_artifacts/gpu_diff_roi_d24_ap20_fullmesh_gap015_pml24_20260726_1/`
- `/home/seunghyun/tairte4_artifacts/gpu_diff_roi_d24_ap20_zdeep_pml32_20260726_1/`
- `/home/seunghyun/tairte4_artifacts/gpu_tfsf_compatibility_v261_final_20260726_1/`
