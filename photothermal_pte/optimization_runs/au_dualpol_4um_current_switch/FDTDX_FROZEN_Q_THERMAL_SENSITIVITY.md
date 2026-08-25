# FDTDX frozen-Q thermal boundary and interface sensitivity

## Decision

The selected frozen-Q prototype mesh/domain has a complete integrity-certified
thermal scenario sensitivity characterization for Ea and Eb.  The diagnostic
configuration remains x/y factor 2, z factor 2, and a `48/30/3 um` lateral
half-span / Si-depth / top-air domain (`548 x 548 x 72`).

This is not physical-parameter convergence.  No device-specific confidence
bounds or measurements were supplied.  The certificate therefore keeps all
boundary-condition, interface-parameter, material-parameter, production, and
optimizer gates false even though its artifact-integrity status is ready.

The dominant blocker is the TaIrTe4-SiO2 thermal contact scenario.  Replacing
the `7.37e6 W/(m2 K)` checkpoint with the repository's named `7.37e4 W/(m2 K)`
low scenario changes the Ta temperature-map NRMSE by about 95%, mean
temperature by about 97%, and combined in-plane gradient by 854--877%.
Mesh/domain convergence cannot compensate for an unverified interface model.

## Parameter status

The values are deliberately labeled scenarios, not confidence intervals:

- `G(TaIrTe4/SiO2)=7.37e6 W/(m2 K)` is the existing numerical checkpoint.
  The repository's physical-model audit states that neither it nor the
  `7.37e4` deposited-SiO2 estimate is established as uniquely correct by a
  traceable device-specific source.
- The baseline `G(Au/TaIrTe4)=17.24 MW/(m2 K)` is derived from a calculated
  Au/MoS2 resistance of `5.8e-8 m2 K/W`; it is explicitly not Au/TaIrTe4
  measurement data.  The 1 and 100 MW/(m2 K) and perfect-contact cases are
  numerical brackets.
- TaIrTe4 `kz=0.5, 1, 2 W/(m K)` are numerical scenarios because the current
  contract does not establish a sourced device-specific range.
- far-x/y ambient Dirichlet versus adiabatic and top `h=0,10,20 W/(m2 K)` are
  boundary-model brackets, not experimental boundary validation.

The local TaIrTe4 papers establish the importance of actual flake edges,
electrodes, crystal orientation, and support geometry for PTE response; they
do not turn the above Au/TaIrTe4 analogue or numerical checkpoint into
device-specific measurements.

## Provenance and baseline rebound

Every case revalidates:

- prior domain certificate:
  `/home/seunghyun200/fdtdx_results/frozen_q_thermal_domain_certificate_a7d5a52a/FDTDX_FROZEN_Q_THERMAL_DOMAIN_CERTIFICATE.json`
- prior SHA-256:
  `2402be4a0b669c24acfaf5167cb9a5917edef65a2c2ab4a342fcc185c6bd4ef1`
- blocked optical z32 certificate SHA-256:
  `079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`
- exact-binary 375-cell Au mask and common 285-uW normalization;
- conservative Q mapping, selected mesh/domain, residual, energy balance,
  clean Git state, and exclusive physical GPU ownership.

The newly computed baseline temperature, both gradients, coordinates, and
source map are byte-exact equal to the prior selected `combined_mid` domain
artifacts for both polarizations.  An earlier unpromoted run at commit
`a4d56e02` exposed only a floating-point assembly-order difference (maximum
temperature difference below `4.9e-15 K`).  The certificate thresholds were
not relaxed: commit `94e854ac` restored the original x/y/z boundary insertion
order and all cases were rerun in a new output root.

## Sensitivity results

All values are relative to the baseline unless explicitly shown.  No physical
acceptance threshold is applied.

| scenario | pol. | T-map NRMSE | Tmax relative | Tmean relative | combined-gradient NRMSE |
|---|:---:|---:|---:|---:|---:|
| far x/y adiabatic | Ea | 0.06276% | 0.01367% | 0.12107% | 0.00097% |
| far x/y adiabatic | Eb | 0.06366% | 0.01409% | 0.12229% | 0.00100% |
| top h=0 | Ea | 0.00017% | 0.00003% | 0.00015% | 0.00049% |
| top h=0 | Eb | 0.00018% | 0.00005% | 0.00016% | 0.00053% |
| top h=20 | Ea | 0.00017% | 0.00003% | 0.00015% | 0.00049% |
| top h=20 | Eb | 0.00018% | 0.00005% | 0.00016% | 0.00053% |
| TaIrTe4-SiO2 G=7.37e4 | Ea | 95.35680% | 90.67432% | 96.57930% | 854.02563% |
| TaIrTe4-SiO2 G=7.37e4 | Eb | 95.40891% | 90.81036% | 96.61182% | 877.37634% |
| Au-TaIrTe4 G=1 MW | Ea | 2.28248% | 0.54949% | 0.03184% | 10.82025% |
| Au-TaIrTe4 G=1 MW | Eb | 1.40055% | 0.52413% | 0.04071% | 8.35496% |
| Au-TaIrTe4 G=100 MW | Ea | 0.99881% | 0.28981% | 0.00095% | 5.81547% |
| Au-TaIrTe4 G=100 MW | Eb | 0.82531% | 0.34728% | 0.00153% | 5.67703% |
| Au-TaIrTe4 perfect | Ea | 1.40021% | 0.42257% | 0.00087% | 8.30183% |
| Au-TaIrTe4 perfect | Eb | 1.19717% | 0.51423% | 0.00160% | 8.31274% |
| TaIrTe4 kz=0.5 | Ea | 8.01798% | 7.61938% | 7.99899% | 7.91436% |
| TaIrTe4 kz=0.5 | Eb | 7.21238% | 6.55928% | 7.21187% | 6.86941% |
| TaIrTe4 kz=2 | Ea | 4.60618% | 4.21637% | 4.34870% | 4.05922% |
| TaIrTe4 kz=2 | Eb | 4.08431% | 3.59134% | 3.88755% | 3.51227% |

Every source map is identical to baseline to roundoff (reported NRMSE zero),
so these changes arise from the named thermal scenario rather than Q changes.

The absolute Ta results show the scale of the dominant uncertainty:

| scenario | Ea Tmax / mean | Eb Tmax / mean |
|---|---:|---:|
| baseline | 0.989086 / 0.114817 K | 1.647135 / 0.195033 K |
| TaIrTe4-SiO2 G=7.37e4 | 10.606051 / 3.356544 K | 17.923823 / 5.756274 K |
| TaIrTe4 kz=0.5 | 1.070664 / 0.124800 K | 1.762759 / 0.210192 K |
| TaIrTe4 kz=2 | 0.947383 / 0.109824 K | 1.587981 / 0.187451 K |

These remain properties of an optically unconverged frozen Q and an assumed
rectangular device.  They are not detector predictions.

## Boundary-flow interpretation

In the baseline Ea case, the four artificial lateral Dirichlet entries sum to
about `21.46 uW`, the bottom entry is about `46.31 uW`, and top convection is
only `0.000361 uW`.  These are numerical truncation-boundary fluxes, not
intrinsic physical heat-path fractions.  Making the far x/y boundaries
adiabatic reroutes nearly all mapped power to the bottom while changing the Ta
temperature map by only 0.063%.  This supports numerical robustness to the far
x/y treatment at the selected domain, but it does not validate the real chip,
mount, or flake-support boundary model.

## Runtime

Ea and Eb ran concurrently on physical GPUs 6 and 7.  Other users' Lumerical
jobs on GPUs 0 and 4 were not touched.  Each case took 21--25 seconds; the 20
solves completed in about 3 minutes 55 seconds wall time.  All explicit
relative residuals are below `1e-9` and all energy-balance relative errors are
below `1.6e-10`.

## External artifacts

Case root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_sensitivity_94e854ac/`

- clean runner commit:
  `94e854ac90bd3a4b0889c409c6f677158b609393`
- runner SHA-256:
  `108212a875455bb996cef0d0c04ffa86f07e2513d6730566a2937156769538ee`

| scenario | pol. | report SHA-256 | raw NPZ SHA-256 |
|---|:---:|---|---|
| baseline | Ea | `ee81a14b3015379d5e6044acc488ba355d1cfefab1ec1b5fdcb8b9d0fbcc0777` | `4ae6078ed264764171e78ae5c1d462366f974b2a7468d863c0829a9dced0eb25` |
| baseline | Eb | `02b926cd148d01f619fb64d0c074d2342cbde27b9d155a20c3933609666a3d6c` | `9bc3dd16dee24bed32ff8e939a97c635827f5eedaa8c3a263db7fb7b36b6609b` |
| far x/y adiabatic | Ea | `439fe1b6a15a62597c4b61f66047984cf628d5c2288a60e4e353c49bbee727d8` | `c1d01476479a15566bf3172d03a5e9d90ece6b51cc937cc9e75cdcfc7eed6a11` |
| far x/y adiabatic | Eb | `da0b29ac90a3cbe04738ac5393acdff2aa1c3f76d2634699821504b40a674b65` | `67bcfe0413fd1ebe13a329282e978b10fd7ca435ad93ad936a7835049eaf4f2d` |
| top h=0 | Ea | `3a4a02885dff5b5b3bf437ecb948b29d3e58065a9a8b8afaa271099a21fc5712` | `bd6540f7a9a7ebd3beea479513b55c40f4b24be445da494fd858b0ec264aec59` |
| top h=0 | Eb | `074374ac80d34f5299bf84af7d18ef4f0f00e472ec60a1451f38a5c0a530d408` | `628bcefc5d4613400ca0c6ccfab592ee75c79579dc397b9402b2513d0f6ce6ad` |
| top h=20 | Ea | `61f0cf74b25be57cd6a996acc2b02ac16c1aad61dda6747500d85dedb820ea01` | `2edf23831abdbd4c9938b5affe3b9f7135a2527045fa9e528d9d20cd08f54ab0` |
| top h=20 | Eb | `d8aa316271eea873bb2771fead24f5327f81b5e42414beb910a6e6e69a4d88eb` | `96430932fcba687dee8ee3d2511371f9bc07a25c896b43da46f14a5a0c5dcca3` |
| TaIrTe4-SiO2 low | Ea | `8522b6439eb78741e439eed9e02755a92a9efea5fca15f2f206bb68ae2caac14` | `e65e699904053b8054e882a7d8f2990b4d4daa28eed63f52d72da979087e222d` |
| TaIrTe4-SiO2 low | Eb | `19b95ef13939e63f4343e668178de08ae75f3712791d1c0c204bcd0559b9456b` | `f934cff7a1216495541772d867e1e67223eb45d15ecb05f6759018fa8d91e671` |
| Au-TaIrTe4 1 MW | Ea | `df0123d8be8a35d0dc56d9315557876626d2c16d990ea9c69f3e48f4f7c6be11` | `6d300efc31c380584103b2051d8e4d70289eb3cb8d299084d349a389be18df59` |
| Au-TaIrTe4 1 MW | Eb | `9a8c07f5139ca2d2dcd9586c360e846deb4ce44268119f4fde125c53f36a668c` | `57b3c573a1d270770493ea2e62577b90e97c4a7f4d67b8b17183e1e974f82bc0` |
| Au-TaIrTe4 100 MW | Ea | `98ffce06e3808b949b1941bedae15d98d6a19966fefb49412158aea6f1559e54` | `99613320c9de4d894411391924b0427797658a2e8c7d593d90fddeded7684276` |
| Au-TaIrTe4 100 MW | Eb | `b148ae27f4f3d468c89d55a788b6d34d2efb6362b4b19fa87c6924bbbbda9fa1` | `4d44077609784f1ca9b47f066271bf87b6b54d418013af81d27fb642f96cb89a` |
| Au-TaIrTe4 perfect | Ea | `2afc23681ac4de369d1055a0f08b30f84a3916c7205e4e742ba8dc0200180339` | `1eb5eb50a15e82d993e36690cf8dc2e3c2436f00f1063baf448bbfb936c0d417` |
| Au-TaIrTe4 perfect | Eb | `6dcb024767a621e93a7b175e7dcc2f30b6e0894da2f4e446123301d566dd8321` | `358bd7bed16f3a4eafaecf12b1ffb44edf0de2a10e5a774f00a7ebc8c9420da0` |
| TaIrTe4 kz=0.5 | Ea | `a4cb03f8478c9f1eed9ade27a1ed84696b393df23b1b47d933dc0e66ac7cd957` | `753ebdff4a4ddc774fa1c06cb06fa7488fa7bc44dd6828d53ceb44df25e298c7` |
| TaIrTe4 kz=0.5 | Eb | `6bde01996b1ddb9de71e0178ef765ec6953042f402603594494b77b1eda5a738` | `3156256f186222cbef52cb36056e93ad6dcd20e403e06f32e118e1a9649a1125` |
| TaIrTe4 kz=2 | Ea | `27c6043d2cb39dc533cff4673d40398b2a8bb65f03f8ae8e25cf589936e47c2e` | `73e0e16d499c755725b0256c758a23ec2851d0dc7bf15aa261d614a75a10c7cb` |
| TaIrTe4 kz=2 | Eb | `c8068d9689ec827b33baabb194b48d5bf2b76c9cceb62cc97812f83f7b21a9e0` | `299a809e4ef1cc09755c782f8a321553472657e2dd1e050649650fe6c239bd0b` |

Certificate root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_sensitivity_certificate_94e854ac/`

- certificate: `FDTDX_FROZEN_Q_THERMAL_SENSITIVITY_CERTIFICATE.json`
- SHA-256:
  `d8a7f02b875e21e637668c8ebca17fcffbb020a471c23abda81deb4e760d2da8`
- generator commit:
  `94e854ac90bd3a4b0889c409c6f677158b609393`
- generator SHA-256:
  `21cb788c31ac104290c473be87130316574a942e7abc8042c705886e93b4208d`
- status:
  `VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_SCENARIO_SENSITIVITY_WITH_PHYSICAL_BOUNDS_BLOCKED`
- ready: true
- device-specific physical bounds supplied: false
- thermal domain and boundary converged: false
- production mesh selected: false
- optimizer start allowed: false

No raw NPZ, log, image, or iteration artifact is committed to Git.

## Required next information and action

1. Obtain the actual flake support/contact description: directly supported on
   thermally grown SiO2, contamination/encapsulation layers, suspended or
   partially supported regions, and measured or accepted TaIrTe4-SiO2 thermal
   boundary conductance.
2. Obtain a sourced or accepted TaIrTe4 cross-plane conductivity range for the
   actual thickness and temperature range.
3. Obtain measured/accepted Au-TaIrTe4 thermal and electrical contact ranges.
   The thermal gradient already changes by 5.7--10.8% across the numerical
   contact scenarios even when mean temperature barely changes.
4. Obtain the real flake/electrode/pad polygons, axis angle, terminal sign,
   oxide/substrate stack, and patterned-Au electrical role.
5. Only after those inputs exist should the actual-geometry thermal and
   electrical uncertainty ladders be rerun.  FDTDX optical and optimizer gates
   remain blocked independently.
