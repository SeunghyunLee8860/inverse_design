# TaIrTe4 GPU Pabs/flux root-cause report

## Conclusion

The IR slab test is valid. Its saved July 15 results were CPU runs, but a direct GPU replay of the same broadband/conformal FSP also agrees with CPU. The volume pipeline failed because it changed both the source/material bandwidth contract and the mesh contract. In v261 GPU, diagonal-anisotropic y-axis loss is lost for the narrow/standard-pulse plus coarse/precise-volume-average configuration. CPU is unaffected.

No Q clipping, flux gain, or HEAT was used.

## Direct evidence

- Original IR GPU replay: CPU/GPU spectrum RMSE R=1.805e-07, T=1.373e-07.
- Same IR FSP with 4 µm zero-span source: GPU A=-0.000226, TMM A=0.328489.
- Volume air/film/air, broadband but original precise/global mesh: GPU A=0.000126, TMM A=0.328489.
- Volume air/film/air, broadband + no global override + conformal variant 1 accuracy 5: GPU A=0.328904, TMM A=0.328489.
- Full flat stack corrected: GPU local A=0.422765, TMM A=0.423244.
- Patterned disk original: A_Q=0.508019448, local flux=0.453709357, mismatch=11.9702%.
- Patterned disk corrected: A_Q=0.466393700, local flux=0.466118738, mismatch=0.0590%.

## Code difference

IR_tairte4_test1.py uses 3-12 µm source, a 2.7-13.2 µm sampled-material table, conformal variant 1, mesh accuracy 5, and 5 nm film dz. The volume code used TARGET_WL_UM=4 only, sampled 3.6-4.4 µm, bandwidth=0, global_uniform_mesh, precise volume average, and mesh accuracy 2. The exporter then forced the global source start and stop both to 4 µm.

## Required production contract

1. Keep source/material fitting broadband enough to use Lumerical's broadband pulse path; record the exact source span.
2. Override pabs field/index and all flux monitors to one analysis point at 4 µm.
3. Remove global_uniform_mesh for this anisotropic GPU case.
4. Use auto non-uniform + conformal variant 1; accuracy 5 is the validated setting.
5. Keep the TaIrTe4 dz=5 nm override and the design dx=dy=25 nm override.
6. Fail closed unless flat y-pol Q/flux/TMM closure passes before patterned production runs.

## Time-convergence control under the failing contract

- 4 ps: Q-flux = 0.054310090
- 8 ps: Q-flux = 0.054296107
- 16 ps: Q-flux = 0.054296246

The mismatch converges rather than decays, so it is not a transient or auto-shutoff artifact.
