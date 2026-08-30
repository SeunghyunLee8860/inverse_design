# Run 060: +45-degree contacts, evaporated SiO2, E||b

![Fixed crystal axes and rotated device](rotated45_fixed_crystal_axes_geometry.png)

This is the E||b companion to Run 059. It uses the same immutable 24 um x
24 um TaIrTe4 square rotated +45 degrees relative to fixed global x=b and
y=a. Following the Run 058 optical approximation, Maxwell uses the centered
unrotated no-Au sheet while thermal/electrical geometry remains +45 degrees.
It uses the evaporated-SiO2
interface scenario, official Ansys DFM continuation, and exact 500 nm binary
repair/evaluation. As in Run 058, Au is absent from the optical and thermal
models; the opposite full-edge 2 um terminal strips are ideal equipotential
boundary masks only in the electrical weighting-field solve. Only the
incident polarization changes from E||a at 90 degrees to E||b at 0 degrees.

Run 060 is launched after Run 059 by the sequential launcher so that only one
physical GPU is used at a time.
