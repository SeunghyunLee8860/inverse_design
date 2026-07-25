import numpy as np
from tqdm import tqdm
import time
import shutil
import subprocess
import os
import sys
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path


# v5 B1: FoM-monitor spatial-interpolation modes selectable via
# MSOPT_FOM_INTERP (see _apply_env_fdtd_overrides). Default (env unset) is
# the legacy "nearest mesh cell" baked in at monitor creation.
_FOM_INTERP_MODES = {
    "nearest": "nearest mesh cell",
    "specified": "specified position",
    # v10 E2: raw Yee readout (doc: monitor Advanced tab option "NONE") -
    # needed to test whether the GPU engine's fixed convention equals 'none'.
    "none": "none",
}


def _discover_lumapi_path():
    explicit = os.environ.get("LUMERICAL_PYTHONPATH")
    if explicit:
        p = Path(explicit).expanduser()
        if (p / "lumapi.py").exists():
            return str(p)

    root = os.environ.get("LUMERICAL_ROOT")
    if root:
        p = Path(root).expanduser() / "api" / "python"
        if (p / "lumapi.py").exists():
            return str(p)

    candidates = sorted(Path("/opt/lumerical").glob("v*/api/python/lumapi.py")) if Path("/opt/lumerical").exists() else []
    if candidates:
        return str(candidates[-1].parent)
    return None


try:
    import lumapi
except ModuleNotFoundError:
    lumapi_dir = _discover_lumapi_path()
    if lumapi_dir and lumapi_dir not in sys.path:
        sys.path.insert(0, lumapi_dir)
    import lumapi


def _is_lumerical_messaging_error(exc):
    text = str(exc)
    markers = (
        "Failed to start messaging",
        "Failed to set up Ansys license sharing",
        "ANSYSLI exited or could not read server port",
        "Could not bind socket on port",
    )
    return any(marker in text for marker in markers)


def _open_lumerical_fdtd():
    session_hide = os.environ.get("LUMERICAL_SESSION_HIDE", "true").lower() == "true"
    session_platform = os.environ.get("LUMERICAL_SESSION_PLATFORM", "offscreen").strip()
    retries = int(os.environ.get("LUMERICAL_SESSION_OPEN_RETRIES", "6"))
    retry_delay = float(os.environ.get("LUMERICAL_SESSION_OPEN_RETRY_DELAY", "12"))

    attempts = []
    if session_platform:
        attempts.append({"hide": session_hide, "serverArgs": {"platform": session_platform}})
    attempts.append({"hide": session_hide})
    attempts.append({"hide": True})
    attempts.append({})

    last_error = None
    retries = max(1, retries)
    for retry_idx in range(retries):
        retryable_messaging_error = False
        for kwargs in attempts:
            try:
                fdtd = lumapi.FDTD(**kwargs)
                return fdtd
            except Exception as exc:
                last_error = exc
                if _is_lumerical_messaging_error(exc):
                    retryable_messaging_error = True
                    break
                continue

        if retryable_messaging_error and retry_idx < retries - 1:
            print(
                "Failed to open Lumerical FDTD session; "
                f"retrying in {retry_delay:g}s ({retry_idx + 1}/{retries - 1}). "
                f"Last error: {last_error}"
            )
            time.sleep(retry_delay)
            continue
        break

    raise RuntimeError(
        "Failed to open a Lumerical FDTD session. "
        f"Tried platform={session_platform!r} with fallback combinations. "
        f"Set LUMERICAL_ROOT/LUMERICAL_PYTHONPATH or adjust LUMERICAL_SESSION_PLATFORM. "
        f"Last error: {last_error}"
    )


def interpolate_index(index, wavelength):
    """
    Resolve scalar/anisotropic/dispersive index data at wavelength in um.

    Accepted forms:
        [n], [nx, ny, nz]
        complex n or complex anisotropic list
        callable: f(wavelength_um) -> n
        dict with wavelength/n/optional k arrays
    """
    if callable(index):
        return interpolate_index(index(wavelength), wavelength)

    if isinstance(index, dict):
        wl_data = index.get("wavelength", index.get("wavelengths", index.get("wl")))
        if wl_data is None:
            raise ValueError("Dispersive index dict requires wavelength/wavelengths/wl.")
        wl = np.asarray(wl_data, dtype=float)

        n_data = np.asarray(index["n"], dtype=float)
        k_data = np.asarray(index.get("k", np.zeros_like(n_data)), dtype=float)

        order = np.argsort(wl)
        wl = wl[order]
        n_data = n_data[order]
        k_data = k_data[order]

        if n_data.ndim == 1:
            n_val = np.interp(wavelength, wl, n_data)
            k_val = np.interp(wavelength, wl, k_data)
            return np.array([n_val + 1j * k_val], dtype=np.complex128)

        vals = []
        for comp in range(n_data.shape[1]):
            n_val = np.interp(wavelength, wl, n_data[:, comp])
            k_val = np.interp(wavelength, wl, k_data[:, comp])
            vals.append(n_val + 1j * k_val)
        return np.asarray(vals, dtype=np.complex128)

    arr = np.asarray(index, dtype=np.complex128).reshape(-1)
    if arr.size not in (1, 3):
        raise ValueError("Index must be scalar, length-1, length-3, callable, or dispersion dict.")
    return arr


def _dispersion_table(index):
    if not isinstance(index, dict):
        return None
    wl_data = index.get("wavelength", index.get("wavelengths", index.get("wl")))
    if wl_data is None or "n" not in index:
        return None

    wl = np.asarray(wl_data, dtype=float).reshape(-1)
    n_data = np.asarray(index["n"], dtype=float)
    k_data = np.asarray(index.get("k", np.zeros_like(n_data)), dtype=float)
    if n_data.ndim == 1:
        n_data = n_data[:, None]
    if k_data.ndim == 1:
        k_data = k_data[:, None]
    if n_data.shape != k_data.shape or n_data.shape[0] != wl.size:
        raise ValueError(
            "Dispersive material table shape mismatch. Expected wavelength length "
            "to match first dimension of n and k."
        )

    if n_data.shape[1] not in (1, 3):
        raise ValueError("Dispersive material n/k table must be isotropic or 3-axis anisotropic.")

    order = np.argsort(wl)
    return wl[order], n_data[order], k_data[order]


class LumericalFDTDSimulator:
    def __init__(
        self, 
        sim_size=[0,0,0],
        resolution: int = None,                 # [cells per unit], e.g., per 'unit'
        points_per_wavelength: int = None,      # ppw at center_wl in background_index
        bc_x: str = "PML",
        bc_y: str = "PML",
        bc_z: str = "PML",
        unit: float = 1e-6,                     # 1.0 => meters; 1e-6 => inputs in um
        background_index: float = 1.0,
        center_wl: float = 1.0,
        N_f: int = 1,
        create_global_uniform_mesh: bool = True,
    ):
        self.dims = ['x', 'y', 'z']
        self.kw_to_idx = {kw: i for i, kw in enumerate(self.dims)}
        self.idx_to_kw = {i: kw for i, kw in enumerate(self.dims)}
        self.fdtd = _open_lumerical_fdtd()
        self.fdtd.switchtolayout()

        self.c=299792458
        # keep commonly used attrs
        self.unit = unit
        self.N_f = N_f
        self.center_wl = center_wl*unit
        self.center_wl_um = center_wl
        self.n_bg = background_index
        self.resolution = resolution
        self.ppw = points_per_wavelength
        self.create_global_uniform_mesh = bool(create_global_uniform_mesh)
        self.bc = {'x': bc_x, 'y': bc_y, 'z': bc_z}
        self.material_cache = {}
        self.src_wl = np.asarray([center_wl], dtype=float).reshape(-1) * self.unit
        self.src_bw = 0.0

        # FDTD region
        self.fdtd.addfdtd()
        self.fdtd.set("index", background_index)
        self.fdtd.set("auto shutoff min", 1e-4)

        self.sim_center, self.sim_size = self.unit_scaling(center=[0,0,0], size=sim_size)
        nz_dims, n_axis = self.dimension_check(sim_size)
        if len(nz_dims) == 2:  
            self.fdtd.set("dimension", "2D")
        elif len(nz_dims) == 3:
            self.fdtd.set("dimension", "3D")
        else:
            raise ValueError("Simulation must span at least 2 dimensions")

        # region size + BC
        for dim in nz_dims:
            idx = self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", self.sim_center[idx])
            self.fdtd.set(f"{dim} span", self.sim_size[idx])
            self.fdtd.set(f"{dim} min bc", self.bc[dim])
            self.fdtd.set(f"{dim} max bc", self.bc[dim])

        if len(nz_dims) == 2:
            # set the normal axis to something consistent (e.g., Bloch or PML).
            # In Lumerical's 2D mode (X-Y plane) the out-of-plane (z) boundary
            # is inactive, so guard against "property inactive" errors.
            try:
                self.fdtd.set(f"{n_axis} min bc", "Bloch")
                self.fdtd.set(f"{n_axis} max bc", "Bloch")
            except Exception as _bc_exc:
                print(f"[FDTD] skip out-of-plane '{n_axis}' BC (inactive in 2D): {_bc_exc}")

        # --- MESH CONTROL ---
        # If resolution or ppw specified -> create a single mesh override that spans the whole domain
        if self.resolution or self.ppw:
            # compute target step
            if self.resolution:
                # resolution: cells per 'unit' (e.g., per micron if unit=1e-6)
                self.sim_grid = self.unit / float(self.resolution)
                mode_desc = f"uniform dx=dy=dz={self.sim_grid:.3e} m (resolution={self.resolution} per {self.unit:g} m)"
            else:
                # ppw: points per wavelength in background
                self.sim_grid = self.center_wl / (float(self.ppw))
                mode_desc = f"uniform dx=dy=dz={self.sim_grid:.3e} m (ppw={self.ppw} @ λ0={self.center_wl:g} m, n={self.n_bg})"

            if self.create_global_uniform_mesh:
                # Legacy opt-in path. Production auto-mesh builders set this
                # false and keep sim_grid only as the import-grid coordinate
                # reference; no whole-domain mesh object is created.
                self.fdtd.addmesh()
                self.fdtd.set("name", "global_uniform_mesh")
                for dim in nz_dims:
                    idx = self.kw_to_idx[dim]
                    self.fdtd.set(dim, self.sim_center[idx])
                    self.fdtd.set(f"{dim} span", self.sim_size[idx])

                self.fdtd.set("override x mesh", 1)
                self.fdtd.set("override y mesh", 1)
                self.fdtd.set("override z mesh", 1)
                self.fdtd.set("dx", float(self.sim_grid))
                self.fdtd.set("dy", float(self.sim_grid))
                self.fdtd.set("dz", float(self.sim_grid))
                print(f"[FDTD] Mesh override enabled: {mode_desc}")
            else:
                print(
                    "[FDTD] Mesh coordinate reference only; "
                    f"global override disabled ({mode_desc})"
                )
        else:
            # fallback: use mesh accuracy only
            self.fdtd.set("mesh accuracy", 2)
            print("[FDTD] Mesh: default mesh accuracy = 2 (no override)")

    def dimension_check(self, size):
        nonzero_dims = [d for d, s in zip(self.dims, size) if s > 0]
        if len(nonzero_dims) == 2:
            normal_axis = [d for d in self.dims if d not in nonzero_dims][0]
        else:
            normal_axis = None
        return nonzero_dims, normal_axis

    def unit_scaling(self, center, size):
        center=[e * self.unit for e in center]
        size=[e * self.unit for e in size]
        return center, size
    
    def add_source(
            self, 
            mode="custom", 
            name="source", 
            center=[0,0,0], 
            size=[1,1,0], 
            direction="forward", 
            src_wl=[1.0], 
            bandwidth=0.2, 
            Fields=None,
            pol=45,
            theta=0.0,
            phi=0.0,
            single=False,
            mode_num=0,
            broadband=False,
        ):
        self.src_wl = np.asarray(src_wl, dtype=float).reshape(-1) * self.unit
        self.src_bw=0.5*bandwidth
        wl_min = float(np.min(self.src_wl))
        wl_max = float(np.max(self.src_wl))
        nz_dims, n_axis = self.dimension_check(size)
        center, size = self.unit_scaling(center, size=size)
        idx=self.kw_to_idx[n_axis]
        if mode == "custom":
            if Fields is None:
                raise ValueError("Fields required for custom mode")
            dJEx, dJEy, dJEz = Fields
            u_ax, v_ax  = self.kw_to_idx[nz_dims[0]], self.kw_to_idx[nz_dims[1]]
            Su, Sv = size[u_ax], size[v_ax]
            n_ax=self.kw_to_idx[n_axis]

            def full_field_array(arr, nfreq=None):
                if arr is None:
                    return None
                arr = np.asarray(arr, dtype=np.complex128)
                if arr.ndim == 2:
                    shape3 = [1, 1, 1]
                    shape3[u_ax] = arr.shape[0]
                    shape3[v_ax] = arr.shape[1]
                    out = np.zeros(shape3, dtype=np.complex128)
                    if (u_ax, v_ax) == (0, 1):
                        out[:, :, 0] = arr
                    elif (u_ax, v_ax) == (0, 2):
                        out[:, 0, :] = arr
                    elif (u_ax, v_ax) == (1, 2):
                        out[0, :, :] = arr
                    else:
                        raise ValueError("Unexpected source plane axes.")
                    return out
                if arr.ndim == 3 and nfreq is not None and arr.shape[-1] == nfreq and len(nz_dims) == 2:
                    shape4 = [1, 1, 1, nfreq]
                    shape4[u_ax] = arr.shape[0]
                    shape4[v_ax] = arr.shape[1]
                    out = np.zeros(shape4, dtype=np.complex128)
                    if (u_ax, v_ax) == (0, 1):
                        out[:, :, 0, :] = arr
                    elif (u_ax, v_ax) == (0, 2):
                        out[:, 0, :, :] = arr
                    elif (u_ax, v_ax) == (1, 2):
                        out[0, :, :, :] = arr
                    else:
                        raise ValueError("Unexpected source plane axes.")
                    return out
                if arr.ndim == 3:
                    return arr
                if arr.ndim == 4:
                    return arr
                raise ValueError(f"Unexpected imported field shape {arr.shape}.")

            nfreq = len(self.src_wl)
            if broadband and nfreq <= 1:
                broadband = False

            dJEx = full_field_array(dJEx, nfreq=nfreq if broadband else None)
            dJEy = full_field_array(dJEy, nfreq=nfreq if broadband else None)
            dJEz = full_field_array(dJEz, nfreq=nfreq if broadband else None)
            base = next(arr for arr in [dJEx, dJEy, dJEz] if arr is not None)
            if broadband:
                if base.ndim != 4 or base.shape[-1] != nfreq:
                    raise ValueError(
                        "Broadband custom source requires field arrays with final wavelength axis "
                        f"of length {nfreq}; got {base.shape}."
                    )
            Nu, Nv = base.shape[u_ax], base.shape[v_ax]

            du, dv = Su / float(Nu-1), Sv / float(Nv-1)
            u0 = center[u_ax] - 0.5*Su
            v0 = center[v_ax] - 0.5*Sv
            u_grids = u0 + np.arange(Nu)*du
            v_grids = v0 + np.arange(Nv)*dv


            self.u_dim=nz_dims[0]
            self.v_dim=nz_dims[1]
            self.nor_dim=n_axis
            self.norm_pt=center[n_ax]

            def add_imported_source_object(source_name):
                self.fdtd.addimportedsource()
                self.fdtd.set("name", source_name)
                self.fdtd.set("injection axis", self.nor_dim)
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("z", center[2])
                try:
                    self.fdtd.set("wavelength start", wl_min if bandwidth == 0 else wl_min * (1 - self.src_bw))
                    self.fdtd.set("wavelength stop", wl_max if bandwidth == 0 else wl_max * (1 + self.src_bw))
                except Exception:
                    pass

                pts=[0]*3
                pts[u_ax]=u_grids
                pts[v_ax]=v_grids
                pts[n_ax]=self.norm_pt
                self.fdtd.putv('field',self.fdtd.rectilineardataset("field", pts[0],pts[1],pts[2]))

            if broadband:
                add_imported_source_object(name)
                self.fdtd.putv('Ex', dJEx)
                self.fdtd.putv('Ey', dJEy)
                self.fdtd.putv('Ez', dJEz)
                self.fdtd.putv('src_lambdas', self.src_wl)
                self.fdtd.putv('src_freqs', self.c / self.src_wl)
                self.EM_dataset = 'field.addparameter("lambda",src_lambdas,"f",src_freqs);'
                self.EM_dataset += 'field.addattribute("E",Ex,Ey,Ez);'
                self.EM_dataset += 'importdataset(field);'
                self.fdtd.eval(f"{self.EM_dataset}\n")
            else:
                for iidx in range(len(self.src_wl)):
                    source_name = name if len(self.src_wl) == 1 else f"{name}_{iidx}"
                    add_imported_source_object(source_name)
                    self.fdtd.putv('Ex', dJEx[:, :, :, iidx] if dJEx is not None and dJEx.ndim == 4 else dJEx)
                    self.fdtd.putv('Ey', dJEy[:, :, :, iidx] if dJEy is not None and dJEy.ndim == 4 else dJEy)
                    self.fdtd.putv('Ez', dJEz[:, :, :, iidx] if dJEz is not None and dJEz.ndim == 4 else dJEz)
                    self.EM_dataset=f"field.addparameter(\"lambda\",{self.src_wl[iidx]},\"f\",{self.c / self.src_wl[iidx]});"
                    self.EM_dataset+=f"field.addattribute(\"E\",Ex,Ey,Ez);"
                    self.EM_dataset+=f"importdataset(field);"
                    self.fdtd.eval(f"{self.EM_dataset}\n")
                

        elif mode =="eigen":
            self.fdtd.addmode()
            self.fdtd.set('mode selection', 'user select')
            self.fdtd.set('selected mode number', int(mode_num))
            self.fdtd.set("name", name)
            self.fdtd.set("injection axis", n_axis)
            self.fdtd.set("direction", direction)
            self.fdtd.set("x", center[0])
            self.fdtd.set("y", center[1])
            self.fdtd.set("z", center[2])
            if size[0] !=0:
                self.fdtd.set("x span", size[0])
            if size[1] !=0:
                self.fdtd.set("y span", size[1])
            if size[2] !=0:
                self.fdtd.set("z span", size[2])
            if bandwidth == 0:
                self.fdtd.eval(f"set(\"wavelength start\", {wl_min});\nset(\"wavelength stop\", {wl_max});")
            else:
                self.fdtd.eval(
                    f"set(\"wavelength start\", {wl_min*(1-self.src_bw)});\n"
                    f"set(\"wavelength stop\", {wl_max*(1+self.src_bw)});"
                )
        else:
            if single:
                self.fdtd.addplane()
                self.fdtd.set("name", name)
                self.fdtd.set("injection axis", n_axis)
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("z", center[2])
                if size[0] !=0:
                    self.fdtd.set("x span", size[0])
                if size[1] !=0:
                    self.fdtd.set("y span", size[1])
                if size[2] !=0:
                    self.fdtd.set("z span", size[2])
                self.fdtd.set("polarization angle",pol)
                self._set_plane_wave_angles(theta, phi)
                self.fdtd.eval(f"set(\"wavelength start\", {np.min(self.src_wl)});")
                self.fdtd.eval(f"set(\"wavelength stop\", {np.max(self.src_wl)});")
            else:
                for idx in range(len(src_wl)):
                    self.fdtd.addplane()
                    if len(src_wl)==1:
                        self.fdtd.set("name", name)
                    else:
                        self.fdtd.eval(f"addtogroup(\"{name}\");")
                    self.fdtd.set("injection axis", n_axis)
                    self.fdtd.set("direction", direction)
                    self.fdtd.set("x", center[0])
                    self.fdtd.set("y", center[1])
                    self.fdtd.set("z", center[2])
                    if size[0] !=0:
                        self.fdtd.set("x span", size[0])
                    if size[1] !=0:
                        self.fdtd.set("y span", size[1])
                    if size[2] !=0:
                        self.fdtd.set("z span", size[2])
                    self.fdtd.set("polarization angle",pol)
                    self._set_plane_wave_angles(theta, phi)
                    self.fdtd.eval(f"set(\"wavelength start\", {self.src_wl[idx]*(1-self.src_bw)});")
                    self.fdtd.eval(f"set(\"wavelength stop\", {self.src_wl[idx]*(1+self.src_bw)});")
            # self.fdtd.set('optimize for short pulse', False)

    def _set_plane_wave_angles(self, theta=0.0, phi=0.0):
        for prop, value in (
            ("angle theta", theta),
            ("angle phi", phi),
            ("theta", theta),
            ("phi", phi),
        ):
            try:
                self.fdtd.set(prop, float(value))
            except Exception:
                pass
            
    def add_monitor(
            self, 
            name="field_monitor",
            center=[0,0,0], 
            size=[1,1,1],
            N_f=1, 
        ):
        center, size = self.unit_scaling(center=center, size=size)
        nz_dims, n_axis = self.dimension_check(size)

        if len(nz_dims) == 0:
            monitor_type = f"Point"
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Monitor must span at least 1 dimension")
        
        self.fdtd.adddftmonitor()
        self.fdtd.set("name", name)
        self.fdtd.set("monitor type", monitor_type)

        for dim in self.dims:
            idx=self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", center[idx])
            if dim !=n_axis and len(nz_dims) != 0:
                self.fdtd.set(f"{dim} span", size[idx])

        self.fdtd.set("override global monitor settings", True)
        self.fdtd.set("spatial interpolation", "nearest mesh cell")
        # self.fdtd.set("spatial interpolation", "specified position")
        # self.fdtd.set("spatial interpolation", "none")
        if N_f>len(self.src_wl):
            self.fdtd.set("frequency points",N_f)
        else:
            self.fdtd.set("frequency points",len(self.src_wl))
            self.fdtd.set("sample spacing","custom")
            self.fdtd.set("custom frequency samples",np.array(self.c/np.array(self.src_wl)))

    def _create_sampled_material(self, index, material_name=None, max_coefficients=6):
        table = _dispersion_table(index)
        if table is None:
            raise ValueError("Sampled material requires a dispersive index dict.")

        wl_um, n_data, k_data = table
        if material_name is None:
            material_name = index.get("name", "sampled_material")

        if material_name in self.material_cache:
            return self.material_cache[material_name]

        f = self.c / (wl_um * self.unit)
        eps = (n_data + 1j * k_data) ** 2
        if eps.shape[1] == 1:
            sampled_data = np.column_stack([f, eps[:, 0]])
        else:
            sampled_data = np.column_stack([f, eps[:, 0], eps[:, 1], eps[:, 2]])

        temp = self.fdtd.addmaterial(index.get("material_type", "Sampled data"))
        self.fdtd.setmaterial(temp, "name", material_name)
        try:
            self.fdtd.setmaterial(material_name, "max coefficients", int(index.get("max_coefficients", max_coefficients)))
        except Exception:
            pass
        self.fdtd.setmaterial(material_name, "sampled data", sampled_data)
        try:
            self.fdtd.setmaterial(material_name, "tolerance", float(index.get("tolerance", 0.0)))
        except Exception:
            pass

        self.material_cache[material_name] = material_name
        return material_name

    def _set_object_index(self, index, object_name=None, material_name=None, wavelength=None, dispersive=True):
        if isinstance(index, str):
            self.fdtd.set("material", index)
            return

        dispersion = _dispersion_table(index) if isinstance(index, dict) else None
        if isinstance(index, dict) and dispersive and dispersion is not None and dispersion[0].size >= 2:
            if material_name is None:
                material_name = index.get("name", object_name + "_mat" if object_name else "sampled_material")
            mat = self._create_sampled_material(index, material_name=material_name)
            self.fdtd.set("material", mat)
            return

        wl = self.center_wl_um if wavelength is None else wavelength
        idx = interpolate_index(index, wl)
        n_real = np.real(idx)
        k_imag = np.imag(idx)

        if idx.size == 1 and np.max(np.abs(k_imag)) < 1e-15:
            self.fdtd.set("index", float(n_real[0]))
            return

        mat = self.fdtd.addmaterial("(n,k) Material")
        if material_name is None:
            material_name = object_name + "_mat" if object_name else "dispersive_mat"
        try:
            self.fdtd.setmaterial(mat, "name", material_name)
            mat = material_name
        except Exception:
            pass

        if idx.size == 3:
            self.fdtd.setmaterial(mat, "Anisotropy", 1)
            self.fdtd.setmaterial(mat, "Refractive Index", np.asarray(n_real, dtype=float))
            try:
                self.fdtd.setmaterial(mat, "Imaginary Refractive Index", np.asarray(k_imag, dtype=float))
            except Exception:
                if np.max(np.abs(k_imag)) > 1e-15:
                    print(f"[material] Warning: could not set anisotropic k for {material_name}.")
        else:
            self.fdtd.setmaterial(mat, "Refractive Index", float(n_real[0]))
            try:
                self.fdtd.setmaterial(mat, "Imaginary Refractive Index", float(k_imag[0]))
            except Exception:
                if abs(k_imag[0]) > 1e-15:
                    print(f"[material] Warning: could not set k for {material_name}.")
        self.fdtd.set("material", mat)

    def add_geo(
            self,
            center=[0,0,0],
            size=[1,1,1],
            index=[1.0],
            name=None,
            wavelength=None,
            material_name=None,
            dispersive=True,
        ):
        center, size = self.unit_scaling(center=center, size=size)
        self.fdtd.addrect()
        if name:
            self.fdtd.set("name", name)
        for dim in self.dims:
            idx=self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", center[idx])
            self.fdtd.set(f"{dim} span", size[idx])

        self._set_object_index(
            index,
            object_name=name,
            material_name=material_name,
            wavelength=wavelength,
            dispersive=dispersive,
        )


    def add_waveguide(
        self,
        center=[0, 0, 0],
        length=1.0,
        top_width=0.5,
        bottom_width=0.7,
        height=0.3,
        prop_axis="z",
        width_axis="y",
        index=[4.8855, 4.5836, 4.8855],
        name="wg",
        wavelength=None,
        material_name=None,
        dispersive=True,
    ):
        # units
        center_s, _ = self.unit_scaling(center=center, size=[0, 0, 0])
        length_s = length * self.unit
        top_w_s = top_width * self.unit
        bottom_w_s = bottom_width * self.unit
        height_s = height * self.unit

        x0, y0, z0 = center_s

        # Lumerical waveguide object uses base angle instead of separate top/bottom widths
        # angle = sidewall angle from horizontal base in degrees
        half_diff = max(bottom_w_s - top_w_s, 0.0) * 0.5
        if half_diff == 0:
            base_angle_deg = 90.0
        else:
            base_angle_deg = np.degrees(np.arctan(height_s / half_diff))

        self.fdtd.addwaveguide()
        self.fdtd.set("name", name)
        self._set_object_index(
            index,
            object_name=name,
            material_name=material_name,
            wavelength=wavelength,
            dispersive=dispersive,
        )
        self.fdtd.set("base width", float(bottom_w_s))
        self.fdtd.set("base height", float(height_s))
        self.fdtd.set("base angle", float(base_angle_deg))

        # straight waveguide centerline
        if prop_axis == "z":
            poles = np.array([
                [x0, z0 - 0.5 * length_s],
                [x0, z0 + 0.5 * length_s],
            ])
            if width_axis == "y":
                self.fdtd.set("first axis", "x")
                self.fdtd.set("rotation 1", 90)
                self.fdtd.set("third axis", "z")
                self.fdtd.set("rotation 3", 90)
            self.fdtd.set("x", x0)
            self.fdtd.set("y", y0)
            self.fdtd.set("z", 0)


        elif prop_axis == "y":
            poles = np.array([
                [x0, y0 - 0.5 * length_s],
                [x0, y0 + 0.5 * length_s],
            ])
            self.fdtd.set("x", x0)
            self.fdtd.set("y", 0)
            self.fdtd.set("z", z0)

        elif prop_axis == "x":
            poles = np.array([
                [x0 - 0.5 * length_s, y0],
                [x0 + 0.5 * length_s, y0],
            ])
            self.fdtd.set("x", x0)
            self.fdtd.set("y", y0)
            self.fdtd.set("z", z0)

        else:
            raise ValueError("prop_axis='z' or 'y' or 'x'.")
        self.fdtd.set("poles", poles)


    def _require_uniform_mesh(self):
        if not hasattr(self, "sim_grid") or self.sim_grid is None:
            raise RuntimeError(
                "Exact design-field alignment requires a uniform mesh override. "
                "Set resolution=... or points_per_wavelength=... when creating the simulator."
            )

    def _snap_to_mesh(self, value):
        self._require_uniform_mesh()
        return np.round(np.asarray(value, dtype=float) / self.sim_grid) * self.sim_grid

    def _node_axis(self, center, N, d):
        self._require_uniform_mesh()
        target_center = float(center)

        if N <= 1:
            return np.array([float(self._snap_to_mesh(target_center))], dtype=float)

        # Keep the grid symmetric about the snapped center. For even N, this
        # intentionally places nodes on the half-cell phase used by Lumerical
        # DFT monitor sampling instead of snapping the first node to a full cell.
        return target_center + (np.arange(N, dtype=float) - 0.5 * (N - 1)) * d


    def _add_or_update_design_mesh_override(self):
        """
        Make the design region itself use the same uniform mesh as the import grid.
        """
        self._require_uniform_mesh()
        name = f"{self.design_name}_mesh"

        if self.fdtd.getnamednumber(name) == 0:
            self.fdtd.addmesh()
            self.fdtd.set("name", name)
        else:
            self.fdtd.eval(f'select("{name}");')

        extents = [
            self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0),
            self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1),
            self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2),
        ]
        for dim, (vmin, vmax) in zip(self.dims, extents):
            self.fdtd.set(f"{dim} min", float(vmin))
            self.fdtd.set(f"{dim} max", float(vmax))

        self.fdtd.set("override x mesh", 1)
        self.fdtd.set("override y mesh", 1)
        self.fdtd.set("override z mesh", 1)
        self.fdtd.set("dx", float(self.design_dx))
        self.fdtd.set("dy", float(self.design_dy))
        self.fdtd.set("dz", float(self.design_dz))

    def _monitor_extent_from_design_axis(self, axis, d, axis_idx):
        axis = np.atleast_1d(np.asarray(axis, dtype=float))
        return float(np.min(axis)), float(np.max(axis))

    def _assert_design_monitor_alignment(self, monitor_name=None, atol=None):
        """
        Check that the monitor grid returned by Lumerical matches the design grid.
        """
        if monitor_name is None:
            monitor_name = self.design_monitor_name
        if atol is None:
            atol = 1e-12

        Eres = self.fdtd.getresult(monitor_name, "E")
        xm = np.atleast_1d(np.squeeze(np.array(Eres["x"], dtype=float)))
        ym = np.atleast_1d(np.squeeze(np.array(Eres["y"], dtype=float)))
        zm = np.atleast_1d(np.squeeze(np.array(Eres["z"], dtype=float)))

        self._last_design_monitor_slices = [
            self._design_monitor_axis_slice("x", xm),
            self._design_monitor_axis_slice("y", ym),
            self._design_monitor_axis_slice("z", zm),
        ]
        xm = xm[self._last_design_monitor_slices[0]]
        ym = ym[self._last_design_monitor_slices[1]]
        zm = zm[self._last_design_monitor_slices[2]]

        self._adopt_close_monitor_axis("x", xm)
        self._adopt_close_monitor_axis("y", ym)
        self._adopt_close_monitor_axis("z", zm)

        if xm.size != self.design_x.size or not np.allclose(xm, self.design_x, atol=atol, rtol=0):
            raise RuntimeError(
                f"x-grid mismatch: monitor={xm.shape}/{xm[:3]}..., design={self.design_x.shape}/{self.design_x[:3]}..."
            )
        if ym.size != self.design_y.size or not np.allclose(ym, self.design_y, atol=atol, rtol=0):
            raise RuntimeError(
                f"y-grid mismatch: monitor={ym.shape}/{ym[:3]}..., design={self.design_y.shape}/{self.design_y[:3]}..."
            )
        if zm.size != self.design_z.size or not np.allclose(zm, self.design_z, atol=atol, rtol=0):
            raise RuntimeError(
                f"z-grid mismatch: monitor={zm.shape}/{zm[:3]}..., design={self.design_z.shape}/{self.design_z[:3]}..."
            )

    def _design_monitor_axis_slice(self, axis_name, monitor_axis):
        design_axis = np.atleast_1d(np.asarray(getattr(self, f"design_{axis_name}"), dtype=float))
        monitor_axis = np.atleast_1d(np.asarray(monitor_axis, dtype=float))
        if monitor_axis.size == design_axis.size:
            return slice(None)
        if monitor_axis.size < design_axis.size:
            return slice(None)   # fewer samples than nodes: unrecoverable here

        d = float(getattr(self, f"design_d{axis_name}"))
        adopt_atol = max(1e-9, 0.6 * abs(d))
        if monitor_axis.size == design_axis.size + 1:
            head = monitor_axis[: design_axis.size]
            tail = monitor_axis[-design_axis.size :]
            head_delta = float(np.max(np.abs(head - design_axis)))
            tail_delta = float(np.max(np.abs(tail - design_axis)))
            if head_delta <= adopt_atol and head_delta <= tail_delta:
                print(f"[design monitor] cropping extra trailing {axis_name}-sample (no interpolation)")
                return slice(0, design_axis.size)
            if tail_delta <= adopt_atol:
                print(f"[design monitor] cropping extra leading {axis_name}-sample (no interpolation)")
                return slice(1, None)
            return slice(None)
        # LOCAL FORK: a locally refined/graded mesh (e.g. the 5nm flake override
        # grading into the design region above z=0) yields MORE monitor samples
        # than design nodes. Select the nearest raw Yee sample per design node
        # (no interpolation). Tolerance per node: 0.6*step, else fall through so
        # the alignment assert still fails loudly.
        if monitor_axis.size > design_axis.size:
            idx = np.array([int(np.argmin(np.abs(monitor_axis - dv))) for dv in design_axis])
            deltas = np.abs(monitor_axis[idx] - design_axis)
            if np.unique(idx).size == idx.size and float(np.max(deltas)) <= adopt_atol:
                print(
                    f"[design monitor] nearest-sample selection on {axis_name}: "
                    f"{monitor_axis.size} samples -> {design_axis.size} nodes "
                    f"(max shift {float(np.max(deltas)):.3e} m, no interpolation)"
                )
                return idx
        return slice(None)

    def _adopt_close_monitor_axis(self, axis_name, monitor_axis):
        design_attr = f"design_{axis_name}"
        d_attr = f"design_d{axis_name}"
        axis_idx = self.kw_to_idx[axis_name]

        design_axis = np.atleast_1d(np.asarray(getattr(self, design_attr), dtype=float))
        monitor_axis = np.atleast_1d(np.asarray(monitor_axis, dtype=float))
        if monitor_axis.size != design_axis.size:
            return

        if np.allclose(monitor_axis, design_axis, atol=1e-12, rtol=0):
            return

        d = float(getattr(self, d_attr))
        adopt_atol = max(1e-9, 0.6 * abs(d))
        max_delta = float(np.max(np.abs(monitor_axis - design_axis)))
        if max_delta > adopt_atol:
            return

        print(
            f"[design monitor] adopting Lumerical {axis_name}-grid "
            f"(max shift {max_delta:.3e} m, no field interpolation)"
        )
        setattr(self, design_attr, monitor_axis.copy())
        if monitor_axis.size > 1:
            setattr(self, d_attr, float(np.mean(np.diff(monitor_axis))))
        self.design_center[axis_idx] = float(0.5 * (monitor_axis[0] + monitor_axis[-1]))
        self.design_size[axis_idx] = float(monitor_axis[-1] - monitor_axis[0])

    def set_spatial_interp(self, monitor_name, setting="specified position"):
        script = f'select("{monitor_name}"); set("spatial interpolation","{setting}");'
        self.fdtd.eval(script)


    def _configure_design_region_objects(self):
        """
        Design region monitor / index monitor / mesh override를
        design_x, design_y, design_z와 정확히 같은 좌표계로 맞춘다.
        """
        xmin, xmax = self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0)
        ymin, ymax = self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1)
        zmin, zmax = self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2)

        # DFT monitor
        if self.fdtd.getnamednumber(self.design_monitor_name) == 0:
            self.fdtd.adddftmonitor()
            self.fdtd.set("name", self.design_monitor_name)

        nz_dims, n_axis = self.dimension_check([xmax - xmin, ymax - ymin, zmax - zmin])
        if len(nz_dims) == 1:
            monitor_type = "1D"
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Monitor must span at least 1 dimension")

        script = (
            f'select("{self.design_monitor_name}");'
            f'set("monitor type","{monitor_type}");'
            f'set("x min",{xmin});'
            f'set("x max",{xmax});'
            f'set("y min",{ymin});'
            f'set("y max",{ymax});'
            f'set("z min",{zmin});'
            f'set("z max",{zmax});'
            f'set("override global monitor settings",1);'
        )
        self.fdtd.eval(script)
        # self.set_spatial_interp(self.design_monitor_name, "nearest mesh cell")
        self.set_spatial_interp(self.design_monitor_name, "none")
        # self.set_spatial_interp(self.design_monitor_name, "specified position")

        try:
            self.fdtd.setnamed(self.design_monitor_name, "down sample x", 1)
            self.fdtd.setnamed(self.design_monitor_name, "down sample y", 1)
            self.fdtd.setnamed(self.design_monitor_name, "down sample z", 1)
        except Exception:
            pass

        self._configure_design_monitor_spectral_settings()
        self._add_or_update_design_mesh_override()
        self._configure_design_index_monitor()
        # LOCAL FORK: honor an explicit floor if the app set one. The legacy
        # value (= design grid step) silently CLAMPS any finer mesh override
        # (e.g. the validated 5nm flake mesh) and is re-applied on every
        # reconfigure, so it must be overridable here, not just once outside.
        _min_step = getattr(self, "min_mesh_step_m", None)
        if _min_step is None:
            _min_step = min(self.design_dx, self.design_dy, self.design_dz)
        self.fdtd.eval(
            f'setnamed("FDTD","min mesh step",{_min_step});'
        )

    def _configure_design_monitor_spectral_settings(self):
        if not hasattr(self, "src_wl") or self.src_wl is None:
            return
        if self.fdtd.getnamednumber(self.design_monitor_name) == 0:
            return
        
        try:
            self.fdtd.setnamed(self.design_monitor_name, "override global monitor settings", True)
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "use source limits", False)
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "sample spacing", "custom")
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "custom frequency samples",
                            np.array(self.c / np.array(self.src_wl)))
        except Exception:
            try:
                self.fdtd.setnamed(self.design_monitor_name, "frequency points", len(self.src_wl))
            except Exception:
                pass

    def _configure_design_index_monitor(self):
        """
        Create/update an index monitor that matches the design grid extent exactly.
        """
        xmin, xmax = self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0)
        ymin, ymax = self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1)
        zmin, zmax = self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2)

        name = self.design_index_monitor_name

        if self.fdtd.getnamednumber(name) == 0:
            self.fdtd.addindex()
            self.fdtd.set("name", name)

        nz_dims, n_axis = self.dimension_check([xmax - xmin, ymax - ymin, zmax - zmin])
        if len(nz_dims) == 1:
            monitor_type = "linear " + nz_dims[0]
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Index monitor must span at least 1 dimension")

        script = (
            f'select("{name}");'
            f'set("monitor type","{monitor_type}");'
            f'set("x min",{xmin});'
            f'set("x max",{xmax});'
            f'set("y min",{ymin});'
            f'set("y max",{ymax});'
            f'set("z min",{zmin});'
            f'set("z max",{zmax});'
        )
        self.fdtd.eval(script)

        try:
            self.fdtd.setnamed(name, "spatial interpolation", "none")
        except Exception:
            pass

        try:
            self.fdtd.setnamed(name, "down sample x", 1)
            self.fdtd.setnamed(name, "down sample y", 1)
            self.fdtd.setnamed(name, "down sample z", 1)
        except Exception:
            pass

    def density2idx(
            self,
            density
        ):
        d_copy = np.clip(np.array(density.copy()).flatten(), 0.0, 1.0)
        rho=[0]*3
        for ax in range(3):
            n_idx=np.array((d_copy*(self.index1[ax]**2-self.index2[ax]**2)))
            n_idx+=(self.index2[ax]**2)*np.ones(self.design_grids[0]*self.design_grids[1]*self.design_grids[2])
            rho[ax]=np.array(np.sqrt(n_idx))
        return rho[0], rho[1], rho[2]

    def add_design_grid(
            self,
            name="design",
            center=[0, 0, 0],
            size=[1, 1, 1],
            index1=[1.5],
            index2=[1.0],
            design_grids=[50, 50, 50],
            density=None,
            wavelength=None,
            grid_steps=None,   # optional [dx, dy, dz] in METERS for the design import
                               # grid / design-region mesh; None -> global sim_grid
                               # on every axis (legacy behavior, backward compatible)
        ):
        self._require_uniform_mesh()

        self.design_name = name
        self.design_monitor_name = self.design_name + "_monitor"
        self.design_index_monitor_name = self.design_name + "_index_monitor"

        self.design_grids = list(design_grids)
        Nx, Ny, Nz = self.design_grids

        wl = self.center_wl_um if wavelength is None else wavelength
        index1_resolved = interpolate_index(index1, wl)
        index2_resolved = interpolate_index(index2, wl)
        if index1_resolved.size == 1:
            index1_resolved = np.repeat(index1_resolved, 3)
        if index2_resolved.size == 1:
            index2_resolved = np.repeat(index2_resolved, 3)

        if np.max(np.abs(np.imag(index1_resolved))) > 1e-15 or np.max(np.abs(np.imag(index2_resolved))) > 1e-15:
            print(
                "[design] Warning: lossy design endpoints are resolved for dε/dρ, "
                "but imported design geometry currently uses real(n) only."
            )

        self.index1_complex = index1_resolved
        self.index2_complex = index2_resolved
        self.index1 = [float(np.real(index1_resolved[0])), float(np.real(index1_resolved[1])), float(np.real(index1_resolved[2]))]
        self.index2 = [float(np.real(index2_resolved[0])), float(np.real(index2_resolved[1])), float(np.real(index2_resolved[2]))]

        if hasattr(self, "src_wl") and self.src_wl is not None:
            src_wl_um = np.asarray(self.src_wl, dtype=float).reshape(-1) / self.unit
        else:
            src_wl_um = np.array([wl], dtype=float)
        self.index1_spectrum = np.zeros((src_wl_um.size, 3), dtype=np.complex128)
        self.index2_spectrum = np.zeros_like(self.index1_spectrum)
        for i, wl_i in enumerate(src_wl_um):
            idx1_i = interpolate_index(index1, float(wl_i))
            idx2_i = interpolate_index(index2, float(wl_i))
            if idx1_i.size == 1:
                idx1_i = np.repeat(idx1_i, 3)
            if idx2_i.size == 1:
                idx2_i = np.repeat(idx2_i, 3)
            self.index1_spectrum[i] = idx1_i
            self.index2_spectrum[i] = idx2_i

        if grid_steps is None:
            self.design_dx = float(self.sim_grid)
            self.design_dy = float(self.sim_grid)
            self.design_dz = float(self.sim_grid)
        else:
            self.design_dx = float(grid_steps[0]) if grid_steps[0] else float(self.sim_grid)
            self.design_dy = float(grid_steps[1]) if grid_steps[1] else float(self.sim_grid)
            self.design_dz = float(grid_steps[2]) if grid_steps[2] else float(self.sim_grid)

        center_scaled, _ = self.unit_scaling(center=center, size=[0, 0, 0])
        cx, cy, cz = self._snap_to_mesh(center_scaled)

        self.design_center = [cx, cy, cz]

        # node axis
        self.design_x = self._node_axis(cx, Nx, self.design_dx)
        self.design_y = self._node_axis(cy, Ny, self.design_dy)
        self.design_z = self._node_axis(cz, Nz, self.design_dz)

        self.design_size = [
            float(np.max(self.design_x) - np.min(self.design_x)) if Nx > 1 else 0.0,
            float(np.max(self.design_y) - np.min(self.design_y)) if Ny > 1 else 0.0,
            float(np.max(self.design_z) - np.min(self.design_z)) if Nz > 1 else 0.0,
        ]

        self._configure_design_region_objects()

        if density is None:
            density = 0.5 * np.ones(Nx * Ny * Nz, dtype=float)

        self.update_design_density(density)

    def update_design_density(self, density):
        Nx, Ny, Nz = self.design_grids

        density = np.asarray(density, dtype=float).copy()

        if density.ndim != 3:
            if density.size != Nx * Ny * Nz:
                raise ValueError(f"density size mismatch: got {density.size}, expected {Nx*Ny*Nz}")
            density = density.reshape(Nx, Ny, Nz)
        elif density.shape != (Nx, Ny, Nz):
            raise ValueError(f"density shape mismatch: got {density.shape}, expected {(Nx, Ny, Nz)}")
        self.design_density = density.copy()

        rho1, rho2, rho3 = self.density2idx(density)
        rho1 = np.asarray(rho1, dtype=float).reshape(Nx, Ny, Nz)
        rho2 = np.asarray(rho2, dtype=float).reshape(Nx, Ny, Nz)
        rho3 = np.asarray(rho3, dtype=float).reshape(Nx, Ny, Nz)

        n_ani = np.ones((Nx, Ny, Nz, 3), dtype=float)
        n_ani[:, :, :, 0] = np.clip(rho1, 1.0, 100.0)
        n_ani[:, :, :, 1] = np.clip(rho2, 1.0, 100.0)
        n_ani[:, :, :, 2] = np.clip(rho3, 1.0, 100.0)

        self.design_n = n_ani.copy()

        # importnk2 requires the z axis to have >= 2 layers even when the design
        # (and its monitors) are a single 2D z-plane.  Duplicate the single
        # design layer into a thin 2-layer slab centred on the design z-plane
        # for the IMPORT ONLY; design_grids / design_z / monitors stay Nz=1.
        z_geo_imp = np.asarray(self.design_z, dtype=float)
        n_geo_imp = np.ascontiguousarray(n_ani)
        if z_geo_imp.size == 1:
            _dz = float(self.design_dz) if float(self.design_dz) > 0 else float(self.sim_grid)
            z_geo_imp = np.array([z_geo_imp[0] - 0.5 * _dz,
                                  z_geo_imp[0] + 0.5 * _dz], dtype=float)
            n_geo_imp = np.ascontiguousarray(np.repeat(n_ani, 2, axis=2))

        self.fdtd.putv("x_geo", np.asarray(self.design_x, dtype=float))
        self.fdtd.putv("y_geo", np.asarray(self.design_y, dtype=float))
        self.fdtd.putv("z_geo", z_geo_imp)
        self.fdtd.putv("n_geo", n_geo_imp)

        if self.fdtd.getnamednumber(self.design_name) == 0:
            script = (
                f'addimport;'
                f'set("name","{self.design_name}");'
                f'importnk2(n_geo, x_geo, y_geo, z_geo);'
            )
            self.fdtd.eval(script)
        else:
            try:
                self.fdtd.eval(
                    f'select("{self.design_name}");'
                    f'importnk2(n_geo, x_geo, y_geo, z_geo);'
                )
            except Exception:
                # Fallback for cases where the import object cannot be updated in place.
                self.fdtd.eval(
                    f'select("{self.design_name}");'
                    f'delete;'
                    f'addimport;'
                    f'set("name","{self.design_name}");'
                    f'importnk2(n_geo, x_geo, y_geo, z_geo);'
                )

    def add_design_monitor(self):
        self.design_monitor_name = self.design_name + "_monitor"
        if not hasattr(self, "design_index_monitor_name"):
            self.design_index_monitor_name = self.design_name + "_index_monitor"
        self._configure_design_region_objects()


    def _configured_session_resource_names(self, resource_type="GPU"):
        names = []
        config_paths = []
        for home_value in (os.environ.get("HOME"), os.environ.get("EIDL_REAL_HOME")):
            if home_value:
                config_paths.append(Path(home_value).expanduser() / ".config" / "Lumerical" / "FDTD Solutions.ini")
        config_paths.append(Path.home() / ".config" / "Lumerical" / "FDTD Solutions.ini")
        config_path = next((path for path in config_paths if path.exists()), None)
        if config_path is None:
            return names
        parser = configparser.RawConfigParser()
        parser.optionxform = str
        try:
            parser.read(config_path)
            xml_text = parser.get("jobmanager", "FDTD_v2", fallback="")
            if not xml_text:
                return names
            root = ET.fromstring(xml_text)
        except Exception as exc:
            print(f"[FDTD] could not read Lumerical resource config: {exc}")
            return names

        requested = str(resource_type).upper()
        for engine in root.findall("engine"):
            name = (engine.findtext("name") or "").strip()
            device_type = (engine.findtext("DeviceType") or "").strip().upper()
            if not name:
                continue
            if requested == "GPU":
                if device_type.startswith("GPU"):
                    names.append(name)
            elif requested == "CPU":
                if device_type == "CPU":
                    names.append(name)
            else:
                names.append(name)
        return names

    def _session_resource_names(self, resource_type="GPU"):
        explicit = os.environ.get("LUMERICAL_SESSION_RESOURCE_NAME", "").strip()
        names = []
        if explicit:
            names.append(explicit)
        names.extend(self._configured_session_resource_names(resource_type))
        if str(resource_type).upper() == "GPU":
            names.extend([
                "Local GPU",
                "local GPU",
                "Local Host",
                "localhost",
                "Localhost",
                "Local Computer",
                "local host",
            ])
        else:
            names.extend([
                "Local Host",
                "localhost",
                "Localhost",
                "Local Computer",
                "local host",
            ])
        unique = []
        for name in names:
            if name and name not in unique:
                unique.append(name)
        return unique

    def _run_session_only(self, solver="FDTD", resource_type="GPU", run_name=""):
        errors = []
        for resource_name in self._session_resource_names(resource_type):
            try:
                print(
                    f"[FDTD] session run: name={run_name}, solver={solver}, "
                    f"resource_type={resource_type}, resource_name={resource_name}"
                )
                self.fdtd.run(solver, resource_type, resource_name)
                return resource_name
            except Exception as exc:
                errors.append(f"{resource_name}: {exc}")
        raise RuntimeError(
            "Lumerical session run failed for all configured resource names. "
            "External solver fallback is disabled. Set LUMERICAL_SESSION_RESOURCE_NAME "
            "to the exact resource name shown in the Lumerical Resource Manager. "
            "Tried: " + " | ".join(errors)
        )

    def _configure_session_resources(self):
        gpu_device = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 0")
        threads = os.environ.get("FDTD_THREADS", "1")
        try:
            self.fdtd.setresource("FDTD", 1, "active", 1)
            self.fdtd.setresource("FDTD", 1, "processes", "1")
            self.fdtd.setresource("FDTD", 1, "threads", str(threads))
        except Exception:
            pass
        try:
            self.fdtd.setresource("FDTD", 2, "active", 1)
            self.fdtd.setresource("FDTD", 2, "processes", "1")
            self.fdtd.setresource("FDTD", 2, "threads", str(threads))
            self.fdtd.setresource("FDTD", 2, "device type", gpu_device)
            self.fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        except Exception:
            pass

    def _run_log_tail(self, run_name, max_chars=3000):
        log_path = Path(f"{run_name}_p0.log")
        if not log_path.exists():
            return ""
        try:
            text = log_path.read_text(errors="replace")
        except Exception:
            return ""
        return text[-max_chars:]

    def run(self,name="fdtd_tutorial",save=True):
        fsp_path = os.path.abspath(f"{name}.fsp")
        self._last_run_fsp_path = fsp_path
        if save:
            self.fdtd.save(fsp_path)
        self.fdtd.switchtolayout()
        self.fdtd.eval("select(\"FDTD\");")

        self._configure_session_resources()

        try:
            dimension = str(self.fdtd.getnamed("FDTD", "dimension"))
        except Exception:
            dimension = ""
        if dimension == "3D":
            retries = max(1, int(os.environ.get("LUMERICAL_SESSION_RUN_RETRIES", "3")))
            retry_delay = float(os.environ.get("LUMERICAL_SESSION_RUN_RETRY_DELAY", "5"))
            last_error = None
            for attempt in range(retries):
                try:
                    self._configure_session_resources()
                    # MSOPT_SESSION_RESOURCE_TYPE (default GPU) lets a diagnostic
                    # run take the CPU engine instead of competing for the single
                    # configured GPU (protects an in-flight production GPU run).
                    _rtype = (os.environ.get("MSOPT_SESSION_RESOURCE_TYPE", "GPU").strip()
                              or "GPU")
                    self._run_session_only("FDTD", _rtype, run_name=name)
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt >= retries - 1:
                        break
                    print(
                        "[FDTD] session run failed; reloading saved project and "
                        f"retrying in {retry_delay:g}s ({attempt + 1}/{retries - 1}). "
                        f"Last error: {exc}"
                    )
                    time.sleep(retry_delay)
                    try:
                        self.fdtd.switchtolayout()
                        self.fdtd.load(fsp_path)
                    except Exception:
                        try:
                            self.fdtd.close()
                        except Exception:
                            pass
                        self.fdtd = _open_lumerical_fdtd()
                        self.fdtd.switchtolayout()
                        self.fdtd.load(fsp_path)
            raise last_error
        else:
            self.fdtd.run()

