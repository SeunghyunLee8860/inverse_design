"""Periodic fabrication mapping extracted from the local msopt fork."""
import numpy as np
import autograd.numpy as npa
from . import Sub_Mapping

class Mapping:
    def __init__(
        self,
        Symmetry_sim = False,
        Sym_geo_width =False,
        Sym_geo_length=False,
        Sym_geo_C2 = False,
        Sym_geo_C8 = False,
        Sym_offdiag= False,
        Is_waveguide=[False, False, False, 2], # Is_wg?, Is_middle?, Is_3D?, number of local region
        DR_info = [None, None, None, 1, 2, 0], # Dx:0, Dy:1, Dz:2, width-idx, length-idx, hight-idx
        DR_N_info=[None, None, None, None], # Nx, Ny, Nz,  resolution
        Is_Cylindrical=[False, False, 0.0], # Is_cyl?, Is_2D?, Disk_R
        Mask_info=[None, None], #  wg_left mask sz, wg_right mask sz
        Sub_pixels=[False, 0],
        Mask_pixels=0,
        MFS= None,
        MGS= None,
        Is_1D_grating=[False, False], # Is 1D grating?, Is slanted?
        Is_freeform=[False, False, False], # Is Free-form?, Is gray-scale?, Is Single layer?
        Is_slanted_grating=False,
        Is_radial_3d=None,
        periodic_filter_axes=None,  # e.g. [0, 1]: MFS/MGS filtering wraps at the
                                    # width/length boundaries (periodic unit cell);
                                    # None -> legacy Air-pad behavior
    ):
        self.periodic_filter_axes = periodic_filter_axes
        self.MFS = MFS
        self.MFS0 =MFS
        self.MGS = MGS
        self.MGS0 =MGS
        self.DR_res =DR_N_info[3]
        self.Mask_info = Mask_info
        self.Sub_pixels =Sub_pixels
        self.Mask_pixels=Mask_pixels
        self.Sym_geo_C2 = Sym_geo_C2
        self.Sym_geo_C8 = Sym_geo_C8
        self.Sym_offdiag = Sym_offdiag
        self.Is_freeform = Is_freeform
        self.Is_waveguide= Is_waveguide
        self.Symmetry_sim = Symmetry_sim
        self.Sym_geo_width =Sym_geo_width
        self.Sym_geo_length=Sym_geo_length
        self.Is_1D_grating = Is_1D_grating
        self.Is_Cylindrical= Is_Cylindrical
        self.Is_radial_3d = Is_radial_3d
        self.radial_3d_enabled = False
        self.radial_3d_apply_filter = True
        self.radial_3d_radius = None
        self.radial_3d_outside_value = 0.0
        self.N_radius = None
        self.radial_3d_vertical_grating = False
        self.slanted= Is_slanted_grating
        self.N_height=DR_N_info[DR_info[5]]
        self.need_T = True
        if isinstance(Is_radial_3d, dict):
            self.radial_3d_enabled = bool(Is_radial_3d.get("enabled", False))
            self.N_radius = Is_radial_3d.get("N_radius", None)
            self.radial_3d_radius = Is_radial_3d.get("radius", None)
            self.radial_3d_outside_value = Is_radial_3d.get("outside_value", 0.0)
            self.radial_3d_apply_filter = bool(Is_radial_3d.get("apply_filter", True))
            self.radial_3d_vertical_grating = bool(Is_radial_3d.get("vertical_grating", False))
        elif isinstance(Is_radial_3d, bool):
            self.radial_3d_enabled = Is_radial_3d
        elif Is_radial_3d is not None:
            self.radial_3d_enabled = bool(Is_radial_3d[0])
            if len(Is_radial_3d) > 1:
                self.N_radius = Is_radial_3d[1]
            if len(Is_radial_3d) > 2:
                self.radial_3d_radius = Is_radial_3d[2]
            if len(Is_radial_3d) > 3:
                self.radial_3d_outside_value = Is_radial_3d[3]
            if len(Is_radial_3d) > 4:
                self.radial_3d_apply_filter = bool(Is_radial_3d[4])
            if len(Is_radial_3d) > 5:
                self.radial_3d_vertical_grating = bool(Is_radial_3d[5])
        if Is_freeform[0] and not Is_freeform[2] and not self.radial_3d_enabled:
            print("\n This is Freeform topology optimization \n")
            return
        if DR_info[5] == DR_info[4] or DR_info[5] == DR_info[3]:
            raise ValueError("Design layer includes height axis")
        elif DR_info[3] >= DR_info[4] and DR_info[3] !=2:
            raise ValueError("Design layer width axis should be low axis")
        else: # width axis < length axis && layer size = width*length
            if DR_info[5] == 0:
                self.need_T = False
            if Is_waveguide[0]:            
                if Is_waveguide[1]:
                    self.MGS0= MGS*0.5
                    self.MFS0=self.MFS + MGS*0.5
                    self.Mask_info[0] += MGS*0.5 
                    self.Mask_info[1] += MGS*0.5
                    if Is_waveguide[3]/2==int(Is_waveguide[3]/2):
                        print("Reference layer: middle layer")
                    else: # total layer number should be even (mid)
                        self.Is_waveguide[3] = Is_waveguide[3]+1
                        print("Reference layer: middle+1 layer")
                self.DR_width = DR_info[DR_info[3]]
                self.N_width= DR_N_info[DR_info[3]]
                self.DR_length= DR_info[DR_info[4]]
                self.N_length=DR_N_info[DR_info[4]]
            else: # Cylindrical grating use the width axis only
                if not Is_Cylindrical[0]:
                    self.DR_length =DR_info[DR_info[4]]
                    self.N_length=DR_N_info[DR_info[4]]
                self.DR_width = DR_info[DR_info[3]]
                self.N_width =DR_N_info[DR_info[3]]
                self.Mask_info[0]=0
                self.Mask_info[1]=0
        if self.radial_3d_enabled:
            if self.radial_3d_radius is None:
                self.radial_3d_radius = 0.5 * min(self.DR_width, self.DR_length)
            if self.N_radius is None:
                self.N_radius = int(round(self.radial_3d_radius * self.DR_res)) + 1
            self.N_radius = int(self.N_radius)
            if self.radial_3d_vertical_grating:
                self.parameter_shape = (self.N_radius,)
                self.parameter_count = self.N_radius
            else:
                self.parameter_shape = (self.N_radius, self.N_height)
                self.parameter_count = self.N_radius * self.N_height
            self.output_shape = (self.N_width, self.N_length, self.N_height)
            print(
                "Radial 3D mapping: "
                f"rho({'r' if self.radial_3d_vertical_grating else 'r,z'}) "
                f"{self.parameter_shape} -> xyz {self.output_shape}, "
                f"R={self.radial_3d_radius}"
            )
    
    def __call__(
        self,
        x,
        beta,
        Is_opt=True,
    ) -> np.ndarray:
        if Is_opt:  # Filter & Projection based Mapping function
            if self.radial_3d_enabled:
                return Sub_Mapping.radial_cross_section_to_3d(
                    x=x,
                    DR_width=self.DR_width,
                    N_width=self.N_width,
                    DR_length=self.DR_length,
                    N_length=self.N_length,
                    N_height=self.N_height,
                    DR_res=self.DR_res,
                    beta=beta,
                    N_radius=self.N_radius,
                    Radial_R=self.radial_3d_radius,
                    outside_value=self.radial_3d_outside_value,
                    Min_size_top=self.MFS0 if self.MFS0 is not None else 0.0,
                    Min_gap=self.MGS0 if self.MGS0 is not None else 0.0,
                    apply_filter=self.radial_3d_apply_filter,
                    vertical_grating=self.radial_3d_vertical_grating,
                )
            if self.Is_freeform[0]:
                if beta>2:
                    eee=-0.5
                else:
                    eee=0.5
                if self.Is_freeform[1]:
                    if self.Is_freeform[2]:
                        if self.N_height >1:
                            x_copy= Sub_Mapping.Vertical_sidewall(
                                Reference_layer = x,
                                N_height = self.N_height,
                            )
                            x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
                            return x_copy.flatten()
                    else:
                        return x
                elif self.Is_freeform[2]:
                    if self.Sym_geo_C8:
                        x_ref = npa.reshape(x,(self.N_width,self.N_length))
                        x_ref = (npa.fliplr(x_ref) + x_ref)/2
                        x_ref = (npa.flipud(x_ref) + x_ref)/2
                        x_ref = (x_ref.transpose() + x_ref)/2
                        x = x_ref.flatten() 
                    x=Sub_Mapping.tanh_projection_m(x, beta, eee)
                    if self.N_height >1:
                        x_copy= Sub_Mapping.Vertical_sidewall(
                            Reference_layer = x,
                            N_height = self.N_height,
                        )
                        x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
                    return Sub_Mapping.tanh_projection_m(x_copy, beta, 0.5).flatten()
                else:
                    xx=Sub_Mapping.tanh_projection_m(x, beta, eee)
                    return Sub_Mapping.tanh_projection_m(xx, beta, 0.5).flatten()
            else:
                if self.Is_Cylindrical[0]: # 1D Grating structure
                    x_copy= Sub_Mapping.Grating(
                        Disk_R= self.Is_Cylindrical[2],
                        Is_Cyl= self.Is_Cylindrical[1],
                        Mask_pixels=self.Mask_pixels,
                        Min_size_top = self.MFS0,
                        DR_width= self.DR_width,
                        N_width= self.N_width,
                        DR_res= self.DR_res,
                        Min_gap=self.MGS0,
                        beta= beta,
                        x = x,
                    )
                elif self.Is_1D_grating[0]:
                    peudo_width= round(2*(self.MGS + self.MFS),1)
                    peudo_N_width= int(self.DR_res * peudo_width)+1
                    x_copy= Sub_Mapping.get_reference_layer_1D(
                        is_slanted=self.Is_1D_grating[1],
                        Min_size_top = self.MFS,
                        DR_length= self.DR_length,
                        N_length= self.N_length,
                        peudo_width= peudo_width,
                        peudo_N_width= peudo_N_width,
                        DR_res= self.DR_res,
                        Min_gap=self.MGS,
                        beta= beta,
                        x = x,
                    )
                    if self.Is_1D_grating[1]:
                        x_copy= Sub_Mapping.Slant_sidewall(
                            Number_of_local_region=self.Is_waveguide[3],
                            is_Middle = self.Is_waveguide[1],
                            Reference_layer = x_copy,
                            N_height = self.N_height,
                            DR_width = peudo_width,
                            DR_length=self.DR_length,
                            DR_res = self.DR_res,
                            Min_gap= self.MGS,
                            beta = beta,
                        )
                        x_copy = npa.reshape(x_copy,(self.N_height,peudo_N_width,self.N_length))
                        x_copy = x_copy[:,int(peudo_N_width/2),:].flatten()
                        x_copy = npa.reshape(x_copy,(self.N_height,self.N_length)).transpose()
                else:   # 2D Grating structure / 3D structure
                    x_copy = Sub_Mapping.get_reference_layer(
                        Symmetry_in_Sim = self.Symmetry_sim,
                        Width_symmetry = self.Sym_geo_width,
                        Length_symmetry=self.Sym_geo_length,
                        C2_symmetry = self.Sym_geo_C2,
                        QR_symmetry= self.Sym_offdiag,
                        Pseudo_Cyl= self.Sym_geo_C8,
                        DR_length = self.DR_length,
                        DR_width = self.DR_width,
                        N_length= self.N_length,
                        N_width =self.N_width,
                        DR_res =self.DR_res,
                        Is_waveguide=self.Is_waveguide[0],
                        Mask_pixels= self.Mask_pixels,
                        input_w_top=self.Mask_info[0],
                        Sub_pixels= self.Sub_pixels,
                        w_top = self.Mask_info[1],
                        Min_size_top=self.MFS0,
                        Min_gap=self.MGS0,
                        periodic_axes=self.periodic_filter_axes,
                        beta= beta,
                        x = x,
                    )
                    if self.Is_waveguide[2]:# Slanted Waveguide
                        x_copy= Sub_Mapping.Slant_sidewall(
                            Number_of_local_region=self.Is_waveguide[3],
                            is_Middle = self.Is_waveguide[1],
                            Reference_layer = x_copy,
                            N_height = self.N_height,
                            DR_width = self.DR_width,
                            DR_length=self.DR_length,
                            DR_res = self.DR_res,
                            Min_gap= self.MGS,
                            beta = beta,
                        )
                    elif self.N_height >1:
                        if self.slanted:
                            x_copy= Sub_Mapping.Slant_sidewall(
                                Number_of_local_region=2,
                                is_Middle = False,
                                Reference_layer = x_copy,
                                N_height = self.N_height,
                                DR_width = self.DR_width,
                                DR_length=self.DR_length,
                                DR_res = self.DR_res,
                                Min_gap= self.MGS,
                                beta = beta,
                            )
                        else:
                            x_copy= Sub_Mapping.Vertical_sidewall(
                                Reference_layer = x_copy,
                                N_height = self.N_height,
                            )
                        if self.need_T:
                            x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
        else: # 2D -> 3D projection of optimized reference layer
            if self.radial_3d_enabled:
                return Sub_Mapping.radial_cross_section_to_3d(
                    x=x,
                    DR_width=self.DR_width,
                    N_width=self.N_width,
                    DR_length=self.DR_length,
                    N_length=self.N_length,
                    N_height=self.N_height,
                    DR_res=self.DR_res,
                    beta=beta,
                    N_radius=self.N_radius,
                    Radial_R=self.radial_3d_radius,
                    outside_value=self.radial_3d_outside_value,
                    Min_size_top=self.MFS0 if self.MFS0 is not None else 0.0,
                    Min_gap=self.MGS0 if self.MGS0 is not None else 0.0,
                    apply_filter=self.radial_3d_apply_filter,
                    vertical_grating=self.radial_3d_vertical_grating,
                )
            if self.Is_waveguide[2]: # Is Slanted 3D Waveguide?
                x_copy= Sub_Mapping.Slant_sidewall(
                    Number_of_local_region=self.Is_waveguide[3],
                    is_Middle= self.Is_waveguide[1],
                    Reference_layer = x,
                    N_height = self.N_height,
                    DR_width = self.DR_width,
                    DR_length=self.DR_length,
                    DR_res= self.DR_res,
                    Min_gap= self.MGS,
                    beta = beta,
                )
        return x_copy.flatten()
