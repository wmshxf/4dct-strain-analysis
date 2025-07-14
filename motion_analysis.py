"""
Motion Analysis Module for 4D-CT Strain Analysis
Handles displacement field calculation and strain parameter analysis
"""

import numpy as np
import cv2
from config import DEFAULT_PARAMS, OPTICAL_FLOW_PARAMS


class OpticalFlowProcessor:
    """Class for calculating displacement fields using optical flow"""
    
    def __init__(self, downsample_factor=None):
        """
        Initialize the optical flow processor
        
        Parameters:
        - downsample_factor: Downsampling factor for computational efficiency
        """
        self.downsample_factor = downsample_factor or DEFAULT_PARAMS['optical_flow_downsample']
    
    def calculate_optical_flow(self, fixed_image, moving_image):
        """
        Calculate displacement field using Farneback optical flow algorithm
        
        Parameters:
        - fixed_image: Reference image
        - moving_image: Moving image
        
        Returns:
        - displacement_field: 3D displacement field (z, y, x, 3)
        """
        print("Computing optical flow displacement field...")
        
        # Downsample for computational efficiency
        fixed_ds = fixed_image[::self.downsample_factor, 
                               ::self.downsample_factor, 
                               ::self.downsample_factor]
        moving_ds = moving_image[::self.downsample_factor, 
                                ::self.downsample_factor, 
                                ::self.downsample_factor]
        
        shape = fixed_ds.shape
        displacement_field = np.zeros((*shape, 3))
        
        # Calculate optical flow slice by slice
        for i in range(shape[0]):
            flow_2d = self._calculate_slice_optical_flow(
                fixed_ds[i], moving_ds[i]
            )
            
            # Store x and y direction displacements
            displacement_field[i, :, :, 0] = flow_2d[:, :, 0]
            displacement_field[i, :, :, 1] = flow_2d[:, :, 1]
        
        # Upsample back to original size
        full_displacement = self._upsample_displacement_field(
            displacement_field, fixed_image.shape
        )
        
        print(f"Displacement field calculation completed, shape: {full_displacement.shape}")
        return full_displacement
    
    def _calculate_slice_optical_flow(self, fixed_slice, moving_slice):
        """
        Calculate optical flow for a single slice
        
        Parameters:
        - fixed_slice: Reference image slice
        - moving_slice: Moving image slice
        
        Returns:
        - flow: 2D optical flow field
        """
        # Normalize slices
        fixed_slice = fixed_slice.astype(np.float32)
        moving_slice = moving_slice.astype(np.float32)
        
        # Normalize to 0-255 range for OpenCV
        fixed_norm = cv2.normalize(fixed_slice, None, 0, 255, cv2.NORM_MINMAX)
        moving_norm = cv2.normalize(moving_slice, None, 0, 255, cv2.NORM_MINMAX)
        
        # Calculate optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            fixed_norm, moving_norm, None,
            OPTICAL_FLOW_PARAMS['pyr_scale'],
            OPTICAL_FLOW_PARAMS['levels'],
            OPTICAL_FLOW_PARAMS['winsize'],
            OPTICAL_FLOW_PARAMS['iterations'],
            OPTICAL_FLOW_PARAMS['poly_n'],
            OPTICAL_FLOW_PARAMS['poly_sigma'],
            OPTICAL_FLOW_PARAMS['flags']
        )
        
        return flow
    
    def _upsample_displacement_field(self, displacement_field, target_shape):
        """
        Upsample displacement field to original image size
        
        Parameters:
        - displacement_field: Downsampled displacement field
        - target_shape: Target shape for upsampling
        
        Returns:
        - full_displacement: Upsampled displacement field
        """
        full_displacement = np.zeros((*target_shape, 3))
        
        for dim in range(3):
            for i in range(target_shape[0]):
                downsample_idx = i // self.downsample_factor
                if downsample_idx < displacement_field.shape[0]:
                    slice_flow = displacement_field[downsample_idx, :, :, dim]
                    full_slice = cv2.resize(
                        slice_flow, (target_shape[2], target_shape[1]),
                        interpolation=cv2.INTER_LINEAR
                    ) * self.downsample_factor
                    full_displacement[i, :, :, dim] = full_slice
        
        return full_displacement


class StrainAnalyzer:
    """Class for calculating strain parameters from displacement fields"""
    
    def __init__(self, time_interval=None):
        """
        Initialize the strain analyzer
        
        Parameters:
        - time_interval: Time interval between phases (seconds)
        """
        self.time_interval = time_interval or DEFAULT_PARAMS['time_interval']
    
    def calculate_strain_parameters(self, displacement_field, lung_mask):
        """
        Calculate strain parameters from displacement field
        
        Parameters:
        - displacement_field: 3D displacement field
        - lung_mask: Binary lung mask
        
        Returns:
        - strain_params: Dictionary containing PSmax, PSmean, Speedmax
        """
        print("Calculating strain parameters...")
        
        # Extract displacement components
        u = displacement_field[:, :, :, 0]  # x-direction displacement
        v = displacement_field[:, :, :, 1]  # y-direction displacement
        w = displacement_field[:, :, :, 2]  # z-direction displacement
        
        # Calculate strain tensor components
        strain_components = self._calculate_strain_tensor_components(u, v, w)
        
        # Calculate principal strain field
        principal_strains = self._calculate_principal_strain_field(
            strain_components, lung_mask
        )
        
        # Calculate displacement speed
        speed_field = self._calculate_speed_field(u, v, w)
        
        # Extract values within lung region
        lung_principal_strains = principal_strains[lung_mask]
        lung_speeds = speed_field[lung_mask]
        
        # Calculate statistical parameters
        strain_params = self._compile_strain_statistics(
            lung_principal_strains, lung_speeds, principal_strains,
            displacement_field, speed_field
        )
        
        self._print_strain_results(strain_params)
        return strain_params
    
    def _calculate_strain_tensor_components(self, u, v, w):
        """
        Calculate strain tensor components from displacement gradients
        
        Parameters:
        - u, v, w: Displacement components in x, y, z directions
        
        Returns:
        - strain_components: Dictionary of strain tensor components
        """
        # Calculate displacement gradients
        du_dx, du_dy, du_dz = np.gradient(u)
        dv_dx, dv_dy, dv_dz = np.gradient(v)
        dw_dx, dw_dy, dw_dz = np.gradient(w)
        
        # Calculate strain tensor components
        return {
            'epsilon_xx': du_dx,
            'epsilon_yy': dv_dy,
            'epsilon_zz': dw_dz,
            'epsilon_xy': 0.5 * (du_dy + dv_dx),
            'epsilon_xz': 0.5 * (du_dz + dw_dx),
            'epsilon_yz': 0.5 * (dv_dz + dw_dy)
        }
    
    def _calculate_principal_strain_field(self, strain_components, lung_mask):
        """
        Calculate principal strain field
        
        Parameters:
        - strain_components: Dictionary of strain tensor components
        - lung_mask: Binary lung mask
        
        Returns:
        - principal_strains: Principal strain field
        """
        shape = strain_components['epsilon_xx'].shape
        principal_strains = np.zeros(shape)
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if lung_mask[i, j, k]:
                        # Construct strain tensor
                        strain_tensor = np.array([
                            [strain_components['epsilon_xx'][i, j, k], 
                             strain_components['epsilon_xy'][i, j, k], 
                             strain_components['epsilon_xz'][i, j, k]],
                            [strain_components['epsilon_xy'][i, j, k], 
                             strain_components['epsilon_yy'][i, j, k], 
                             strain_components['epsilon_yz'][i, j, k]],
                            [strain_components['epsilon_xz'][i, j, k], 
                             strain_components['epsilon_yz'][i, j, k], 
                             strain_components['epsilon_zz'][i, j, k]]
                        ])
                        
                        # Calculate eigenvalues (principal strains)
                        eigenvals = np.linalg.eigvals(strain_tensor)
                        principal_strains[i, j, k] = np.max(eigenvals)
        
        return principal_strains
    
    def _calculate_speed_field(self, u, v, w):
        """
        Calculate displacement speed field
        
        Parameters:
        - u, v, w: Displacement components
        
        Returns:
        - speed: Speed field
        """
        displacement_magnitude = np.sqrt(u**2 + v**2 + w**2)
        return displacement_magnitude / self.time_interval
    
    def _compile_strain_statistics(self, lung_principal_strains, lung_speeds,
                                  principal_strains, displacement_field, speed_field):
        """
        Compile strain statistics into results dictionary
        
        Parameters:
        - lung_principal_strains: Principal strains within lung region
        - lung_speeds: Speeds within lung region
        - principal_strains: Full principal strain field
        - displacement_field: Full displacement field
        - speed_field: Full speed field
        
        Returns:
        - strain_params: Dictionary containing strain parameters
        """
        PSmax = np.max(lung_principal_strains) if len(lung_principal_strains) > 0 else 0
        PSmean = np.mean(lung_principal_strains) if len(lung_principal_strains) > 0 else 0
        Speedmax = np.max(lung_speeds) if len(lung_speeds) > 0 else 0
        
        return {
            'PSmax': PSmax,
            'PSmean': PSmean,
            'Speedmax': Speedmax,
            'principal_strain_field': principal_strains,
            'displacement_field': displacement_field,
            'speed_field': speed_field
        }
    
    def _print_strain_results(self, strain_params):
        """Print strain calculation results"""
        print(f"Strain parameters calculated:")
        print(f"  PSmax: {strain_params['PSmax']:.6f}")
        print(f"  PSmean: {strain_params['PSmean']:.6f}")
        print(f"  Speedmax: {strain_params['Speedmax']:.4f} mm/s")


# Convenience functions for backward compatibility
def calculate_optical_flow(fixed_image, moving_image, downsample=2):
    """
    Convenience function for calculating optical flow
    
    Parameters:
    - fixed_image: Reference image
    - moving_image: Moving image
    - downsample: Downsampling factor for computational efficiency
    
    Returns:
    - displacement_field: 3D displacement field
    """
    processor = OpticalFlowProcessor(downsample_factor=downsample)
    return processor.calculate_optical_flow(fixed_image, moving_image)


def calculate_strain_parameters(displacement_field, lung_mask):
    """
    Convenience function for calculating strain parameters
    
    Parameters:
    - displacement_field: 3D displacement field
    - lung_mask: Binary lung mask
    
    Returns:
    - strain_params: Dictionary containing strain parameters
    """
    analyzer = StrainAnalyzer()
    return analyzer.calculate_strain_parameters(displacement_field, lung_mask)
