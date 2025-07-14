"""
Visualization Module for 4D-CT Strain Analysis
Handles motion vector and strain field visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import DEFAULT_PARAMS, configure_matplotlib


class MotionVectorVisualizer:
    """Class for visualizing motion vectors"""
    
    def __init__(self, arrow_density=None, arrow_scale=None):
        """
        Initialize the motion vector visualizer
        
        Parameters:
        - arrow_density: Density of arrows in the visualization
        - arrow_scale: Scaling factor for arrow size
        """
        self.arrow_density = arrow_density or DEFAULT_PARAMS['arrow_density']
        self.arrow_scale = arrow_scale or DEFAULT_PARAMS['arrow_scale']
        configure_matplotlib()
    
    def visualize_motion_vectors(self, image, displacement_field, lung_mask, slice_index=None):
        """
        Visualize motion vectors across different anatomical views
        
        Parameters:
        - image: CT image
        - displacement_field: Displacement field
        - lung_mask: Binary lung mask
        - slice_index: Slice index for display (if None, uses middle slice)
        
        Returns:
        - fig: Matplotlib figure object
        """
        if slice_index is None:
            slice_index = image.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        views = ['Axial', 'Coronal', 'Sagittal']
        slice_indices = [slice_index, image.shape[1]//2, image.shape[2]//2]
        
        for idx, (view, s_idx) in enumerate(zip(views, slice_indices)):
            self._create_motion_vector_view(
                axes[idx], image, displacement_field, lung_mask, view, s_idx
            )
        
        plt.tight_layout()
        return fig
    
    def _create_motion_vector_view(self, ax, image, displacement_field, lung_mask, view, s_idx):
        """
        Create motion vector visualization for a specific anatomical view
        
        Parameters:
        - ax: Matplotlib axis object
        - image: CT image
        - displacement_field: Displacement field
        - lung_mask: Binary lung mask
        - view: Anatomical view ('Axial', 'Coronal', 'Sagittal')
        - s_idx: Slice index
        """
        # Extract appropriate slice data based on view
        img_slice, mask_slice, u, v = self._extract_view_data(
            image, displacement_field, lung_mask, view, s_idx
        )
        
        # Display CT image as background
        ax.imshow(img_slice, cmap='gray', alpha=0.8)
        
        # Create and display motion vectors
        quiver = self._create_motion_vectors(ax, img_slice, mask_slice, u, v)
        
        # Configure plot appearance
        ax.set_title(f'{view} View (Slice {s_idx})')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(quiver, cax=cax, label='Displacement Magnitude (mm)')
    
    def _extract_view_data(self, image, displacement_field, lung_mask, view, s_idx):
        """
        Extract slice data for specific anatomical view
        
        Parameters:
        - image: CT image
        - displacement_field: Displacement field
        - lung_mask: Binary lung mask
        - view: Anatomical view
        - s_idx: Slice index
        
        Returns:
        - img_slice: Image slice
        - mask_slice: Mask slice
        - u, v: Displacement components for the view
        """
        if view == 'Axial':
            img_slice = image[s_idx]
            mask_slice = lung_mask[s_idx]
            u = displacement_field[s_idx, :, :, 0]
            v = displacement_field[s_idx, :, :, 1]
        elif view == 'Coronal':
            img_slice = image[:, s_idx, :]
            mask_slice = lung_mask[:, s_idx, :]
            u = displacement_field[:, s_idx, :, 0]
            v = displacement_field[:, s_idx, :, 2]
        elif view == 'Sagittal':
            img_slice = image[:, :, s_idx]
            mask_slice = lung_mask[:, :, s_idx]
            u = displacement_field[:, :, s_idx, 1]
            v = displacement_field[:, :, s_idx, 2]
        
        return img_slice, mask_slice, u, v
    
    def _create_motion_vectors(self, ax, img_slice, mask_slice, u, v):
        """
        Create motion vector arrows on the plot
        
        Parameters:
        - ax: Matplotlib axis
        - img_slice: Image slice
        - mask_slice: Mask slice
        - u, v: Displacement components
        
        Returns:
        - quiver: Matplotlib quiver object
        """
        # Create arrow grid
        y, x = np.mgrid[0:img_slice.shape[0]:self.arrow_density, 
                        0:img_slice.shape[1]:self.arrow_density]
        
        # Sample displacement field at arrow positions
        U, V, C = self._sample_displacement_field(x, y, mask_slice, u, v)
        
        # Create quiver plot
        quiver = ax.quiver(x, y, U, V, C, cmap='jet', scale=self.arrow_scale)
        
        return quiver
    
    def _sample_displacement_field(self, x, y, mask_slice, u, v):
        """
        Sample displacement field at arrow positions
        
        Parameters:
        - x, y: Arrow position grids
        - mask_slice: Mask slice
        - u, v: Displacement components
        
        Returns:
        - U, V: Displacement components at arrow positions
        - C: Color values (displacement magnitude)
        """
        U = np.zeros_like(x, dtype=float)
        V = np.zeros_like(y, dtype=float)
        C = np.zeros_like(x, dtype=float)
        
        for j in range(y.shape[0]):
            for k in range(x.shape[1]):
                yy, xx = y[j, k], x[j, k]
                if mask_slice[yy, xx]:
                    U[j, k] = u[yy, xx]
                    V[j, k] = v[yy, xx]
                    C[j, k] = np.sqrt(U[j, k]**2 + V[j, k]**2)
        
        return U, V, C


class StrainFieldVisualizer:
    """Class for visualizing strain fields"""
    
    def __init__(self):
        """Initialize the strain field visualizer"""
        configure_matplotlib()
    
    def visualize_strain_field(self, image, strain_field, lung_mask, slice_index=None):
        """
        Visualize strain field across different anatomical views
        
        Parameters:
        - image: CT image
        - strain_field: Principal strain field
        - lung_mask: Binary lung mask
        - slice_index: Slice index for display (if None, uses middle slice)
        
        Returns:
        - fig: Matplotlib figure object
        """
        if slice_index is None:
            slice_index = image.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        views = ['Axial', 'Coronal', 'Sagittal']
        slice_indices = [slice_index, image.shape[1]//2, image.shape[2]//2]
        
        # Calculate strain field statistics for consistent color scaling
        strain_max = np.percentile(strain_field[lung_mask], 95)
        
        for idx, (view, s_idx) in enumerate(zip(views, slice_indices)):
            self._create_strain_field_view(
                axes[idx], image, strain_field, lung_mask, view, s_idx, strain_max
            )
        
        plt.tight_layout()
        return fig
    
    def _create_strain_field_view(self, ax, image, strain_field, lung_mask, view, s_idx, strain_max):
        """
        Create strain field visualization for a specific anatomical view
        
        Parameters:
        - ax: Matplotlib axis object
        - image: CT image
        - strain_field: Principal strain field
        - lung_mask: Binary lung mask
        - view: Anatomical view
        - s_idx: Slice index
        - strain_max: Maximum strain value for color scaling
        """
        # Extract slice data based on view
        img_slice, mask_slice, strain_slice = self._extract_strain_view_data(
            image, strain_field, lung_mask, view, s_idx
        )
        
        # Display CT image as background
        ax.imshow(img_slice, cmap='gray', alpha=0.7)
        
        # Mask strain field to show only lung regions
        masked_strain = np.ma.masked_where(mask_slice == 0, strain_slice)
        
        # Display strain field overlay
        im = ax.imshow(masked_strain, cmap='jet', alpha=0.8, vmin=0, vmax=strain_max)
        
        # Configure plot appearance
        ax.set_title(f'Principal Strain - {view} (Slice {s_idx})')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax, label='Principal Strain')
    
    def _extract_strain_view_data(self, image, strain_field, lung_mask, view, s_idx):
        """
        Extract strain field data for specific anatomical view
        
        Parameters:
        - image: CT image
        - strain_field: Principal strain field
        - lung_mask: Binary lung mask
        - view: Anatomical view
        - s_idx: Slice index
        
        Returns:
        - img_slice: Image slice
        - mask_slice: Mask slice
        - strain_slice: Strain field slice
        """
        if view == 'Axial':
            img_slice = image[s_idx]
            mask_slice = lung_mask[s_idx]
            strain_slice = strain_field[s_idx]
        elif view == 'Coronal':
            img_slice = image[:, s_idx, :]
            mask_slice = lung_mask[:, s_idx, :]
            strain_slice = strain_field[:, s_idx, :]
        elif view == 'Sagittal':
            img_slice = image[:, :, s_idx]
            mask_slice = lung_mask[:, :, s_idx]
            strain_slice = strain_field[:, :, s_idx]
        
        return img_slice, mask_slice, strain_slice


# Convenience functions for backward compatibility
def visualize_motion_vectors(image, displacement_field, lung_mask, slice_index=None, 
                           arrow_density=5, arrow_scale=50):
    """
    Convenience function for visualizing motion vectors
    
    Parameters:
    - image: CT image
    - displacement_field: Displacement field
    - lung_mask: Binary lung mask
    - slice_index: Slice index for display
    - arrow_density: Arrow density
    - arrow_scale: Arrow scaling factor
    
    Returns:
    - fig: Matplotlib figure object
    """
    visualizer = MotionVectorVisualizer(arrow_density=arrow_density, arrow_scale=arrow_scale)
    return visualizer.visualize_motion_vectors(image, displacement_field, lung_mask, slice_index)


def visualize_strain_field(image, strain_field, lung_mask, slice_index=None):
    """
    Convenience function for visualizing strain field
    
    Parameters:
    - image: CT image
    - strain_field: Principal strain field
    - lung_mask: Binary lung mask
    - slice_index: Slice index for display
    
    Returns:
    - fig: Matplotlib figure object
    """
    visualizer = StrainFieldVisualizer()
    return visualizer.visualize_strain_field(image, strain_field, lung_mask, slice_index)
