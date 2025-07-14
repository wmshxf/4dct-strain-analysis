"""
Segmentation Module for 4D-CT Strain Analysis
Handles lung segmentation and lesion region exclusion
"""

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure, morphology
from config import DEFAULT_PARAMS, MORPHOLOGY_PARAMS


class LungSegmentationProcessor:
    """Class for processing lung segmentation tasks"""
    
    def __init__(self, threshold=None, min_region_area=None, border_margin=None):
        """
        Initialize the segmentation processor
        
        Parameters:
        - threshold: HU threshold for lung segmentation
        - min_region_area: Minimum area for valid lung regions
        - border_margin: Margin for border detection
        """
        self.threshold = threshold or DEFAULT_PARAMS['lung_threshold']
        self.min_region_area = min_region_area or DEFAULT_PARAMS['min_region_area']
        self.border_margin = border_margin or DEFAULT_PARAMS['border_margin']
    
    def segment_lungs(self, image):
        """
        Segment lung regions from CT images
        
        Parameters:
        - image: CT image (numpy array or SimpleITK image)
        
        Returns:
        - segmented_lungs: Segmented lung image
        - lung_mask: Binary lung mask
        """
        if isinstance(image, sitk.Image):
            image_array = sitk.GetArrayFromImage(image)
        else:
            image_array = image.copy()
        
        print(f"Performing lung segmentation, image shape: {image_array.shape}")
        
        lung_mask = np.zeros_like(image_array, dtype=bool)
        segmented_lungs = np.zeros_like(image_array)
        
        # Process slice by slice for better accuracy
        for i in range(image_array.shape[0]):
            slice_img = image_array[i].copy()
            slice_mask = self._segment_lung_slice(slice_img)
            
            lung_mask[i] = slice_mask
            segmented_lungs[i] = slice_img * slice_mask
        
        print(f"Lung segmentation completed, valid voxels: {np.sum(lung_mask)}")
        return segmented_lungs, lung_mask
    
    def _segment_lung_slice(self, slice_img):
        """
        Segment lung regions in a single slice
        
        Parameters:
        - slice_img: 2D CT slice image
        
        Returns:
        - slice_mask: Binary mask for lung regions in this slice
        """
        # Apply threshold segmentation
        binary_image = slice_img < self.threshold
        
        # Label connected regions
        labels = measure.label(binary_image)
        regions = measure.regionprops(labels)
        valid_label_mask = np.zeros_like(binary_image, dtype=bool)
        
        # Filter regions based on area and border criteria
        for props in regions:
            if (props.area > self.min_region_area and 
                not self._is_on_border(props.bbox, binary_image.shape)):
                valid_label_mask = valid_label_mask | (labels == props.label)
        
        # Apply morphological operations for cleanup
        cleaned_mask = morphology.binary_closing(
            valid_label_mask, 
            morphology.disk(MORPHOLOGY_PARAMS['closing_disk_size'])
        )
        filled_mask = ndimage.binary_fill_holes(cleaned_mask)
        
        return filled_mask
    
    def _is_on_border(self, bbox, shape):
        """
        Check if a region is near the image border
        
        Parameters:
        - bbox: Bounding box of the region (min_row, min_col, max_row, max_col)
        - shape: Shape of the image
        
        Returns:
        - is_border: Boolean indicating if region is on border
        """
        min_row, min_col, max_row, max_col = bbox
        return (min_row < self.border_margin or 
                min_col < self.border_margin or
                max_row > shape[0] - self.border_margin or 
                max_col > shape[1] - self.border_margin)


class LesionExclusionProcessor:
    """Class for handling lesion region exclusion"""
    
    def __init__(self, exclusion_margin=None):
        """
        Initialize the lesion exclusion processor
        
        Parameters:
        - exclusion_margin: Margin around lesion to exclude (in mm)
        """
        self.exclusion_margin = exclusion_margin or DEFAULT_PARAMS['exclusion_margin']
    
    def exclude_lesion_region(self, lung_mask, lesion_center, lesion_radius):
        """
        Exclude tumor lesion and surrounding area from lung mask
        
        Parameters:
        - lung_mask: Binary lung mask
        - lesion_center: Lesion center coordinates (z, y, x)
        - lesion_radius: Lesion radius in voxels
        
        Returns:
        - modified_mask: Lung mask with lesion region excluded
        """
        modified_mask = lung_mask.copy()
        
        if lesion_center is None:
            return modified_mask
        
        z_center, y_center, x_center = lesion_center
        total_radius = lesion_radius + self.exclusion_margin
        
        # Create exclusion region
        exclusion_mask = self._create_spherical_exclusion_mask(
            lung_mask.shape, lesion_center, total_radius
        )
        
        # Apply exclusion
        modified_mask[exclusion_mask] = False
        
        excluded_voxels = np.sum(exclusion_mask)
        print(f"Excluded lesion region: center({z_center}, {y_center}, {x_center}), "
              f"radius {lesion_radius} + margin {self.exclusion_margin}, "
              f"excluded voxels: {excluded_voxels}")
        
        return modified_mask
    
    def _create_spherical_exclusion_mask(self, shape, center, radius):
        """
        Create a spherical exclusion mask
        
        Parameters:
        - shape: Shape of the volume
        - center: Center coordinates (z, y, x)
        - radius: Radius of exclusion sphere
        
        Returns:
        - exclusion_mask: Boolean array indicating exclusion region
        """
        z_center, y_center, x_center = center
        z_indices, y_indices, x_indices = np.ogrid[:shape[0], :shape[1], :shape[2]]
        
        # Calculate distance to lesion center
        distance = np.sqrt((z_indices - z_center)**2 + 
                          (y_indices - y_center)**2 + 
                          (x_indices - x_center)**2)
        
        return distance <= radius


# Convenience functions for backward compatibility
def segment_lungs(image, threshold=-400):
    """
    Convenience function for lung segmentation
    
    Parameters:
    - image: CT image (numpy array or SimpleITK image)
    - threshold: Lung segmentation threshold (HU value)
    
    Returns:
    - segmented_lungs: Segmented lung image
    - lung_mask: Binary lung mask
    """
    processor = LungSegmentationProcessor(threshold=threshold)
    return processor.segment_lungs(image)


def exclude_lesion_region(lung_mask, lesion_center, lesion_radius, exclusion_margin=10):
    """
    Convenience function for lesion exclusion
    
    Parameters:
    - lung_mask: Binary lung mask
    - lesion_center: Lesion center coordinates (z, y, x)
    - lesion_radius: Lesion radius in voxels
    - exclusion_margin: Exclusion margin in mm
    
    Returns:
    - modified_mask: Lung mask with lesion region excluded
    """
    processor = LesionExclusionProcessor(exclusion_margin=exclusion_margin)
    return processor.exclude_lesion_region(lung_mask, lesion_center, lesion_radius)
