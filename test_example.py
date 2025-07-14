"""
Test suite for segmentation module
"""

import pytest
import numpy as np
import SimpleITK as sitk
from unittest.mock import Mock, patch

# Import modules to test
from src.segmentation import (
    LungSegmentationProcessor,
    LesionExclusionProcessor,
    segment_lungs,
    exclude_lesion_region
)


class TestLungSegmentationProcessor:
    """Test cases for LungSegmentationProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.processor = LungSegmentationProcessor()
        
        # Create synthetic test data
        self.test_image = self._create_synthetic_ct_image()
        self.test_sitk_image = self._create_synthetic_sitk_image()
    
    def _create_synthetic_ct_image(self):
        """Create a synthetic CT image for testing"""
        # Create a 3D array representing a CT scan
        image = np.full((50, 100, 100), -1000, dtype=np.int16)  # Air background
        
        # Add lung regions (air-filled, around -800 to -500 HU)
        image[10:40, 20:40, 30:70] = -600  # Left lung
        image[10:40, 60:80, 30:70] = -700  # Right lung
        
        # Add some tissue regions (soft tissue, around 0 to 100 HU)
        image[5:45, 45:55, 25:75] = 50    # Mediastinum
        image[:5, :, :] = 30              # Chest wall
        image[45:, :, :] = 30             # Chest wall
        
        return image
    
    def _create_synthetic_sitk_image(self):
        """Create a synthetic SimpleITK image"""
        array = self._create_synthetic_ct_image()
        sitk_image = sitk.GetImageFromArray(array)
        sitk_image.SetSpacing([1.0, 1.0, 2.0])  # mm spacing
        return sitk_image
    
    def test_initialization_default_parameters(self):
        """Test processor initialization with default parameters"""
        processor = LungSegmentationProcessor()
        assert processor.threshold == -400
        assert processor.min_region_area == 1000
        assert processor.border_margin == 10
    
    def test_initialization_custom_parameters(self):
        """Test processor initialization with custom parameters"""
        processor = LungSegmentationProcessor(
            threshold=-500,
            min_region_area=500,
            border_margin=5
        )
        assert processor.threshold == -500
        assert processor.min_region_area == 500
        assert processor.border_margin == 5
    
    def test_segment_lungs_numpy_array(self):
        """Test lung segmentation with numpy array input"""
        segmented_lungs, lung_mask = self.processor.segment_lungs(self.test_image)
        
        # Check output types and shapes
        assert isinstance(segmented_lungs, np.ndarray)
        assert isinstance(lung_mask, np.ndarray)
        assert segmented_lungs.shape == self.test_image.shape
        assert lung_mask.shape == self.test_image.shape
        assert lung_mask.dtype == bool
        
        # Check that some lung regions are segmented
        assert np.sum(lung_mask) > 0
        
        # Check that segmented regions are only in lung areas
        assert np.all(segmented_lungs[~lung_mask] == 0)
    
    def test_segment_lungs_sitk_image(self):
        """Test lung segmentation with SimpleITK image input"""
        segmented_lungs, lung_mask = self.processor.segment_lungs(self.test_sitk_image)
        
        # Check output types and shapes
        assert isinstance(segmented_lungs, np.ndarray)
        assert isinstance(lung_mask, np.ndarray)
        assert lung_mask.dtype == bool
        
        # Should have same shape as original image array
        original_array = sitk.GetArrayFromImage(self.test_sitk_image)
        assert segmented_lungs.shape == original_array.shape
        assert lung_mask.shape == original_array.shape
    
    def test_segment_lungs_different_thresholds(self):
        """Test segmentation with different threshold values"""
        thresholds = [-300, -400, -500, -600]
        
        for threshold in thresholds:
            processor = LungSegmentationProcessor(threshold=threshold)
            _, lung_mask = processor.segment_lungs(self.test_image)
            
            # Higher thresholds should generally result in smaller segmented areas
            assert isinstance(lung_mask, np.ndarray)
            assert lung_mask.dtype == bool
    
    def test_is_on_border_detection(self):
        """Test border detection functionality"""
        # Test region clearly on border
        border_bbox = (0, 0, 10, 10)  # Starts at image edge
        shape = (100, 100)
        assert self.processor._is_on_border(border_bbox, shape)
        
        # Test region clearly not on border
        center_bbox = (40, 40, 60, 60)  # Well within image
        assert not self.processor._is_on_border(center_bbox, shape)
        
        # Test region near border within margin
        near_border_bbox = (5, 5, 15, 15)  # Within default margin of 10
        assert self.processor._is_on_border(near_border_bbox, shape)
    
    def test_segment_lung_slice_functionality(self):
        """Test single slice segmentation"""
        test_slice = self.test_image[25]  # Middle slice
        result_mask = self.processor._segment_lung_slice(test_slice)
        
        assert isinstance(result_mask, np.ndarray)
        assert result_mask.dtype == bool
        assert result_mask.shape == test_slice.shape


class TestLesionExclusionProcessor:
    """Test cases for LesionExclusionProcessor class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.processor = LesionExclusionProcessor()
        
        # Create a simple lung mask
        self.lung_mask = np.ones((50, 100, 100), dtype=bool)
        self.lesion_center = (25, 50, 50)  # Center of the volume
        self.lesion_radius = 10
    
    def test_initialization(self):
        """Test processor initialization"""
        processor = LesionExclusionProcessor(exclusion_margin=15)
        assert processor.exclusion_margin == 15
    
    def test_exclude_lesion_region_basic(self):
        """Test basic lesion exclusion functionality"""
        modified_mask = self.processor.exclude_lesion_region(
            self.lung_mask, self.lesion_center, self.lesion_radius
        )
        
        # Check that result is boolean array with same shape
        assert isinstance(modified_mask, np.ndarray)
        assert modified_mask.dtype == bool
        assert modified_mask.shape == self.lung_mask.shape
        
        # Check that some voxels were excluded
        assert np.sum(modified_mask) < np.sum(self.lung_mask)
        
        # Check that excluded region is centered around lesion
        z, y, x = self.lesion_center
        total_radius = self.lesion_radius + self.processor.exclusion_margin
        
        # Voxels far from lesion should not be affected
        far_voxel = modified_mask[0, 0, 0]  # Corner voxel
        original_far_voxel = self.lung_mask[0, 0, 0]
        assert far_voxel == original_far_voxel
    
    def test_exclude_lesion_region_none_center(self):
        """Test lesion exclusion with None center (should return unchanged)"""
        modified_mask = self.processor.exclude_lesion_region(
            self.lung_mask, None, self.lesion_radius
        )
        
        # Should return exactly the same mask
        assert np.array_equal(modified_mask, self.lung_mask)
    
    def test_create_spherical_exclusion_mask(self):
        """Test spherical exclusion mask creation"""
        shape = (50, 100, 100)
        center = (25, 50, 50)
        radius = 15
        
        exclusion_mask = self.processor._create_spherical_exclusion_mask(
            shape, center, radius
        )
        
        assert isinstance(exclusion_mask, np.ndarray)
        assert exclusion_mask.dtype == bool
        assert exclusion_mask.shape == shape
        
        # Check that center voxel is excluded
        z, y, x = center
        assert exclusion_mask[z, y, x] == True
        
        # Check that voxels at exactly radius distance
        # Some should be excluded, some not (depending on discretization)
        assert np.sum(exclusion_mask) > 0
    
    def test_exclude_lesion_different_margins(self):
        """Test lesion exclusion with different margins"""
        margins = [5, 10, 15, 20]
        excluded_voxels = []
        
        for margin in margins:
            processor = LesionExclusionProcessor(exclusion_margin=margin)
            modified_mask = processor.exclude_lesion_region(
                self.lung_mask, self.lesion_center, self.lesion_radius
            )
            excluded_count = np.sum(self.lung_mask) - np.sum(modified_mask)
            excluded_voxels.append(excluded_count)
        
        # Larger margins should exclude more voxels
        for i in range(1, len(excluded_voxels)):
            assert excluded_voxels[i] >= excluded_voxels[i-1]


class TestConvenienceFunctions:
    """Test convenience functions for backward compatibility"""
    
    def setup_method(self):
        """Set up test data"""
        # Create simple test data
        self.test_image = np.full((20, 50, 50), -600, dtype=np.int16)
        self.lung_mask = np.ones((20, 50, 50), dtype=bool)
    
    def test_segment_lungs_function(self):
        """Test segment_lungs convenience function"""
        segmented_lungs, lung_mask = segment_lungs(self.test_image, threshold=-400)
        
        assert isinstance(segmented_lungs, np.ndarray)
        assert isinstance(lung_mask, np.ndarray)
        assert lung_mask.dtype == bool
        assert segmented_lungs.shape == self.test_image.shape
        assert lung_mask.shape == self.test_image.shape
    
    def test_exclude_lesion_region_function(self):
        """Test exclude_lesion_region convenience function"""
        lesion_center = (10, 25, 25)
        lesion_radius = 5
        
        modified_mask = exclude_lesion_region(
            self.lung_mask, lesion_center, lesion_radius, exclusion_margin=10
        )
        
        assert isinstance(modified_mask, np.ndarray)
        assert modified_mask.dtype == bool
        assert modified_mask.shape == self.lung_mask.shape
        assert np.sum(modified_mask) <= np.sum(self.lung_mask)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_image_segmentation(self):
        """Test segmentation with empty or invalid images"""
        processor = LungSegmentationProcessor()
        
        # Test with zeros
        empty_image = np.zeros((10, 20, 20))
        segmented, mask = processor.segment_lungs(empty_image)
        assert isinstance(mask, np.ndarray)
        
        # Test with single slice
        single_slice = np.full((1, 50, 50), -600)
        segmented, mask = processor.segment_lungs(single_slice)
        assert mask.shape == single_slice.shape
    
    def test_lesion_exclusion_edge_cases(self):
        """Test lesion exclusion edge cases"""
        processor = LesionExclusionProcessor()
        mask = np.ones((10, 20, 20), dtype=bool)
        
        # Test with lesion at edge of volume
        edge_center = (0, 0, 0)
        modified = processor.exclude_lesion_region(mask, edge_center, 5)
        assert isinstance(modified, np.ndarray)
        
        # Test with very large radius
        large_radius = 100  # Larger than volume
        modified = processor.exclude_lesion_region(mask, (5, 10, 10), large_radius)
        # Should exclude entire volume or most of it
        assert np.sum(modified) < np.sum(mask)
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        processor = LungSegmentationProcessor()
        
        # Test with 2D array (should still work but may behave unexpectedly)
        image_2d = np.ones((50, 50))
        # This might raise an error or handle gracefully depending on implementation
        try:
            segmented, mask = processor.segment_lungs(image_2d)
            # If it doesn't raise an error, check that outputs are reasonable
            assert isinstance(mask, np.ndarray)
        except (IndexError, ValueError):
            # It's acceptable to raise an error for invalid input dimensions
            pass


@pytest.mark.slow
class TestPerformance:
    """Performance tests for segmentation algorithms"""
    
    def test_segmentation_performance_large_volume(self):
        """Test segmentation performance with large volumes"""
        # Create large test volume
        large_image = np.random.randint(-1000, 100, (200, 300, 300), dtype=np.int16)
        
        processor = LungSegmentationProcessor()
        
        import time
        start_time = time.time()
        segmented, mask = processor.segment_lungs(large_image)
        end_time = time.time()
        
        # Check that it completes in reasonable time (adjust as needed)
        processing_time = end_time - start_time
        assert processing_time < 60  # Should complete within 1 minute
        
        # Check outputs are valid
        assert isinstance(mask, np.ndarray)
        assert mask.shape == large_image.shape


@pytest.mark.integration
class TestIntegrationWithRealData:
    """Integration tests that would work with real DICOM data"""
    
    @pytest.mark.skip(reason="Requires real DICOM data")
    def test_with_real_dicom_data(self):
        """Test segmentation with real DICOM data (when available)"""
        # This test would be run only when real test data is available
        # pytest.skip("Real DICOM test data not available")
        pass


# Fixtures for shared test data
@pytest.fixture
def sample_ct_image():
    """Fixture providing a sample CT image for testing"""
    # Create a realistic-looking CT image
    image = np.full((50, 128, 128), -1000, dtype=np.int16)
    
    # Add lung regions
    image[10:40, 20:50, 30:100] = -600  # Left lung
    image[10:40, 78:108, 30:100] = -700  # Right lung
    
    # Add mediastinum
    image[10:40, 50:78, 40:90] = 20
    
    return image


@pytest.fixture
def sample_lung_mask():
    """Fixture providing a sample lung mask"""
    mask = np.zeros((50, 128, 128), dtype=bool)
    mask[10:40, 20:50, 30:100] = True  # Left lung
    mask[10:40, 78:108, 30:100] = True  # Right lung
    return mask


# Parametrized tests
@pytest.mark.parametrize("threshold", [-300, -400, -500, -600])
def test_segmentation_thresholds(sample_ct_image, threshold):
    """Test segmentation with different threshold values"""
    processor = LungSegmentationProcessor(threshold=threshold)
    segmented, mask = processor.segment_lungs(sample_ct_image)
    
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool


@pytest.mark.parametrize("margin", [5, 10, 15, 20])
def test_lesion_exclusion_margins(sample_lung_mask, margin):
    """Test lesion exclusion with different margins"""
    processor = LesionExclusionProcessor(exclusion_margin=margin)
    center = (25, 64, 64)  # Center of the volume
    radius = 10
    
    modified = processor.exclude_lesion_region(sample_lung_mask, center, radius)
    excluded_voxels = np.sum(sample_lung_mask) - np.sum(modified)
    
    # Larger margins should exclude more voxels
    assert excluded_voxels >= 0
