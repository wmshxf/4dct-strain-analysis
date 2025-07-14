"""
Data Loader Module for 4D-CT Strain Analysis
Handles loading and preprocessing of 4D-CT DICOM data
"""

import os
import numpy as np
import pydicom
import SimpleITK as sitk
from config import DEFAULT_PARAMS


class FourDCTDataLoader:
    """Class for loading and preprocessing 4D-CT data"""
    
    def __init__(self, data_directory, num_phases=None, phase_pattern=None, resample_factor=None):
        """
        Initialize the data loader
        
        Parameters:
        - data_directory: Path to the 4D-CT data directory
        - num_phases: Number of respiratory phases
        - phase_pattern: Naming pattern for phase directories
        - resample_factor: Resampling factor to reduce memory usage
        """
        self.data_directory = data_directory
        self.num_phases = num_phases or DEFAULT_PARAMS['num_phases']
        self.phase_pattern = phase_pattern or DEFAULT_PARAMS['phase_pattern']
        self.resample_factor = resample_factor or DEFAULT_PARAMS['resample_factor']
        
    def load_4dct_data(self):
        """
        Load 4D-CT data with each phase stored in separate subdirectories
        
        Returns:
        - phases: List of image data arrays for each phase
        """
        phases = [None] * self.num_phases
        
        print(f"Loading {self.num_phases} phases from directory: {self.data_directory}")
        
        for phase_idx in range(self.num_phases):
            phase_dir = os.path.join(self.data_directory, self.phase_pattern.format(phase_idx))
            
            if not os.path.exists(phase_dir):
                print(f"Phase {phase_idx} directory does not exist: {phase_dir}")
                continue
                
            try:
                phase_array = self._load_single_phase(phase_dir, phase_idx)
                phases[phase_idx] = phase_array
                
            except Exception as e:
                print(f"Error loading phase {phase_idx}: {str(e)}")
        
        return phases
    
    def _load_single_phase(self, phase_dir, phase_idx):
        """
        Load a single phase from DICOM files
        
        Parameters:
        - phase_dir: Directory containing DICOM files for this phase
        - phase_idx: Phase index for logging
        
        Returns:
        - phase_array: Numpy array containing the phase data
        """
        # Find DICOM files
        dicom_files = self._find_dicom_files(phase_dir)
        
        if not dicom_files:
            print(f"No DICOM files found in phase {phase_idx}")
            return None
            
        print(f"Found {len(dicom_files)} DICOM files in phase {phase_idx}")
        
        # Read DICOM series using SimpleITK
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_files)
        phase_img = reader.Execute()
        
        # Apply resampling if needed
        if self.resample_factor < 1.0:
            phase_img = self._resample_image(phase_img)
        
        # Convert to numpy array
        phase_array = sitk.GetArrayFromImage(phase_img)
        print(f"Successfully loaded phase {phase_idx}, shape: {phase_array.shape}")
        
        return phase_array
    
    def _find_dicom_files(self, directory):
        """
        Recursively find all DICOM files in a directory
        
        Parameters:
        - directory: Directory to search
        
        Returns:
        - dicom_files: List of DICOM file paths
        """
        dicom_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    pydicom.dcmread(file_path, force=True)
                    dicom_files.append(file_path)
                except:
                    continue
                    
        return dicom_files
    
    def _resample_image(self, image):
        """
        Resample image to reduce memory usage
        
        Parameters:
        - image: SimpleITK image
        
        Returns:
        - resampled_image: Resampled SimpleITK image
        """
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(s * self.resample_factor) for s in original_size]
        new_spacing = [orig_spacing / self.resample_factor for orig_spacing in original_spacing]
        
        resample = sitk.ResampleImageFilter()
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing)
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetInterpolator(sitk.sitkLinear)
        
        return resample.Execute(image)
    
    def validate_data_directory(self):
        """
        Validate that the data directory exists and contains expected structure
        
        Returns:
        - is_valid: Boolean indicating if directory structure is valid
        - message: Validation message
        """
        if not os.path.exists(self.data_directory):
            return False, f"Data directory does not exist: {self.data_directory}"
        
        found_phases = 0
        for phase_idx in range(self.num_phases):
            phase_dir = os.path.join(self.data_directory, self.phase_pattern.format(phase_idx))
            if os.path.exists(phase_dir):
                found_phases += 1
        
        if found_phases == 0:
            return False, f"No phase directories found matching pattern '{self.phase_pattern}'"
        
        return True, f"Found {found_phases} out of {self.num_phases} expected phase directories"


def load_4dct_data(data_directory, num_phases=11, phase_pattern="phase_{}", resample_factor=0.5):
    """
    Convenience function for loading 4D-CT data
    
    Parameters:
    - data_directory: Path to the data directory
    - num_phases: Number of respiratory phases
    - phase_pattern: Naming pattern for phase directories
    - resample_factor: Resampling factor to reduce memory usage
    
    Returns:
    - phases: List of image data arrays for each phase
    """
    loader = FourDCTDataLoader(data_directory, num_phases, phase_pattern, resample_factor)
    return loader.load_4dct_data()
