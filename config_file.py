"""
Configuration file for 4D-CT Strain Analysis
Contains default parameters and constants used throughout the application
"""

import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Matplotlib configuration for consistent output
MATPLOTLIB_CONFIG = {
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.unicode_minus': False,
    'font.family': 'sans-serif',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
}

def configure_matplotlib():
    """Apply matplotlib configuration"""
    for key, value in MATPLOTLIB_CONFIG.items():
        plt.rcParams[key] = value

# Default analysis parameters
DEFAULT_PARAMS = {
    'num_phases': 11,
    'phase_pattern': "phase_{}",
    'resample_factor': 0.5,
    'lung_threshold': -400,
    'baseline_phase_idx': 0,
    'lesion_radius': 15,
    'exclusion_margin': 10,
    'time_interval': 0.1,  # seconds
    'min_region_area': 1000,
    'border_margin': 10,
    'optical_flow_downsample': 2,
    'arrow_density': 5,
    'arrow_scale': 50
}

# Optical flow parameters
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

# Morphological operation parameters
MORPHOLOGY_PARAMS = {
    'closing_disk_size': 5
}

# File patterns and extensions
FILE_PATTERNS = {
    'dicom_extensions': ['.dcm', '.DCM', '.dicom', '.DICOM'],
    'output_extensions': {
        'report': '.txt',
        'motion_vectors': '_motion_vectors.png',
        'strain_field': '_strain_field.png'
    }
}
