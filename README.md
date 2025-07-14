# 4D-CT Strain Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python toolkit for analyzing respiratory motion and strain from 4D-CT imaging data. This application provides automated lung segmentation, displacement field calculation, and strain parameter analysis for medical imaging research.

## Features

- **Multi-phase 4D-CT Data Processing**: Automated loading and preprocessing of respiratory-gated CT datasets
- **Advanced Lung Segmentation**: Threshold-based segmentation with morphological post-processing
- **Motion Analysis**: Optical flow-based displacement field calculation using Farneback algorithm
- **Strain Parameter Calculation**: Principal strain analysis with comprehensive statistical metrics
- **Lesion Exclusion**: Automated exclusion of tumor regions with configurable safety margins
- **Comprehensive Visualization**: Multi-view motion vector and strain field visualizations
- **Detailed Reporting**: Automated generation of analysis reports in multiple formats (TXT, CSV, JSON)

## Technical Specifications

### Core Analysis Parameters

- **PSmax (Maximum Principal Strain)**: Peak deformation capacity within functional lung tissue
- **PSmean (Mean Principal Strain)**: Average strain level across analyzed lung regions
- **Speedmax (Maximum Displacement Speed)**: Peak velocity of respiratory motion in mm/s

### Processing Pipeline

1. **Data Validation and Loading**: DICOM series validation and multi-phase data loading
2. **Lung Segmentation**: Automated threshold-based segmentation (-400 HU) with region filtering
3. **Lesion Exclusion**: Optional tumor region exclusion with 10mm safety margin
4. **Displacement Calculation**: Farneback optical flow algorithm for inter-phase motion analysis
5. **Strain Analysis**: Principal eigenvalue analysis of deformation tensor fields
6. **Statistical Extraction**: Comprehensive parameter calculation from functional lung regions

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended for processing full-resolution 4D-CT datasets
- CUDA-compatible GPU (optional, for accelerated processing)

### Standard Installation

```bash
git clone https://github.com/yourusername/4dct-strain-analysis.git
cd 4dct-strain-analysis
pip install -r requirements.txt
```

### Development Installation

```bash
git clone https://github.com/yourusername/4dct-strain-analysis.git
cd 4dct-strain-analysis
pip install -e .
pip install -r requirements-dev.txt
```

## Quick Start

### Command Line Interface

```bash
# Basic analysis
python main.py --data-dir /path/to/4dct/data --patient-id PATIENT001

# Analysis with lesion exclusion
python main.py --data-dir /path/to/data --patient-id PATIENT001 \
               --lesion-center 50,100,120 --lesion-radius 15 \
               --output-dir /path/to/output
```

### Programmatic Usage

```python
from main import FourDCTStrainAnalyzer

# Initialize analyzer
analyzer = FourDCTStrainAnalyzer(
    data_directory="/path/to/4dct/data",
    patient_id="PATIENT001",
    output_directory="./output"
)

# Execute complete analysis
results = analyzer.run_complete_analysis(
    lesion_center=(50, 100, 120),  # z, y, x coordinates
    lesion_radius=15
)

# Access results
print(f"PSmax: {results['PSmax_all']:.6f}")
print(f"PSmean: {results['PSmean_all']:.6f}")
print(f"Speedmax: {results['Speedmax_all']:.4f} mm/s")
```

## Data Structure Requirements

### Input Directory Structure

```
data_directory/
├── phase_0/
│   ├── IM_0001.dcm
│   ├── IM_0002.dcm
│   └── ...
├── phase_1/
│   ├── IM_0001.dcm
│   └── ...
└── phase_N/
    └── ...
```

### DICOM Compatibility

- Standard DICOM format compliance required
- Multi-slice CT series with consistent spacing
- Respiratory-gated acquisition (4D-CT) with 10+ phases recommended
- Axial slice orientation preferred

## Configuration Options

### Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_phases` | 11 | Number of respiratory phases |
| `resample_factor` | 0.5 | Memory optimization resampling factor |
| `lung_threshold` | -400 HU | Lung segmentation threshold |
| `exclusion_margin` | 10 mm | Lesion exclusion safety margin |
| `baseline_phase_idx` | 0 | Reference phase for strain calculation |

### Advanced Configuration

```python
# Custom configuration example
analyzer = FourDCTStrainAnalyzer(
    data_directory="/path/to/data",
    patient_id="PATIENT001",
    num_phases=11,
    resample_factor=0.5,
    lung_threshold=-400,
    exclusion_margin=10,
    baseline_phase_idx=0
)
```

## Output Files

### Generated Reports

- **Text Report**: `{patient_id}_4dct_strain_analysis_report.txt`
- **CSV Summary**: `{patient_id}_strain_summary.csv`
- **JSON Results**: `{patient_id}_results.json`

### Visualization Outputs

- **Motion Vectors**: `{patient_id}_motion_vectors.png`
- **Strain Fields**: `{patient_id}_strain_field.png`

## Clinical Applications

### Research Applications

- Pulmonary function assessment in respiratory disease
- Treatment response monitoring in lung cancer patients
- Regional lung compliance analysis
- Radiation therapy planning optimization

### Quality Assurance Considerations

- Results should be interpreted alongside clinical presentation
- Motion artifacts and patient compliance may affect measurement accuracy
- Correlation with pulmonary function tests recommended when available
- Temporal resolution limited by 4D-CT acquisition parameters

## API Reference

### Core Classes

#### `FourDCTStrainAnalyzer`
Primary interface for complete analysis workflow

#### `FourDCTDataLoader`
Handles DICOM data loading and preprocessing

#### `LungSegmentationProcessor`
Automated lung region segmentation

#### `OpticalFlowProcessor`
Displacement field calculation using optical flow

#### `StrainAnalyzer`
Strain parameter computation and analysis

### Module Structure

```
4dct_strain_analysis/
├── config.py           # Configuration parameters
├── data_loader.py      # Data loading functionality
├── segmentation.py     # Image segmentation methods
├── motion_analysis.py  # Motion and strain analysis
├── visualization.py    # Visualization utilities
├── report_generator.py # Report generation tools
└── main.py            # Main analysis pipeline
```

## Development

### Running Tests

```bash
pytest tests/
pytest tests/ --cov=src/ --cov-report=html
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## Performance Considerations

### Memory Optimization

- Default resampling factor (0.5) reduces memory usage by 87.5%
- Processing time scales with dataset size and number of phases
- Recommended minimum 8GB RAM for standard datasets

### Computational Requirements

- CPU-based processing with OpenCV acceleration
- Typical processing time: 5-15 minutes per patient (11 phases)
- GPU acceleration not currently implemented but planned for future releases

## Limitations

- Analysis excludes tumor region and surrounding tissue (10mm margin)
- Temporal resolution limited by 4D-CT acquisition parameters
- Results represent relative rather than absolute strain measurements
- Requires consistent patient positioning across respiratory phases

## Citation

If you use this software in your research, please cite:

```bibtex
@software{4dct_strain_analysis,
  title={4D-CT Strain Analysis: A Python Toolkit for Respiratory Motion Assessment},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/4dct-strain-analysis}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/4dct-strain-analysis/issues)
- **Documentation**: [Project Wiki](https://github.com/yourusername/4dct-strain-analysis/wiki)
- **Email**: your.email@institution.edu

## Acknowledgments

- Medical Imaging Research Center
- Radiation Oncology Department
- Open-source scientific Python community

---

**Note**: This software is intended for research purposes only and should not be used for clinical decision-making without appropriate validation and regulatory approval.
