"""
Main Module for 4D-CT Strain Analysis
Orchestrates the complete analysis pipeline and provides CLI interface
"""

import os
import sys
import argparse
import traceback
from typing import Dict, Optional, Any

# Import analysis modules
from data_loader import FourDCTDataLoader
from segmentation import LungSegmentationProcessor, LesionExclusionProcessor
from motion_analysis import OpticalFlowProcessor, StrainAnalyzer
from visualization import MotionVectorVisualizer, StrainFieldVisualizer
from report_generator import AnalysisReportGenerator
from config import DEFAULT_PARAMS, configure_matplotlib


class FourDCTStrainAnalyzer:
    """Main class for coordinating 4D-CT strain analysis workflow"""
    
    def __init__(self, data_directory: str, patient_id: Optional[str] = None, 
                 output_directory: Optional[str] = None, **kwargs):
        """
        Initialize the 4D-CT strain analyzer
        
        Parameters:
        - data_directory: Path to 4D-CT data directory
        - patient_id: Patient identifier
        - output_directory: Output directory for results
        - **kwargs: Additional configuration parameters
        """
        self.data_directory = data_directory
        self.patient_id = patient_id or "unknown"
        self.output_directory = output_directory or "./output"
        
        # Analysis parameters
        self.config = {**DEFAULT_PARAMS, **kwargs}
        
        # Initialize processors
        self.data_loader = FourDCTDataLoader(
            data_directory=self.data_directory,
            num_phases=self.config['num_phases'],
            phase_pattern=self.config['phase_pattern'],
            resample_factor=self.config['resample_factor']
        )
        
        self.lung_segmentor = LungSegmentationProcessor(
            threshold=self.config['lung_threshold'],
            min_region_area=self.config['min_region_area'],
            border_margin=self.config['border_margin']
        )
        
        self.lesion_excluder = LesionExclusionProcessor(
            exclusion_margin=self.config['exclusion_margin']
        )
        
        self.optical_flow_processor = OpticalFlowProcessor(
            downsample_factor=self.config['optical_flow_downsample']
        )
        
        self.strain_analyzer = StrainAnalyzer(
            time_interval=self.config['time_interval']
        )
        
        self.report_generator = AnalysisReportGenerator(self.output_directory)
        
        # Configure visualization
        configure_matplotlib()
        
        # Ensure output directory exists
        self._ensure_output_directory()
    
    def run_complete_analysis(self, lesion_center: Optional[tuple] = None, 
                            lesion_radius: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Execute the complete 4D-CT strain analysis pipeline
        
        Parameters:
        - lesion_center: Lesion center coordinates (z, y, x)
        - lesion_radius: Lesion radius in voxels
        
        Returns:
        - results: Dictionary containing analysis results, or None if failed
        """
        try:
            print("=" * 60)
            print("4D-CT Strain Analysis Pipeline Starting")
            print("=" * 60)
            print(f"Patient ID: {self.patient_id}")
            print(f"Data Directory: {self.data_directory}")
            print(f"Output Directory: {self.output_directory}")
            
            # Step 1: Validate and load data
            results = self._execute_data_loading()
            if not results:
                return None
            
            phases, baseline_phase_idx = results
            
            # Step 2: Perform lung segmentation
            lung_masks = self._execute_lung_segmentation(phases, lesion_center, lesion_radius)
            if not lung_masks:
                return None
            
            # Step 3: Calculate strain parameters
            strain_results = self._execute_strain_analysis(phases, lung_masks, baseline_phase_idx)
            if not strain_results:
                return None
            
            # Step 4: Generate visualizations and reports
            self._execute_visualization_and_reporting(strain_results, phases, lung_masks, baseline_phase_idx)
            
            print("\n" + "=" * 60)
            print("Analysis completed successfully")
            print("=" * 60)
            
            return strain_results
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            traceback.print_exc()
            return None
    
    def _execute_data_loading(self):
        """Execute data loading phase"""
        print("\nStep 1: Data Validation and Loading")
        print("-" * 40)
        
        # Validate data directory structure
        is_valid, message = self.data_loader.validate_data_directory()
        print(f"Data validation: {message}")
        
        if not is_valid:
            print("Error: Invalid data directory structure")
            return None
        
        # Load 4D-CT data
        phases = self.data_loader.load_4dct_data()
        
        if all(p is None for p in phases):
            print("Error: Unable to load any phase data")
            return None
        
        # Determine baseline phase
        baseline_phase_idx = self._determine_baseline_phase(phases)
        print(f"Using phase {baseline_phase_idx} as baseline reference")
        
        return phases, baseline_phase_idx
    
    def _execute_lung_segmentation(self, phases, lesion_center, lesion_radius):
        """Execute lung segmentation phase"""
        print("\nStep 2: Lung Segmentation and Lesion Exclusion")
        print("-" * 50)
        
        lung_masks = []
        
        for phase_idx, phase in enumerate(phases):
            if phase is None:
                lung_masks.append(None)
                continue
            
            print(f"Segmenting lungs for phase {phase_idx}")
            
            # Perform lung segmentation
            _, mask = self.lung_segmentor.segment_lungs(phase)
            
            # Exclude lesion region if specified
            if lesion_center is not None and lesion_radius is not None:
                mask = self.lesion_excluder.exclude_lesion_region(
                    mask, lesion_center, lesion_radius
                )
            
            lung_masks.append(mask)
        
        # Validate that we have at least one valid mask
        if all(m is None for m in lung_masks):
            print("Error: Unable to segment lungs for any phase")
            return None
        
        return lung_masks
    
    def _execute_strain_analysis(self, phases, lung_masks, baseline_phase_idx):
        """Execute strain analysis phase"""
        print("\nStep 3: Displacement Field Calculation and Strain Analysis")
        print("-" * 60)
        
        reference_phase = phases[baseline_phase_idx]
        reference_mask = lung_masks[baseline_phase_idx]
        
        if reference_phase is None or reference_mask is None:
            print("Error: Invalid reference phase or mask")
            return None
        
        all_strain_params = []
        
        # Analyze each phase against baseline
        for i, (phase, mask) in enumerate(zip(phases, lung_masks)):
            if i == baseline_phase_idx or phase is None or mask is None:
                continue
            
            print(f"Analyzing phase {i} → baseline phase {baseline_phase_idx}")
            
            # Calculate displacement field
            displacement_field = self.optical_flow_processor.calculate_optical_flow(
                reference_phase, phase
            )
            
            # Calculate strain parameters
            strain_params = self.strain_analyzer.calculate_strain_parameters(
                displacement_field, reference_mask
            )
            strain_params['phase_index'] = i
            all_strain_params.append(strain_params)
        
        if not all_strain_params:
            print("Error: Unable to calculate strain parameters for any phase")
            return None
        
        # Compile final results
        return self._compile_final_results(all_strain_params, baseline_phase_idx, 
                                         reference_phase, reference_mask)
    
    def _execute_visualization_and_reporting(self, results, phases, lung_masks, baseline_phase_idx):
        """Execute visualization and reporting phase"""
        print("\nStep 4: Visualization and Report Generation")
        print("-" * 45)
        
        # Select representative phase for visualization
        best_phase = results['individual_phases'][len(results['individual_phases'])//2]
        
        # Generate motion vector visualization
        motion_visualizer = MotionVectorVisualizer(
            arrow_density=self.config['arrow_density'],
            arrow_scale=self.config['arrow_scale']
        )
        
        fig1 = motion_visualizer.visualize_motion_vectors(
            results['reference_image'],
            best_phase['displacement_field'],
            results['reference_mask']
        )
        
        # Generate strain field visualization
        strain_visualizer = StrainFieldVisualizer()
        fig2 = strain_visualizer.visualize_strain_field(
            results['reference_image'],
            best_phase['principal_strain_field'],
            results['reference_mask']
        )
        
        # Save visualization figures
        fig1.savefig(
            os.path.join(self.output_directory, f"{self.patient_id}_motion_vectors.png"),
            dpi=300, bbox_inches='tight'
        )
        fig2.savefig(
            os.path.join(self.output_directory, f"{self.patient_id}_strain_field.png"),
            dpi=300, bbox_inches='tight'
        )
        
        print("Visualization figures saved")
        
        # Generate comprehensive report
        metadata = {
            'baseline_phase': baseline_phase_idx,
            'total_phases_analyzed': len(results['individual_phases']),
            'lesion_exclusion': 'Applied' if 'lesion_center' in self.config else 'Not applied',
            'resample_factor': self.config['resample_factor']
        }
        
        self.report_generator.generate_analysis_report(results, self.patient_id, metadata)
    
    def _determine_baseline_phase(self, phases):
        """Determine the optimal baseline phase"""
        baseline_idx = self.config['baseline_phase_idx']
        
        if phases[baseline_idx] is not None:
            return baseline_idx
        
        # Find first available phase
        for i, phase in enumerate(phases):
            if phase is not None:
                return i
        
        return 0
    
    def _compile_final_results(self, all_strain_params, baseline_phase_idx, 
                              reference_phase, reference_mask):
        """Compile final analysis results"""
        # Calculate global statistics
        psmax_values = [sp['PSmax'] for sp in all_strain_params]
        psmean_values = [sp['PSmean'] for sp in all_strain_params]
        speedmax_values = [sp['Speedmax'] for sp in all_strain_params]
        
        final_results = {
            'PSmax_all': max(psmax_values),
            'PSmean_all': sum(psmean_values) / len(psmean_values),
            'Speedmax_all': max(speedmax_values),
            'individual_phases': all_strain_params,
            'reference_phase': baseline_phase_idx,
            'reference_image': reference_phase,
            'reference_mask': reference_mask,
            'patient_id': self.patient_id,
            'success': True
        }
        
        print(f"\nFinal Analysis Results:")
        print(f"  PSmax (Global): {final_results['PSmax_all']:.6f}")
        print(f"  PSmean (Average): {final_results['PSmean_all']:.6f}")
        print(f"  Speedmax (Global): {final_results['Speedmax_all']:.4f} mm/s")
        
        return final_results
    
    def _ensure_output_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)


def create_argument_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="4D-CT Strain Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data-dir /path/to/4dct/data --patient-id PATIENT001
  python main.py --data-dir /path/to/data --output-dir /path/to/output --lesion-center 50,100,120 --lesion-radius 15
        """
    )
    
    # Required arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to 4D-CT data directory')
    
    # Optional arguments
    parser.add_argument('--patient-id', type=str, default=None,
                       help='Patient identifier')
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='Output directory for results')
    parser.add_argument('--lesion-center', type=str, default=None,
                       help='Lesion center coordinates (z,y,x)')
    parser.add_argument('--lesion-radius', type=int, default=15,
                       help='Lesion radius in voxels')
    parser.add_argument('--num-phases', type=int, default=11,
                       help='Number of respiratory phases')
    parser.add_argument('--resample-factor', type=float, default=0.5,
                       help='Resampling factor for memory optimization')
    parser.add_argument('--baseline-phase', type=int, default=0,
                       help='Baseline phase index')
    
    return parser


def parse_lesion_center(lesion_center_str):
    """Parse lesion center coordinates from string"""
    if lesion_center_str is None:
        return None
    
    try:
        coords = [int(x.strip()) for x in lesion_center_str.split(',')]
        if len(coords) != 3:
            raise ValueError("Lesion center must have exactly 3 coordinates")
        return tuple(coords)
    except ValueError as e:
        print(f"Error parsing lesion center: {e}")
        return None


def main():
    """Main entry point for command-line interface"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Parse lesion center if provided
    lesion_center = parse_lesion_center(args.lesion_center)
    
    # Determine patient ID
    patient_id = args.patient_id
    if patient_id is None:
        patient_id = os.path.basename(args.data_dir.rstrip(os.sep))
    
    # Create analyzer instance
    analyzer = FourDCTStrainAnalyzer(
        data_directory=args.data_dir,
        patient_id=patient_id,
        output_directory=args.output_dir,
        num_phases=args.num_phases,
        resample_factor=args.resample_factor,
        baseline_phase_idx=args.baseline_phase
    )
    
    # Execute analysis
    results = analyzer.run_complete_analysis(
        lesion_center=lesion_center,
        lesion_radius=args.lesion_radius
    )
    
    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        sys.exit(0)
    else:
        print(f"\nAnalysis failed. Please check the error messages above.")
        sys.exit(1)


# Convenience function for programmatic use
def main_4dct_strain_analysis(data_directory, patient_id=None, output_dir=None, 
                             lesion_center=None, lesion_radius=15, num_phases=11,
                             baseline_phase_idx=0, resample_factor=0.5):
    """
    Convenience function for backward compatibility with original interface
    
    Parameters:
    - data_directory: 4D-CT data directory
    - patient_id: Patient identifier
    - output_dir: Output directory
    - lesion_center: Lesion center coordinates (z, y, x)
    - lesion_radius: Lesion radius in voxels
    - num_phases: Number of respiratory phases
    - baseline_phase_idx: Baseline phase index
    - resample_factor: Resampling factor
    
    Returns:
    - strain_params: Dictionary containing strain parameters
    """
    analyzer = FourDCTStrainAnalyzer(
        data_directory=data_directory,
        patient_id=patient_id,
        output_directory=output_dir,
        num_phases=num_phases,
        baseline_phase_idx=baseline_phase_idx,
        resample_factor=resample_factor
    )
    
    return analyzer.run_complete_analysis(
        lesion_center=lesion_center,
        lesion_radius=lesion_radius
    )


if __name__ == "__main__":
    main()
