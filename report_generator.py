"""
Report Generation Module for 4D-CT Strain Analysis
Handles analysis report generation and data export
"""

import os
import pandas as pd
from datetime import datetime
import json


class AnalysisReportGenerator:
    """Class for generating comprehensive analysis reports"""
    
    def __init__(self, output_directory=None):
        """
        Initialize the report generator
        
        Parameters:
        - output_directory: Directory for saving reports
        """
        self.output_directory = output_directory or os.getcwd()
        self._ensure_output_directory()
    
    def generate_analysis_report(self, strain_params, patient_id=None, analysis_metadata=None):
        """
        Generate comprehensive analysis report
        
        Parameters:
        - strain_params: Dictionary containing strain parameters
        - patient_id: Patient identifier
        - analysis_metadata: Additional metadata about the analysis
        
        Returns:
        - report_path: Path to the generated report file
        """
        if patient_id is None:
            patient_id = 'unknown'
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate report content
        report_content = self._generate_report_content(
            strain_params, patient_id, timestamp, analysis_metadata
        )
        
        # Save text report
        report_path = self._save_text_report(report_content, patient_id)
        
        # Generate additional outputs
        self._generate_csv_summary(strain_params, patient_id)
        self._generate_json_results(strain_params, patient_id, analysis_metadata)
        
        print(f"Analysis report generated and saved to: {report_path}")
        return report_path
    
    def _generate_report_content(self, strain_params, patient_id, timestamp, metadata):
        """
        Generate the main report content
        
        Parameters:
        - strain_params: Strain analysis results
        - patient_id: Patient identifier
        - timestamp: Analysis timestamp
        - metadata: Analysis metadata
        
        Returns:
        - report_content: Formatted report text
        """
        # Extract key results
        psmax = strain_params.get('PSmax_all', strain_params.get('PSmax', 0))
        psmean = strain_params.get('PSmean_all', strain_params.get('PSmean', 0))
        speedmax = strain_params.get('Speedmax_all', strain_params.get('Speedmax', 0))
        
        # Build report sections
        header = self._generate_header_section(patient_id, timestamp)
        summary = self._generate_summary_section(psmax, psmean, speedmax)
        methods = self._generate_methods_section()
        clinical = self._generate_clinical_significance_section()
        metadata_section = self._generate_metadata_section(metadata)
        
        return f"{header}\n\n{summary}\n\n{methods}\n\n{clinical}\n\n{metadata_section}"
    
    def _generate_header_section(self, patient_id, timestamp):
        """Generate report header section"""
        return f"""4D-CT Strain Analysis Report
{'=' * 50}

Patient ID: {patient_id}
Analysis Date: {timestamp}
Software Version: 4DCT-StrainAnalysis v1.0
Institution: Medical Imaging Research Center"""
    
    def _generate_summary_section(self, psmax, psmean, speedmax):
        """Generate results summary section"""
        return f"""Analysis Results Summary
{'=' * 30}

Primary Strain Parameters:
• Maximum Principal Strain (PSmax): {psmax:.6f}
• Mean Principal Strain (PSmean): {psmean:.6f}
• Maximum Displacement Speed (Speedmax): {speedmax:.4f} mm/s

Interpretation:
• PSmax represents the maximum deformation capacity observed in lung tissue
• PSmean indicates the average strain level across functional lung regions
• Speedmax quantifies the peak velocity of respiratory motion"""
    
    def _generate_methods_section(self):
        """Generate methodology section"""
        return f"""Analysis Methodology
{'=' * 25}

Data Processing Pipeline:
1. Multi-phase 4D-CT data loading and preprocessing
2. Automated lung segmentation using threshold-based approach
3. Tumor lesion exclusion with 10mm safety margin
4. Displacement field calculation via Farneback optical flow algorithm
5. Strain tensor analysis and principal strain computation
6. Statistical parameter extraction from functional lung regions

Technical Specifications:
• Lung segmentation threshold: -400 HU
• Optical flow algorithm: Farneback method with pyramid scaling
• Strain calculation: Principal eigenvalue analysis of deformation tensor
• Temporal resolution: Phase-to-phase displacement analysis
• Spatial processing: Full-resolution analysis with computational optimization"""
    
    def _generate_clinical_significance_section(self):
        """Generate clinical significance section"""
        return f"""Clinical Significance
{'=' * 25}

Parameter Interpretation:
• PSmax values > 0.1 may indicate regions of high mechanical stress
• PSmean provides baseline assessment of overall lung compliance
• Speedmax correlates with respiratory efficiency and diaphragmatic function

Quality Assurance:
• Results should be interpreted alongside clinical presentation
• Motion artifacts and patient compliance may affect accuracy
• Recommend correlation with pulmonary function tests when available

Limitations:
• Analysis excludes tumor region and 10mm surrounding tissue
• Temporal resolution limited by 4D-CT acquisition parameters
• Results represent relative rather than absolute strain measurements"""
    
    def _generate_metadata_section(self, metadata):
        """Generate metadata section"""
        if not metadata:
            return "Processing Parameters\n" + "=" * 25 + "\n\nStandard processing parameters applied."
        
        metadata_text = "Processing Parameters\n" + "=" * 25 + "\n\n"
        for key, value in metadata.items():
            metadata_text += f"• {key}: {value}\n"
        
        return metadata_text
    
    def _save_text_report(self, content, patient_id):
        """Save text report to file"""
        filename = f"{patient_id}_4dct_strain_analysis_report.txt"
        report_path = os.path.join(self.output_directory, filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return report_path
    
    def _generate_csv_summary(self, strain_params, patient_id):
        """Generate CSV summary of results"""
        # Extract results for CSV
        results_data = {
            'Patient_ID': [patient_id],
            'PSmax': [strain_params.get('PSmax_all', strain_params.get('PSmax', 0))],
            'PSmean': [strain_params.get('PSmean_all', strain_params.get('PSmean', 0))],
            'Speedmax': [strain_params.get('Speedmax_all', strain_params.get('Speedmax', 0))],
            'Analysis_Date': [datetime.now().strftime('%Y-%m-%d')],
            'Analysis_Time': [datetime.now().strftime('%H:%M:%S')]
        }
        
        # Add individual phase results if available
        if 'individual_phases' in strain_params:
            phase_count = len(strain_params['individual_phases'])
            results_data['Analyzed_Phases'] = [phase_count]
        
        df = pd.DataFrame(results_data)
        csv_path = os.path.join(self.output_directory, f"{patient_id}_strain_summary.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"CSV summary saved to: {csv_path}")
    
    def _generate_json_results(self, strain_params, patient_id, metadata):
        """Generate JSON results file for programmatic access"""
        json_data = {
            'patient_id': patient_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'software_version': '4DCT-StrainAnalysis v1.0',
            'results': {
                'PSmax': strain_params.get('PSmax_all', strain_params.get('PSmax', 0)),
                'PSmean': strain_params.get('PSmean_all', strain_params.get('PSmean', 0)),
                'Speedmax': strain_params.get('Speedmax_all', strain_params.get('Speedmax', 0))
            },
            'metadata': metadata or {}
        }
        
        # Add individual phase results if available
        if 'individual_phases' in strain_params:
            json_data['individual_phases'] = []
            for phase_result in strain_params['individual_phases']:
                phase_data = {
                    'phase_index': phase_result.get('phase_index'),
                    'PSmax': phase_result.get('PSmax', 0),
                    'PSmean': phase_result.get('PSmean', 0),
                    'Speedmax': phase_result.get('Speedmax', 0)
                }
                json_data['individual_phases'].append(phase_data)
        
        json_path = os.path.join(self.output_directory, f"{patient_id}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"JSON results saved to: {json_path}")
    
    def _ensure_output_directory(self):
        """Ensure output directory exists"""
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            print(f"Created output directory: {self.output_directory}")


class BatchReportGenerator:
    """Class for generating batch analysis reports"""
    
    def __init__(self, output_directory):
        """
        Initialize batch report generator
        
        Parameters:
        - output_directory: Directory for saving batch reports
        """
        self.output_directory = output_directory
        self.report_generator = AnalysisReportGenerator(output_directory)
    
    def generate_batch_summary(self, batch_results):
        """
        Generate summary report for batch analysis
        
        Parameters:
        - batch_results: List of individual analysis results
        
        Returns:
        - summary_path: Path to batch summary file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compile batch statistics
        batch_data = []
        for result in batch_results:
            if result and 'patient_id' in result:
                batch_data.append({
                    'Patient_ID': result['patient_id'],
                    'PSmax': result.get('PSmax_all', result.get('PSmax', 0)),
                    'PSmean': result.get('PSmean_all', result.get('PSmean', 0)),
                    'Speedmax': result.get('Speedmax_all', result.get('Speedmax', 0)),
                    'Status': 'Success' if result.get('success', True) else 'Failed'
                })
        
        # Create batch summary DataFrame
        df = pd.DataFrame(batch_data)
        
        # Generate summary statistics
        if len(df) > 0:
            summary_stats = {
                'Total_Patients': len(df),
                'Successful_Analyses': len(df[df['Status'] == 'Success']),
                'Failed_Analyses': len(df[df['Status'] == 'Failed']),
                'Mean_PSmax': df[df['Status'] == 'Success']['PSmax'].mean(),
                'Mean_PSmean': df[df['Status'] == 'Success']['PSmean'].mean(),
                'Mean_Speedmax': df[df['Status'] == 'Success']['Speedmax'].mean()
            }
        else:
            summary_stats = {'Total_Patients': 0}
        
        # Save batch summary
        summary_path = os.path.join(self.output_directory, f"batch_summary_{timestamp}.csv")
        df.to_csv(summary_path, index=False)
        
        # Save batch statistics
        stats_path = os.path.join(self.output_directory, f"batch_statistics_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Batch summary saved to: {summary_path}")
        print(f"Batch statistics saved to: {stats_path}")
        
        return summary_path


# Convenience function for backward compatibility
def generate_analysis_report(strain_params, patient_id=None, output_dir=None):
    """
    Convenience function for generating analysis reports
    
    Parameters:
    - strain_params: Dictionary containing strain parameters
    - patient_id: Patient identifier
    - output_dir: Output directory
    
    Returns:
    - report_path: Path to generated report
    """
    generator = AnalysisReportGenerator(output_dir)
    return generator.generate_analysis_report(strain_params, patient_id)
