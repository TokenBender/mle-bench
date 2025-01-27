#!/usr/bin/env python3
import json
import os
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
RUN_GROUPS_DIR = Path('runs')
RUN_GROUP_CSV = RUN_GROUPS_DIR / 'run_group_experiments.csv'
OUTPUT_CSV = 'model_failure_analysis.csv'

# AIDE Model configurations
AIDE_MODELS = {
    'scaffolding-gpt4o-aide': {'model': 'GPT-4o', 'env': 'AIDE', 'notes': 'AIDE scaffolding'},
    'models-claude35sonnet-aide': {'model': 'Claude-3.5', 'env': 'AIDE', 'notes': 'Claude 3.5 Sonnet'},
    'models-llama-3.1-405B-instruct-aide': {'model': 'LLama-3.1', 'env': 'AIDE', 'notes': 'LLama 3.1 405B'},
    'models-o1-preview-aide': {'model': 'O1-Preview', 'env': 'AIDE', 'notes': 'o1-preview model'},
}

def load_run_group_mapping():
    """Load experiment to run group mapping from CSV"""
    mapping = defaultdict(list)
    with open(RUN_GROUP_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only include AIDE models
            if row['experiment_id'] in AIDE_MODELS:
                mapping[row['run_group']].append(row['experiment_id'])
    return mapping

def analyze_run_group(run_group_path, experiment_id):
    """Process a single run group directory"""
    task_data = defaultdict(lambda: {'successes': 0, 'total': 0, 'valid_submissions': 0})
    
    for report_file in run_group_path.glob('*grading_report.json'):
        try:
            with open(report_file) as f:
                report = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading {report_file}")
            continue
            
        for comp in report.get('competition_reports', []):
            task_id = comp['competition_id']
            success = comp.get('any_medal', False)
            valid = comp.get('valid_submission', False)
            
            if comp.get('submission_exists', False):
                task_data[task_id]['total'] += 1
                task_data[task_id]['successes'] += int(success)
                task_data[task_id]['valid_submissions'] += int(valid)
            
    return task_data

def generate_analysis_report(all_task_data):
    """Generate consolidated report"""
    report_data = []
    
    for task_id, model_stats in all_task_data.items():
        for model_id, stats in model_stats.items():
            total_runs = stats['total']
            if total_runs == 0:
                continue
                
            success_rate = stats['successes'] / total_runs if total_runs > 0 else 0
            valid_rate = stats['valid_submissions'] / total_runs if total_runs > 0 else 0
            failure_rate = 1 - success_rate
            
            # Categorize failure rate
            if failure_rate == 1.0:
                category = 'always_fail'
            elif failure_rate > 0.6:
                category = 'high_failure'
            elif failure_rate > 0.3:
                category = 'medium_failure'
            else:
                category = 'low_failure'
                
            model_info = AIDE_MODELS[model_id]
            report_data.append({
                'task_id': task_id,
                'model': model_info['model'],
                'environment': model_info['env'],
                'total_runs': total_runs,
                'valid_submissions': stats['valid_submissions'],
                'success_rate': success_rate,
                'valid_rate': valid_rate,
                'failure_category': category,
                'notes': model_info['notes']
            })
    
    return report_data

def main():
    run_group_mapping = load_run_group_mapping()
    all_data = defaultdict(lambda: defaultdict(lambda: {'successes': 0, 'total': 0, 'valid_submissions': 0}))
    
    # Process all run groups
    for run_group_name, experiment_ids in run_group_mapping.items():
        run_group_path = RUN_GROUPS_DIR / run_group_name
        if not run_group_path.is_dir():
            continue
            
        experiment_id = experiment_ids[0]  # Use first experiment ID
        group_data = analyze_run_group(run_group_path, experiment_id)
        
        # Aggregate data
        for task, stats in group_data.items():
            for key in ['successes', 'total', 'valid_submissions']:
                all_data[task][experiment_id][key] += stats[key]
    
    # Generate report
    final_report = generate_analysis_report(all_data)
    
    if not final_report:
        print("No data found in the reports!")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(final_report)
    
    # Save detailed CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("Model-wise Failure Analysis Report (AIDE only)")
    print("-" * 100)
    
    # Print per-model statistics
    print("\nPer-Model Failure Categories:")
    model_stats = df.groupby(['model', 'failure_category']).size().unstack(fill_value=0)
    print(model_stats)
    
    # Print per-model average success rate
    print("\nPer-Model Average Success Rate:")
    model_success = df.groupby('model')['success_rate'].agg(['mean', 'count'])
    print(model_success)
    
    print(f"\nDetailed results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main() 