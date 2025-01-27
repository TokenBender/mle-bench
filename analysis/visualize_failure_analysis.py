#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dominate import document
from dominate.tags import *
from dominate.util import raw

# Configuration
INPUT_CSV = 'model_failure_analysis.csv'
INPUT_JSON = 'model_failure_categories.json'
OUTPUT_DIR = Path('visualizations')
REPORT_HTML = OUTPUT_DIR / 'analysis_report.html'

# Style configuration
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
OUTPUT_DIR.mkdir(exist_ok=True)

EXPLANATIONS = {
    'success_rate_distribution': {
        'title': 'Success Rate Distribution',
        'content': '''
        <b>How to read:</b> This boxplot shows the distribution of success rates across all tasks for each model.
        - The box represents the interquartile range (IQR)
        - The line inside the box is the median
        - Whiskers show the range of typical values
        - Points outside whiskers are outliers
        
        <b>Insight:</b> Higher median and tighter IQR indicate more consistent performance.
        '''
    },
    'valid_submission_rates': {
        'title': 'Valid Submission Rates',
        'content': '''
        <b>How to read:</b> Bar heights show the average percentage of valid (technically correct) submissions.
        
        <b>Insight:</b> High valid rates indicate good technical execution, even if solutions aren't optimal.
        '''
    },
    'failure_category_distribution': {
        'title': 'Failure Category Breakdown',
        'content': '''
        <b>How to read:</b> Stacked bars show count of tasks in each failure category.
        - Always Fail: 0% success rate
        - High Failure: >60% failure rate
        - Medium Failure: 30-60% failure rate
        - Low Failure: <30% failure rate
        
        <b>Insight:</b> More low-failure tasks indicate better overall performance.
        '''
    },
    'task_success_heatmap': {
        'title': 'Task Success Heatmap',
        'content': '''
        <b>How to read:</b> Color intensity shows success rate per task-model combination.
        - Green: High success
        - Yellow: Medium success
        - Red: Low success
        
        <b>Insight:</b> Identify tasks where models excel or struggle collectively.
        '''
    },
    'failure_category_radar': {
        'title': 'Failure Profile Radar',
        'content': '''
        <b>How to read:</b> Shape shows proportion of tasks in each failure category.
        - Larger area in green (low failure) is better
        - Spikes in red (always fail) indicate problem areas
        
        <b>Insight:</b> Compare overall failure profiles at a glance.
        '''
    }
}

def create_html_report(image_files):
    """Generate HTML report with explanations"""
    with document(title='AI Model Failure Analysis') as doc:
        h1('AI Model Performance Analysis Report')
        hr()
        
        with div():
            h2('Report Contents')
            ul(
                li(a(exp['title'], href=f"#{path.stem}")) 
                for path, exp in zip(image_files, EXPLANATIONS.values())
            )
        
        for path, exp in zip(image_files, EXPLANATIONS.values()):
            with div(id=path.stem):
                h2(exp['title'])
                img(src=str(path.name), style="width: 80%; margin: 20px auto; display: block;")
                div(raw(exp['content']), style="padding: 20px; background: #f5f5f5; border-radius: 5px;")
                hr()
    
    with open(REPORT_HTML, 'w') as f:
        f.write(doc.render())

def load_data():
    """Load data from both CSV and JSON files"""
    df = pd.read_csv(INPUT_CSV)
    with open(INPUT_JSON) as f:
        json_data = json.load(f)
    return df, json_data

def plot_model_performance(df):
    """Create performance comparison plots"""
    # Model success rate distribution
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='success_rate', data=df)
    plt.title('Success Rate Distribution by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'success_rate_distribution.png')
    plt.close()

    # Valid submission rate comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='valid_rate', data=df, errorbar=None)
    plt.title('Average Valid Submission Rate by Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'valid_submission_rates.png')
    plt.close()

def plot_failure_categories(json_data):
    """Create failure category distribution plots"""
    # Prepare data
    category_counts = []
    for model, categories in json_data.items():
        for category, tasks in categories.items():
            category_counts.append({
                'model': model,
                'category': category,
                'count': len(tasks)
            })
    
    df_categories = pd.DataFrame(category_counts)
    
    # Stacked bar chart
    plt.figure(figsize=(14, 8))
    df_pivot = df_categories.pivot(index='model', columns='category', values='count').fillna(0)
    df_pivot.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Failure Category Distribution by Model')
    plt.ylabel('Number of Tasks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'failure_category_distribution.png')
    plt.close()

def plot_task_heatmap(df):
    """Create heatmap of task success rates"""
    plt.figure(figsize=(20, 40))
    df_pivot = df.pivot(index='task_id', columns='model', values='success_rate')
    
    sns.heatmap(
        df_pivot,
        annot=True, fmt=".0%",
        cmap='RdYlGn',
        cbar_kws={'label': 'Success Rate'},
        linewidths=0.5
    )
    
    plt.title('Task Success Rates by Model')
    plt.xlabel('Model')
    plt.ylabel('Task ID')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'task_success_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(json_data):
    """Create radar chart of failure category proportions"""
    from math import pi
    
    # Calculate proportions
    radar_data = []
    for model, categories in json_data.items():
        total = sum(len(tasks) for tasks in categories.values())
        model_data = {'model': model}
        for category in ['always_fail', 'high_failure', 'medium_failure', 'low_failure']:
            model_data[category] = len(categories.get(category, [])) / total
        radar_data.append(model_data)
    
    df_radar = pd.DataFrame(radar_data)
    
    # Plot configuration
    categories = list(df_radar.columns[1:])
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each model
    for idx, row in df_radar.iterrows():
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['model'])
        ax.fill(angles, values, alpha=0.1)
    
    plt.title('Failure Category Proportions by Model', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'failure_category_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df, json_data = load_data()
    
    # Generate visualization images
    plot_files = [
        OUTPUT_DIR / 'success_rate_distribution.png',
        OUTPUT_DIR / 'valid_submission_rates.png',
        OUTPUT_DIR / 'failure_category_distribution.png',
        OUTPUT_DIR / 'task_success_heatmap.png',
        OUTPUT_DIR / 'failure_category_radar.png'
    ]
    
    plot_model_performance(df)
    plot_failure_categories(json_data)
    plot_task_heatmap(df)
    plot_radar_chart(json_data)
    
    # Create HTML report
    create_html_report(plot_files)
    
    print(f"Visualizations saved to {OUTPUT_DIR}")
    print(f"HTML report: {REPORT_HTML}")

if __name__ == '__main__':
    main() 