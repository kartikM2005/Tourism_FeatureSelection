import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def generate_plots(results, output_dir):
    """Programmatic entrypoint to generate and save comparative figures dynamically."""
    os.makedirs(output_dir, exist_ok=True)
    
    plot_data = []
    for method, models in results.items():
        for model_name, metrics in models.items():
            plot_data.append({
                'Feature Selection': method,
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'F1 Score': metrics['F1 Score'],
                'Time (s)': metrics['Time (s)']
            })
            
    df_plot = pd.DataFrame(plot_data)
    sns.set_theme(style="whitegrid")
    
    min_acc = df_plot['Accuracy'].min()
    min_f1 = df_plot['F1 Score'].min()
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Feature Selection', y='Accuracy', hue='Model', palette='viridis')
    plt.title('Model Accuracy by Feature Selection Method')
    plt.xticks(rotation=45)
    plt.ylim(max(0.0, min_acc - 0.05), 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # 2. F1 Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Feature Selection', y='F1 Score', hue='Model', palette='magma')
    plt.title('Model F1 Score by Feature Selection Method')
    plt.xticks(rotation=45)
    plt.ylim(max(0.0, min_f1 - 0.05), 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison.png'))
    plt.close()
    
    # 3. Execution Time Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_plot, x='Feature Selection', y='Time (s)', hue='Model', palette='cividis')
    plt.title('Training Time by Feature Selection Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'))
    plt.close()
    
    return {
        'accuracy_plot': 'accuracy_comparison.png',
        'f1_plot': 'f1_comparison.png',
        'time_plot': 'time_comparison.png'
    }

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    results_path = os.path.join(data_dir, "model_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    generate_plots(results, data_dir)
    print("Visualizations generated and saved in the data directory.")

if __name__ == "__main__":
    main()
