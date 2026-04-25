import pickle
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def generate_bar_chart():
    try:
        full_data = load_data('full_precision_data.pkl')
        peft_data = load_data('peft_lora_data.pkl')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return
    
    # Extract final metrics
    full_time = full_data['time'][-1]
    full_acc = full_data['improvement'][-1]
    
    peft_time = peft_data['time'][-1]
    peft_acc = peft_data['improvement'][-1]

    # Create a side-by-side bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    labels = ['Full Precision Base', 'PEFT + LoRA']
    colors = ['#1f77b4', '#ff7f0e']

    # Subplot 1: Time Comparison
    bars1 = ax1.bar(labels, [full_time, peft_time], color=colors, alpha=0.85, edgecolor='black')
    ax1.set_title('Total Training Time (Lower is Better)', pad=10)
    ax1.set_ylabel('Days')
    ax1.set_ylim(0, max(full_time, peft_time) * 1.2)
    ax1.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.1f} Days", ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Accuracy Comparison
    bars2 = ax2.bar(labels, [full_acc, peft_acc], color=colors, alpha=0.85, edgecolor='black')
    ax2.set_title('Final Accuracy (Higher is Better)', pad=10)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, max(full_acc, peft_acc) * 1.2)
    ax2.grid(axis='y', linestyle=':', alpha=0.7)
    
    # Add values on top of bars
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold')
        
    plt.suptitle('Performance Trade-off Summary', fontsize=16, y=1.05, fontweight='bold')
    plt.tight_layout()

    output_filename = 'training_bar_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved successfully as '{output_filename}'")

if __name__ == '__main__':
    generate_bar_chart()
