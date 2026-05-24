import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. ENTER YOUR BENCHMARK RESULTS HERE
# =====================================================================
# Replace these placeholder numbers with your actual script outputs
metrics = [
    'TTFT\n(Seconds)\n[Lower is Better]', 
    'Throughput\n(Tokens/Sec)\n[Higher is Better]', 
    'Peak VRAM\n(GB)\n[Lower is Better]', 
    'Perplexity\n(PPL Score)\n[Lower is Better]'
]

cold_run_data = [185.0378, 0.12, 8.72, 1.08]
hot_run_data  = [0.6829,   1.91, 7.45, 22.75]

# =====================================================================
# 2. GRAPH GENERATION CODE
# =====================================================================
x = np.arange(len(metrics))
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Create side-by-side bars
rects1 = ax.bar(x - width/2, cold_run_data, width, label='Cold Run (Uncached)', color='#e74c3c')
rects2 = ax.bar(x + width/2, hot_run_data, width, label='Hot Run (Cached)', color='#2ecc71')

# Add labels, title, and styling
ax.set_ylabel('Measured values (Log/Mixed Scale Notice)', fontsize=11, fontweight='bold')
ax.set_title('Gemma 2 2B Caching Benchmark: Cold vs. Hot Performance', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Add a warning note on metric scaling for hackathon presentation
plt.figtext(0.15, 0.01, "*Note: Metrics use different unit systems. Avoid comparing absolute bar heights across different categories.", 
            fontsize=9, style='italic', color='#555555')

plt.tight_layout()

# Save the file to your local directory
output_filename = "gemma2_cache_benchmark.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Success! Your benchmark graph has been saved as: {output_filename}")
plt.show()
