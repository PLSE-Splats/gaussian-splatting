import pandas as pd
import matplotlib.pyplot as plt
import io

SKM_FILE_PATH = "../scene/skm_grads_stats.csv"
GS_FILE_PATH = "../../gs-splats/scene/gs_grads_stats.csv"

def plot_count_and_ratio(df_gs, df_skm):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot total counts on the primary y-axis (ax1)
    ax1.plot(df_gs.index, df_gs['total_count'], 'b-', label='Total Count (3dgs)')
    ax1.plot(df_skm.index, df_skm['total_count'], 'r-', label='Total Count (skm)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Total Count', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.grid(True)

    # Create a secondary y-axis for the percentages
    ax2 = ax1.twinx()
    ax2.plot(df_gs.index, df_gs['high_count_ratio'], 'c--', label='Percentage (3dgs)')
    ax2.plot(df_skm.index, df_skm['high_count_ratio'], 'm--', label='Percentage (skm)')
    ax2.set_ylabel('Percentage', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    plt.title('Total Count and Percentage Over Time')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.savefig('comparison_plot.png')
    print("The plot has been generated and saved as 'comparison_plot.png'")
    
    
def plot_mean_and_variance(df_gs, df_skm):
    # Create the plot with a secondary y-axis for variance
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot mean on the primary y-axis (ax1)
    ax1.plot(df_gs.index, df_gs['mean'], color='orangered', linestyle='-', label='Mean (3dgs)')
    ax1.fill_between(df_gs.index, df_gs['mean'] - df_gs['variance'], df_gs['mean'] + df_gs['variance'],
                color='orangered', alpha=0.2, label='Mean ± Var (3dgs)')
    ax1.plot(df_skm.index, df_skm['mean'], color='dodgerblue', linestyle='-', label='Mean (skm)')
    ax1.fill_between(df_skm.index, df_skm['mean'] - df_skm['variance'], df_skm['mean'] + df_skm['variance'],
                     color='dodgerblue', alpha=0.2, label='Mean ± Var (skm)')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Mean', color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.grid(True)
    
    GS_MEAN_THRESHOLD = 0.0002
    ax1.axhline(y=GS_MEAN_THRESHOLD, color='red', linestyle=':', linewidth=2, label=f'Mean Threshold ({GS_MEAN_THRESHOLD:.5f})')

    SKM_MEAN_THRESHOLD = 0.0008
    ax1.axhline(y=SKM_MEAN_THRESHOLD, color='orange', linestyle=':', linewidth=2, label=f'Mean Threshold ({SKM_MEAN_THRESHOLD:.5f})')

    plt.title('Mean with Variance Band Over Time')
    plt.legend(loc='upper left')

    plt.savefig('mean_with_variance_band_plot.png')

    print("The plot has been generated and saved as 'mean_with_variance_band_plot.png'")
        
if __name__ == "__main__":
    # Column names based on the user-provided script
    columns = ['total_count', 'high_grad_count', 'high_count_ratio', 'mean', 'variance', 'max']

    # Create DataFrames for both datasets
    df_gs = pd.read_csv(GS_FILE_PATH, header=None, names=columns)
    df_skm = pd.read_csv(SKM_FILE_PATH, header=None, names=columns)
    plot_count_and_ratio(df_gs, df_skm)
    plot_mean_and_variance(df_gs, df_skm)