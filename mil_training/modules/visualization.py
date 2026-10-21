"""
Visualization utilities for training and evaluation plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def get_color_palette(theme):
    """Get color palette based on theme."""
    palettes = {
        'red': {
            'primary': '#DC143C',      # Crimson
            'secondary': '#B22222',    # FireBrick
            'light': '#FFB6C1',       # LightPink
            'dark': '#8B0000',        # DarkRed
            'accent': '#FF6347'       # Tomato
        },
        'blue': {
            'primary': '#1E90FF',      # DodgerBlue
            'secondary': '#4169E1',    # RoyalBlue
            'light': '#87CEEB',       # SkyBlue
            'dark': '#191970',        # MidnightBlue
            'accent': '#00BFFF'       # DeepSkyBlue
        },
        'green': {
            'primary': '#32CD32',      # LimeGreen
            'secondary': '#228B22',    # ForestGreen
            'light': '#90EE90',       # LightGreen
            'dark': '#006400',        # DarkGreen
            'accent': '#00FF7F'       # SpringGreen
        },
        'purple': {
            'primary': '#8A2BE2',      # BlueViolet
            'secondary': '#9932CC',    # DarkOrchid
            'light': '#DDA0DD',       # Plum
            'dark': '#4B0082',        # Indigo
            'accent': '#BA55D3'       # MediumOrchid
        },
        'orange': {
            'primary': '#FF8C00',      # DarkOrange
            'secondary': '#FF6347',    # Tomato
            'light': '#FFE4B5',       # Moccasin
            'dark': '#FF4500',        # OrangeRed
            'accent': '#FFA500'       # Orange
        }
    }
    return palettes.get(theme, palettes['blue'])


def set_matplotlib_font(font_family='DejaVu Sans'):
    """Set matplotlib font configuration."""
    import matplotlib
    
    # Set the font family
    plt.rcParams['font.family'] = font_family
    
    # If DejaVu Sans is not available, fallback to sans-serif
    try:
        matplotlib.font_manager.findfont(font_family, fallback_to_default=False)
    except:
        print(f"Warning: Font '{font_family}' not found, using default sans-serif")
        plt.rcParams['font.family'] = 'sans-serif'


def style_plot(ax, fontsize=12, font_family='DejaVu Sans', remove_spines=True):
    """Apply consistent styling to plots."""
    # Remove top and right spines
    if remove_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Set font sizes and family
    ax.tick_params(labelsize=fontsize-2)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)
    ax.xaxis.label.set_fontfamily(font_family)
    ax.yaxis.label.set_fontfamily(font_family)
    
    if ax.get_title():
        ax.set_title(ax.get_title(), fontsize=fontsize+2, fontweight='bold', fontfamily=font_family)
    
    # Set font for tick labels
    for label in ax.get_xticklabels():
        label.set_fontfamily(font_family)
    for label in ax.get_yticklabels():
        label.set_fontfamily(font_family)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def get_themed_colormap(theme):
    """Get a colormap based on the theme for heatmaps."""
    theme_cmaps = {
        'red': 'Reds',
        'blue': 'Blues', 
        'green': 'Greens',
        'purple': 'Purples',
        'orange': 'Oranges'
    }
    return theme_cmaps.get(theme, 'Blues')


def plot_confusion_matrix(cm, class_names, title, output_path, normalize='row', 
                          color_theme='blue', fontsize=12):
    """
    Plot and save confusion matrix with row-wise or column-wise percentage normalization.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        output_path: Output file path
        normalize: 'row' (default) or 'col' - whether to normalize by rows or columns
        color_theme: Color theme to use (default: 'blue')
        fontsize: Font size for text (default: 12)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalize == 'row':
        # Row-wise normalization: each row sums to 100%
        # Shows: "Of all actual class X samples, what % were predicted as each class?"
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm, nan=0.0) * 100
        axis_label = 'Predicted'
        y_axis_label = 'Actual'
    elif normalize == 'col':
        # Column-wise normalization: each column sums to 100%
        # Shows: "Of all predicted class X samples, what % were actually from each class?"
        cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        cm_norm = np.nan_to_num(cm_norm, nan=0.0) * 100
        axis_label = 'Predicted'
        y_axis_label = 'Actual'
    else:
        raise ValueError("normalize must be 'row' or 'col'")
    
    # Format annotation as percentage with 1 decimal
    annot = np.empty_like(cm_norm).astype(str)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            annot[i, j] = f"{cm_norm[i, j]:.1f}%"
    
    # Get themed colormap
    cmap = get_themed_colormap(color_theme)
    
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, annot_kws={'fontsize': fontsize-2})
    
    ax.set_title(title, fontsize=fontsize+2, fontweight='bold', pad=20)
    ax.set_xlabel(axis_label, fontsize=fontsize, fontweight='bold')
    ax.set_ylabel(y_axis_label, fontsize=fontsize, fontweight='bold')
    
    # Style tick labels
    ax.tick_params(labelsize=fontsize-2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics_df, title, output_path, color_theme='blue', fontsize=12):
    """Plot per-class precision, recall, and F1-score."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Get color palette
    colors = get_color_palette(color_theme)
    
    # Precision
    bars1 = axes[0].bar(metrics_df.index, metrics_df['precision'], 
                       color=colors['primary'], alpha=0.8, edgecolor=colors['dark'], linewidth=1)
    axes[0].set_title('Precision by Class')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Precision')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylim(0, 1.0)
    for label in axes[0].get_xticklabels():
        label.set_ha('right')
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize-3)
    
    # Recall
    bars2 = axes[1].bar(metrics_df.index, metrics_df['recall'], 
                       color=colors['secondary'], alpha=0.8, edgecolor=colors['dark'], linewidth=1)
    axes[1].set_title('Recall by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Recall')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1.0)
    for label in axes[1].get_xticklabels():
        label.set_ha('right')
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize-3)
    
    # F1-Score
    bars3 = axes[2].bar(metrics_df.index, metrics_df['f1'], 
                       color=colors['accent'], alpha=0.8, edgecolor=colors['dark'], linewidth=1)
    axes[2].set_title('F1-Score by Class')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('F1-Score')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].set_ylim(0, 1.0)
    for label in axes[2].get_xticklabels():
        label.set_ha('right')
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize-3)
    
    # Apply styling to all subplots
    for ax in axes:
        style_plot(ax, fontsize, remove_spines=True)
    
    plt.suptitle(title, fontsize=fontsize+4, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_metrics(train_metrics, val_metrics, metric_name, title, output_path):
    """Plot training and validation metrics over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics, label=f'Training {metric_name}', marker='o')
    plt.plot(val_metrics, label=f'Validation {metric_name}', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, class_names, title, output_path):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True binary labels (0, 1)
        y_proba: Predicted probabilities for the positive class
        class_names: List of class names [negative_class, positive_class]
        title: Plot title
        output_path: Output file path
    
    Returns:
        float: ROC AUC score
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_proba, class_names, title, output_path):
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True binary labels (0, 1)
        y_proba: Predicted probabilities for the positive class
        class_names: List of class names [negative_class, positive_class]
        title: Plot title
        output_path: Output file path
    
    Returns:
        float: Average Precision (AUPRC) score
    """
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)
    
    # Calculate baseline (random classifier performance)
    positive_ratio = np.mean(y_true)
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AP = {average_precision:.3f})')
    plt.axhline(y=positive_ratio, color='navy', lw=2, linestyle='--', 
                label=f'Random classifier (AP = {positive_ratio:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # Add text with positive class information
    plt.text(0.02, 0.95, f'Positive class: {class_names[1]}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.text(0.02, 0.88, f'Prevalence: {positive_ratio:.1%}', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return average_precision
