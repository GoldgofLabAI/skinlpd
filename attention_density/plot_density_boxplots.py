import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from itertools import combinations

plt.rcParams['font.family'] = 'DejaVu Sans'


def parse_args():
    parser = argparse.ArgumentParser(description="Case-level nuclear-density boxplots (Figure 4 E-G)")
    parser.add_argument("--predictions-csv", required=True, help="Slide/case predictions with true_class_case and predicted_class_case")
    parser.add_argument("--density-csv", required=True, help="density_metrics_results.csv from compute_attention_density.py")
    parser.add_argument("--subtype-csv", default=None, help="Optional CSV with slide_name and subtype")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slide-id-column", default="slide_name")
    return parser.parse_args()


args = parse_args()
predictions_data = pd.read_csv(args.predictions_csv)
spatial_data = pd.read_csv(args.density_csv)
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
predictions_data['base_file_name'] = predictions_data['slide_name'].str.replace('.svs', '', regex=False)
joined_data = pd.merge(predictions_data, spatial_data, on='base_file_name')

print(len(joined_data))

# Subtype name mapping (underscored -> polished display names)
SUBTYPE_DISPLAY_NAMES = {
    # Positive subtypes
    "mf_non_folliculotropic": "Non-Folliculotropic MF",
    "mf_folliculotropic": "Folliculotropic MF",
    "cytotoxic_ctcl_not_cd30": "Cytotoxic CTCL\n(Non-CD30)",
    "cd30_tcell_lpd": "CD30+ T-Cell LPD",
    "sezary_syndrome": "Sézary Syndrome",
    "extracutaneous_or_pctcl_nos": "Extracutaneous\n/PTCL-NOS",
    "other_ctcl_tcell": "Other CTCL/CLPD",
    
    # Negative subtypes
    "spongiotic_dermatosis": "Spongiotic Dermatosis",
    "psoriasiform_dermatosis": "Psoriasiform Dermatosis",
    "lichenoid_dermatosis": "Lichenoid Dermatosis",
    "interface_dermatosis": "Interface Dermatosis",
    "panniculitis": "Panniculitis",
    "perivascular_dermatitis": "Perivascular Dermatitis",
    "interstitial_dermatitis": "Interstitial Dermatitis",
    "pityriasiform_dermatosis": "Pityriasiform Dermatosis",
    "depigmentation": "Depigmentation",
    "folliculitis": "Folliculitis",
    "blistering_dermatosis": "Blistering Dermatosis",
    "mucinoses_fibrosing_dermopathy": "Mucinoses/Fibrosing Dermopathy",
    "neutrophilic_dermatosis": "Neutrophilic Dermatosis",
    "infection": "Infection",
    "vasculitis": "Vasculitis",
    "thrombotic_vasculopathy": "Thrombotic Vasculopathy",
    "mast_cell": "Mast Cell",
    "other_inflammatory": "Other Inflammatory",
    "other_dermatosis": "Other Dermatosis"
}

# Desired ordering for subtypes
POSITIVE_SUBTYPE_ORDER = [
    "mf_non_folliculotropic",
    "cd30_tcell_lpd",
    "cytotoxic_ctcl_not_cd30",
    "extracutaneous_or_pctcl_nos",
    "other_ctcl_tcell",
    "mf_folliculotropic",
    "sezary_syndrome",
]

NEGATIVE_SUBTYPE_ORDER = [
    "spongiotic_dermatosis",
    "lichenoid_dermatosis",
    "interface_dermatosis",
    "perivascular_dermatitis",
    "psoriasiform_dermatosis",
    "infection",
    "folliculitis",
    "other_inflammatory",
    "interstitial_dermatitis",
    "pityriasiform_dermatosis",
    "neutrophilic_dermatosis",
    "blistering_dermatosis",
]

# Load subtype data
if args.subtype_csv:
    subtype_data = pd.read_csv(args.subtype_csv)
    slide_col = args.slide_id_column if args.slide_id_column in subtype_data.columns else 'slide_name'
    subtype_data['base_file_name'] = subtype_data[slide_col].astype(str).str.replace('.svs', '', regex=False)
    joined_data = pd.merge(joined_data, subtype_data[['base_file_name', 'subtype']], on='base_file_name', how='left')

# The merge above may create `subtype_x`/`subtype_y` if `joined_data` already had a `subtype` column.
# For the rest of the script we expect a single `subtype` column.
if 'subtype' not in joined_data.columns:
    if 'subtype_x' in joined_data.columns and 'subtype_y' in joined_data.columns:
        joined_data['subtype'] = joined_data['subtype_y'].combine_first(joined_data['subtype_x'])
    elif 'subtype_y' in joined_data.columns:
        joined_data['subtype'] = joined_data['subtype_y']
    elif 'subtype_x' in joined_data.columns:
        joined_data['subtype'] = joined_data['subtype_x']

print(joined_data.columns)

true_positives = joined_data[(joined_data['true_class_case'] == "positive") & (joined_data['predicted_class_case'] == "positive")]
true_negatives = joined_data[(joined_data['true_class_case'] == "negative") & (joined_data['predicted_class_case'] == "negative")]

print(f"True Positives: {len(true_positives)}")
print(f"True Negatives: {len(true_negatives)}")

def add_significance_bracket(ax, pos1, pos2, pvalue, y_pos=None):
    """
    Add significance bracket with asterisks between two groups.
    
    Parameters:
    -----------
    ax : matplotlib axes
    pos1, pos2 : float
        x positions of the two groups
    pvalue : float
        p-value for significance
    y_pos : float, optional
        y position for the bracket. If None, uses 95% of y-axis range
    """
    # Determine significance level and asterisks
    if pvalue < 0.001:
        sig_text = '***'
    elif pvalue < 0.01:
        sig_text = '**'
    elif pvalue < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    # Get y-axis limits
    y_min, y_max = ax.get_ylim()
    if y_pos is None:
        y_pos = y_min + 0.95 * (y_max - y_min)
    
    # Draw horizontal line
    line_height = y_pos + 0.02 * (y_max - y_min)
    ax.plot([pos1, pos2], [line_height, line_height], 'k-', linewidth=1.5)
    
    # Draw vertical lines at ends
    ax.plot([pos1, pos1], [y_pos, line_height], 'k-', linewidth=1.5)
    ax.plot([pos2, pos2], [y_pos, line_height], 'k-', linewidth=1.5)
    
    # Add text
    text_y = line_height + 0.01 * (y_max - y_min)
    ax.text((pos1 + pos2) / 2, text_y, sig_text, 
            ha='center', va='bottom', fontsize=20, fontweight='bold')

# High attention density - boxplot
fig, ax = plt.subplots(figsize=(8, 6))
data_neg = true_negatives['density_high_attn'].dropna().values
data_pos = true_positives['density_high_attn'].dropna().values

# Perform Mann-Whitney U test
stat, pvalue = stats.mannwhitneyu(data_neg, data_pos, alternative='two-sided')
print(f"\nHigh Attention Density - Mann-Whitney U test:")
print(f"  U-statistic: {stat:.4f}, p-value: {pvalue:.4e}")

# Create boxplot with styling matching multi box plots
bp = plt.boxplot(
    [data_neg, data_pos],
    tick_labels=["True Negatives", "True Positives"],
    widths=0.6,
    patch_artist=True,
    showfliers=False
)
# Set face color
for patch in bp['boxes']:
    patch.set_facecolor('#ffb3b3')

plt.ylabel('Nuclear density per\ntissue area (mm$^{-2}$)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 16000)
ax.set_yticks(range(0, 15001, 2500))

# Add significance bracket
if pvalue < 0.05:
    add_significance_bracket(ax, 1, 2, pvalue)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'density_high_attention_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Weighted density - boxplot
fig, ax = plt.subplots(figsize=(10, 6))
data_neg = true_negatives['density_weighted'].dropna().values
data_pos = true_positives['density_weighted'].dropna().values

# Perform Mann-Whitney U test
stat, pvalue = stats.mannwhitneyu(data_neg, data_pos, alternative='two-sided')
print(f"\nWeighted Density - Mann-Whitney U test:")
print(f"  U-statistic: {stat:.4f}, p-value: {pvalue:.4e}")

# Create boxplot with styling matching multi box plots
bp = plt.boxplot(
    [data_neg, data_pos],
    tick_labels=["True Negatives", "True Positives"],
    widths=0.6,
    patch_artist=True,
    showfliers=False
)
# Set face color
for patch in bp['boxes']:
    patch.set_facecolor('#ffb3b3')

plt.ylabel('Nuclear density per\ntissue area (mm$^{-2}$)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([0, 16000])

# Add significance bracket
if pvalue < 0.05:
    add_significance_bracket(ax, 1, 2, pvalue)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'density_weighted_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Violin plot versions
# High attention density - violinplot
fig, ax = plt.subplots(figsize=(10, 6))
data_neg = true_negatives['density_high_attn'].dropna().values
data_pos = true_positives['density_high_attn'].dropna().values

# Perform Mann-Whitney U test
stat, pvalue = stats.mannwhitneyu(data_neg, data_pos, alternative='two-sided')
print(f"\nHigh Attention Density (Violin) - Mann-Whitney U test:")
print(f"  U-statistic: {stat:.4f}, p-value: {pvalue:.4e}")

vp = plt.violinplot(
    [data_neg, data_pos],
    positions=[1, 2],
    widths=0.6,
    showmeans=True,
    showmedians=True
)
# Set face color and alpha
for pc in vp['bodies']:
    pc.set_facecolor('#ffb3b3')
    pc.set_alpha(0.7)

plt.xticks([1, 2], ["True Negatives", "True Positives"], fontsize=20)
plt.ylabel('Nuclear density per\ntissue area (mm$^{-2}$)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([0, 16000])
plt.yticks(range(0, 15001, 2500))


# Add significance bracket
if pvalue < 0.05:
    add_significance_bracket(ax, 1, 2, pvalue)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'density_high_attention_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Weighted density - violinplot
fig, ax = plt.subplots(figsize=(10, 6))
data_neg = true_negatives['density_weighted'].dropna().values
data_pos = true_positives['density_weighted'].dropna().values

# Perform Mann-Whitney U test
stat, pvalue = stats.mannwhitneyu(data_neg, data_pos, alternative='two-sided')
print(f"\nWeighted Density (Violin) - Mann-Whitney U test:")
print(f"  U-statistic: {stat:.4f}, p-value: {pvalue:.4e}")

vp = plt.violinplot(
    [data_neg, data_pos],
    positions=[1, 2],
    widths=0.6,
    showmeans=True,
    showmedians=True
)
# Set face color and alpha
for pc in vp['bodies']:
    pc.set_facecolor('#ffb3b3')
    pc.set_alpha(0.7)

plt.xticks([1, 2], ["True Negatives", "True Positives"], fontsize=20)
plt.ylabel('Nuclear density per\ntissue area (mm$^{-2}$)', fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.ylim([0, 16000])
plt.yticks(range(0, 15001, 2500))

# Add significance bracket
if pvalue < 0.05:
    add_significance_bracket(ax, 1, 2, pvalue)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'density_weighted_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Subtype comparison plots
# Filter to only true positives and true negatives with subtype information
true_positives_subtype = true_positives[true_positives['subtype'].notna()].copy()
true_negatives_subtype = true_negatives[true_negatives['subtype'].notna()].copy()

print(f"\nTrue Positives with subtype: {len(true_positives_subtype)}")
print(f"True Negatives with subtype: {len(true_negatives_subtype)}")

# Get unique subtypes for positive and negative
positive_subtypes = sorted(true_positives_subtype['subtype'].unique())
negative_subtypes = sorted(true_negatives_subtype['subtype'].unique())

print(f"\nPositive subtypes: {positive_subtypes}")
print(f"Negative subtypes: {negative_subtypes}")

def generate_compact_letter_display(labels, data_list, pvalues_matrix, alpha=0.05):
    """
    Generate compact letter display from pairwise comparison matrix.
    Groups that share letters are not significantly different.
    Uses the algorithm from Piepho (2004) - An Algorithm for a Letter-Based 
    Representation of All-Pairwise Comparisons.
    """
    n = len(labels)
    
    # Calculate medians for sorting
    medians = [np.median(data) for data in data_list]
    
    # Sort groups by median (ascending)
    sorted_indices = np.argsort(medians)
    
    # Create adjacency matrix (True if NOT significantly different)
    not_sig_diff = np.ones((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and pvalues_matrix[i, j] < alpha:
                not_sig_diff[i, j] = False
    
    # Initialize letter assignments
    letters = [set() for _ in range(n)]
    available_letters = 'abcdefghijklmnopqrstuvwxyz'
    current_letter = 0
    
    # Process groups in order of increasing median
    for idx in sorted_indices:
        if len(letters[idx]) == 0:  # If group doesn't have a letter yet
            # Assign current letter to this group
            letters[idx].add(available_letters[current_letter])
            
            # Find all groups not significantly different from this one
            # that come after it in the sorted order and don't have this letter
            for other_idx in sorted_indices:
                if other_idx != idx and not_sig_diff[idx, other_idx]:
                    # Check if this letter can be added (must be compatible with all groups that have it)
                    can_add = True
                    for check_idx in range(n):
                        if available_letters[current_letter] in letters[check_idx]:
                            if not not_sig_diff[other_idx, check_idx]:
                                can_add = False
                                break
                    if can_add:
                        letters[other_idx].add(available_letters[current_letter])
            
            current_letter += 1
    
    # Convert sets to sorted strings
    letter_strings = [''.join(sorted(s)) for s in letters]
    
    return letter_strings


def plot_single_group_subtypes(data, subtypes_list, metric_col, metric_name, 
                               group_name, color, plot_type='boxplot', show_outliers=False):
    """
    Create boxplot or violin plot for a single group (positive or negative) with statistical tests
    
    Parameters:
    -----------
    show_outliers : bool, default=False
        Whether to show outlier points in boxplots
    """
    # Determine which ordering to use based on group name
    if 'Positive' in group_name:
        desired_order = POSITIVE_SUBTYPE_ORDER
    else:
        desired_order = NEGATIVE_SUBTYPE_ORDER
    
    # Get subtypes that exist in the data, in the desired order
    available_subtypes = set(subtypes_list)
    ordered_subtypes = [st for st in desired_order if st in available_subtypes]
    
    # Add any subtypes not in the desired order at the end
    remaining_subtypes = [st for st in subtypes_list if st not in ordered_subtypes]
    ordered_subtypes.extend(remaining_subtypes)
    
    # Prepare data
    subtype_data = []
    subtype_labels = []
    subtype_labels_display = []
    
    for st in ordered_subtypes:
        st_values = data[data['subtype'] == st][metric_col].dropna().values
        if len(st_values) > 0:
            subtype_data.append(st_values)
            subtype_labels.append(st)
            # Use display name if available, otherwise use original
            subtype_labels_display.append(SUBTYPE_DISPLAY_NAMES.get(st, st))
    
    if len(subtype_data) < 2:
        print(f"Not enough subtypes for {group_name} (need at least 2)")
        return
    
    n_subtypes = len(subtype_data)
    
    # Perform Kruskal-Wallis test
    kw_stat, kw_pvalue = stats.kruskal(*subtype_data)
    print(f"\n{group_name} - Kruskal-Wallis test:")
    print(f"  H-statistic: {kw_stat:.4f}, p-value: {kw_pvalue:.4e}")
    
    # Initialize p-value matrix and letter display
    pvalues_matrix = np.ones((n_subtypes, n_subtypes))
    compact_letters = None
    
    # If Kruskal-Wallis is significant, perform post-hoc tests
    if kw_pvalue < 0.05:
        print(f"  Kruskal-Wallis is significant, performing post-hoc pairwise tests...")
        
        # Perform all pairwise Mann-Whitney U tests
        pairwise_pvalues = []
        pairs = []
        
        for i, j in combinations(range(n_subtypes), 2):
            stat, pval = stats.mannwhitneyu(subtype_data[i], subtype_data[j], 
                                           alternative='two-sided')
            pairwise_pvalues.append(pval)
            pairs.append((i, j))
            pvalues_matrix[i, j] = pval
            pvalues_matrix[j, i] = pval
        
        # Apply Bonferroni correction
        n_comparisons = len(pairwise_pvalues)
        bonferroni_alpha = 0.05 / n_comparisons
        corrected_pvalues = [p * n_comparisons for p in pairwise_pvalues]
        
        print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.4e}")
        print(f"  Significant pairs (after Bonferroni correction):")
        
        # Update matrix with corrected p-values
        for idx, (i, j) in enumerate(pairs):
            pvalues_matrix[i, j] = min(corrected_pvalues[idx], 1.0)
            pvalues_matrix[j, i] = min(corrected_pvalues[idx], 1.0)
            if corrected_pvalues[idx] < 0.05:
                print(f"    {subtype_labels_display[i]} vs {subtype_labels_display[j]}: "
                      f"p={pairwise_pvalues[idx]:.4e}, corrected p={corrected_pvalues[idx]:.4e}")
        
        # Generate compact letter display
        compact_letters = generate_compact_letter_display(subtype_labels, subtype_data, pvalues_matrix)
        print(f"  Compact letter display:")
        for label, display, letter in zip(subtype_labels, subtype_labels_display, compact_letters):
            print(f"    {display}: {letter}")
    else:
        print(f"  Kruskal-Wallis is not significant, skipping post-hoc tests")
        compact_letters = ['a'] * n_subtypes  # All groups get same letter
    
    # Create plot
    fig_width = max(10, n_subtypes * 1.2)
    fig = plt.figure(figsize=(fig_width, 8))
    ax = fig.gca()
    
    # Set larger font sizes globally for this plot
    plt.rcParams.update({'font.size': 20})
    
    positions = list(range(1, n_subtypes + 1))
    group_tops = None
    
    if plot_type == 'boxplot':
        # Control outlier display
        flierprops = dict(marker='o', markersize=3, alpha=0.5) if show_outliers else dict(marker='', markersize=0)
        bp = plt.boxplot(subtype_data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=show_outliers, flierprops=flierprops)
        for patch in bp['boxes']:
            patch.set_facecolor(color)

        # Upper whisker cap y-position for each group.
        # `bp['caps']` has 2 entries per box: lower and upper caps.
        group_tops = []
        caps = bp.get('caps', [])
        for i in range(n_subtypes):
            cap_pair = caps[2 * i : 2 * i + 2]
            if len(cap_pair) != 2:
                group_tops.append(np.nan)
                continue
            y0 = cap_pair[0].get_ydata()
            y1 = cap_pair[1].get_ydata()
            cap_upper = float(max(np.nanmax(y0), np.nanmax(y1)))
            group_tops.append(cap_upper)
    else:
        vp = plt.violinplot(subtype_data, positions=positions, widths=0.6,
                           showmeans=True, showmedians=True)
        for pc in vp['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # For violins there is no whisker cap; use observed max.
        group_tops = [np.nanmax(vals) if len(vals) else np.nan for vals in subtype_data]
    
    # Set labels (use display names)
    plt.xticks(positions, subtype_labels_display, rotation=45, ha='right', fontsize=20)
    plt.ylabel(metric_name, fontsize=20)
    
    # # Add title with test results
    # title = f'{metric_name} by Subtype: {group_name} ({plot_type.capitalize()})'
    # title += f'\nKruskal-Wallis: H={kw_stat:.2f}, p={kw_pvalue:.4e}'
    # plt.title(title, fontsize=20)
    
    # Increase tick label size
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limit
    plt.ylim([0, 16000])
    plt.yticks(range(0, 15001, 2500))
    
    # Add compact letter display if available
    if compact_letters:
        y_min, y_max = plt.ylim()
        y_range = y_max - y_min

        # Constant gap per plot (absolute units), avoiding any per-box IQR math.
        gap = max(0.03 * y_range, 200)
        y_floor = y_min + 0.01 * y_range
        y_ceiling = y_max - 0.01 * y_range

        for pos, letter, top in zip(positions, compact_letters, group_tops):
            if top is None or np.isnan(top):
                continue
            y_text = min(top + gap, y_ceiling)
            y_text = max(y_text, y_floor)
            plt.text(pos, y_text, letter, ha='center', va='bottom',
                     fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    return fig_width

# Create separate plots for positive and negative subtypes with statistical tests

# Positive subtypes - density_high_attn - boxplot
plot_single_group_subtypes(
    true_positives_subtype, positive_subtypes,
    'density_high_attn', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Positives', '#ffb3b3', 'boxplot'
)
plt.savefig(os.path.join(output_dir, 'density_high_attention_positive_subtype_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Negative subtypes - density_high_attn - boxplot
plot_single_group_subtypes(
    true_negatives_subtype, negative_subtypes,
    'density_high_attn', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Negatives', '#ffb3b3', 'boxplot'
)
plt.savefig(os.path.join(output_dir, 'density_high_attention_negative_subtype_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Positive subtypes - density_high_attn - violinplot
plot_single_group_subtypes(
    true_positives_subtype, positive_subtypes,
    'density_high_attn', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Positives', '#ffb3b3', 'violinplot'
)
plt.savefig(os.path.join(output_dir, 'density_high_attention_positive_subtype_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Negative subtypes - density_high_attn - violinplot
plot_single_group_subtypes(
    true_negatives_subtype, negative_subtypes,
    'density_high_attn', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Negatives', '#ffb3b3', 'violinplot'
)
plt.savefig(os.path.join(output_dir, 'density_high_attention_negative_subtype_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Positive subtypes - density_weighted - boxplot
plot_single_group_subtypes(
    true_positives_subtype, positive_subtypes,
    'density_weighted', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Positives', '#ffb3b3', 'boxplot'
)
plt.savefig(os.path.join(output_dir, 'density_weighted_positive_subtype_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Negative subtypes - density_weighted - boxplot
plot_single_group_subtypes(
    true_negatives_subtype, negative_subtypes,
    'density_weighted', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Negatives', '#ffb3b3', 'boxplot'
)
plt.savefig(os.path.join(output_dir, 'density_weighted_negative_subtype_boxplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Positive subtypes - density_weighted - violinplot
plot_single_group_subtypes(
    true_positives_subtype, positive_subtypes,
    'density_weighted', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Positives', '#ffb3b3', 'violinplot'
)
plt.savefig(os.path.join(output_dir, 'density_weighted_positive_subtype_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()

# Negative subtypes - density_weighted - violinplot
plot_single_group_subtypes(
    true_negatives_subtype, negative_subtypes,
    'density_weighted', 'Nuclear density per\ntissue area (mm$^{-2}$)',
    'True Negatives', '#ffb3b3', 'violinplot'
)
plt.savefig(os.path.join(output_dir, 'density_weighted_negative_subtype_violinplot.png'), 
            dpi=150, bbox_inches='tight')
plt.close()