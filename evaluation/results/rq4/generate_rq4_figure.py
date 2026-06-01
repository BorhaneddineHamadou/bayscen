import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 4.5,
    'pdf.fonttype': 42,
})

agg  = pd.read_excel('failure_characterization.xlsx', sheet_name='AggregatedByRun')
path = pd.read_excel('failure_characterization.xlsx', sheet_name='Conflict_Geometry')

VARS    = ['Sensor_Perception', 'Surface_Traction', 'Lateral_Stability']
VTITLES = ['Sensor Perception ($a_{perc}$)',
           'Surface Traction ($a_{trac}$)',
           'Lateral Stability ($a_{stab}$)']
SUTS   = ['Interfuser', 'Modular']
SCENS  = [1, 2]
SNAMES = ['S1: Vehicle–Vehicle', 'S2: Vehicle–Cyclist']

C_IF  = '#2166ac'
C_MOD = '#d6604d'

def get_line(var, sut, scen, baseline):
    sub = agg[(agg.abstract_var == var) & (agg.sut == sut) &
              (agg.scenario == scen) & (agg.baseline == baseline)].copy()
    sub = sub[sub['level'].apply(lambda x: isinstance(x, (int, float, np.integer)))]
    sub['level'] = sub['level'].astype(int)
    sub = sub[sub['level'] <= 4].sort_values('level')
    return (sub['level'].values,
            sub['failure_rate_mean_pct'].values,
            sub['failure_rate_std_pct'].values)

def row_ymax(scen):
    vals = []
    for var in VARS:
        for sut in SUTS:
            for bl in ['BayScen', 'BayScen-Common']:
                _, fr, sd = get_line(var, sut, scen, bl)
                if len(fr):
                    vals.extend((fr + sd).tolist())
    return np.ceil(max(vals) / 5) * 5 + 3

def path_ymax():
    vals = []
    for scen in SCENS:
        for sut in SUTS:
            sub = path[(path.sut == sut) & (path.scenario == scen) &
                       (path.baseline == 'BayScen')]
            for g in ['c1', 'c2', 'c4']:
                row = sub[sub['level'] == g]
                if len(row):
                    v = row['failure_rate_mean_pct'].values[0]
                    e = row['failure_rate_std_pct'].values[0]
                    vals.append(v + e)
    return np.ceil(max(vals) / 5) * 5 + 4

ymax_row  = {scen: row_ymax(scen) for scen in SCENS}
ymax_path = path_ymax()   # shared y-scale for both path panels

# ── Layout: 2 rows × 4 cols; col 3 = conflict geometry, one panel per row
fig = plt.figure(figsize=(10.0, 4.8))
gs = gridspec.GridSpec(
    2, 4,
    width_ratios=[1, 1, 1, 0.75],
    left=0.065, right=0.985,
    top=0.88, bottom=0.18,
    hspace=0.55, wspace=0.40,
)

xticks      = [0, 1, 2, 3, 4]
xticklabels = ['0\n(clear)', '1', '2', '3', '4\n(severe)']

geom_keys   = ['c1', 'c2', 'c4']
geom_labels = ['$g_1$', '$g_2$', '$g_3$']
x_bar = np.arange(3)
w     = 0.20

for ri, scen in enumerate(SCENS):

    # ── Line plots: cols 0–2
    for ci, (var, vtitle) in enumerate(zip(VARS, VTITLES)):
        ax = fig.add_subplot(gs[ri, ci])

        for sut, col, mk in [(SUTS[0], C_IF, 'o'), (SUTS[1], C_MOD, 's')]:
            lvl, fr, sd = get_line(var, sut, scen, 'BayScen')
            ax.plot(lvl, fr, color=col, marker=mk, ls='-', zorder=4)
            ax.fill_between(lvl, np.maximum(fr - sd, 0), fr + sd,
                            color=col, alpha=0.12, zorder=2)

            lvl_c, fr_c, _ = get_line(var, sut, scen, 'BayScen-Common')
            if len(lvl_c):
                ax.plot(lvl_c, fr_c, color=col, marker=mk,
                        ls='--', lw=1.1, alpha=0.55, zorder=3)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim(-0.35, 4.35)
        ax.set_ylim(0, ymax_row[scen])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g%%'))
        ax.grid(axis='y', linewidth=0.35, alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ri == 0:
            ax.set_title(vtitle, fontsize=8, fontweight='bold', pad=5)

        if ci == 0:
            ax.set_ylabel('Collision rate', labelpad=3)
            ax.text(0.03, 0.97, SNAMES[ri],
                    transform=ax.transAxes, fontsize=7.5,
                    fontweight='bold', va='top', ha='left', color='#222222')

        if ri == 1:
            ax.set_xlabel('Degradation level', labelpad=2)

    # ── Conflict geometry: col 3, one per row
    ax_p = fig.add_subplot(gs[ri, 3])

    for si, sut in enumerate(SUTS):
        sub = path[(path.sut == sut) & (path.scenario == scen) &
                   (path.baseline == 'BayScen')]
        vals = []
        errs = []
        for g in geom_keys:
            row = sub[sub['level'] == g]
            vals.append(row['failure_rate_mean_pct'].values[0] if len(row) else 0)
            errs.append(row['failure_rate_std_pct'].values[0]  if len(row) else 0)

        col    = C_IF if sut == SUTS[0] else C_MOD
        offset = (si - 0.5) * w
        ax_p.bar(x_bar + offset, vals, w,
                 color=col, alpha=0.9,
                 edgecolor='white', linewidth=0.3,
                 zorder=3,
                 yerr=errs, capsize=2,
                 error_kw=dict(linewidth=0.6))
        for xi, v in zip(x_bar + offset, vals):
            if v >= 6:
                ax_p.text(xi, v + 0.8, f'{v:.0f}',
                          ha='center', va='bottom', fontsize=5.8,
                          color='#222222')

    ax_p.set_xticks(x_bar)
    ax_p.set_xticklabels(geom_labels, fontsize=9.5)
    ax_p.set_ylim(0, ymax_path)
    ax_p.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g%%'))
    ax_p.grid(axis='y', linewidth=0.35, alpha=0.5, zorder=0)
    ax_p.set_axisbelow(True)
    ax_p.spines['top'].set_visible(False)
    ax_p.spines['right'].set_visible(False)

    if ri == 0:
        ax_p.set_title('Conflict Geometry\n(BayScen)', fontsize=8,
                       fontweight='bold', pad=5)
        ax_p.set_ylabel('Collision rate (%)', labelpad=3)
    else:
        ax_p.set_xlabel('Conflict geometry', labelpad=3)
        ax_p.set_ylabel('Collision rate (%)', labelpad=3)

    # Scenario tag inside bar panel
    ax_p.text(0.97, 0.97, SNAMES[ri],
              transform=ax_p.transAxes, fontsize=7,
              fontweight='bold', va='top', ha='right', color='#222222')

# ── Shared legend
legend_handles = [
    Line2D([0],[0], color=C_IF,  marker='o', ls='-',  lw=1.5, ms=4.5,
           label='InterFuser — BayScen'),
    Line2D([0],[0], color=C_IF,  marker='o', ls='--', lw=1.1, ms=4.5,
           alpha=0.6, label='InterFuser — BayScen-Common'),
    Line2D([0],[0], color=C_MOD, marker='s', ls='-',  lw=1.5, ms=4.5,
           label='Modular — BayScen'),
    Line2D([0],[0], color=C_MOD, marker='s', ls='--', lw=1.1, ms=4.5,
           alpha=0.6, label='Modular — BayScen-Common'),
]
fig.legend(handles=legend_handles,
           loc='lower center', bbox_to_anchor=(0.46, -0.01),
           ncol=4, frameon=False, fontsize=7,
           columnspacing=1.4, handlelength=2.0, handletextpad=0.5)

plt.savefig('rq4_failure_characterization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('rq4_failure_characterization.png', dpi=300, bbox_inches='tight')
print("Saved.")