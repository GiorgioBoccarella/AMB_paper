import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

which_clade = 5

clades = {
    1: {"KG": 0.219692291518742, "MIC_S": 1},
    3: {"KG": 0.120284472246658, "MIC_S": 1},
    4: {"KG": 0.192553737889521, "MIC_S": 2},
    5: {"KG": 0.176254143936869, "MIC_S": 2}
}

clade = clades[which_clade]

# Drug and population parameters
Bmax = 9 # Maximum bacterial carying capacity
KG = clade["KG"] # Growth rate, clade specific
Gmin_A = -3 # Minimum growth rate for antibiotic
HILL_A = 4 # Hill coefficient for antibiotic
MIC_S = clade["MIC_S"] # MIC for susceptible cells
S0 = 10**6 # Initial susceptible cells
RA0 = 1000 # Initial resistant cells
A0 = 1.5 # Initial antibiotic concentration

# Parameters for 
length_experiment = 24 * 8
t = np.linspace(0, length_experiment, length_experiment)

data = {
    "5": [
        {"strain": "WT", "RR": 1, "GR": 1},
        {"strain": "31", "RR": 2, "GR": 0.7177506931368},
        {"strain": "32", "RR": 2, "GR": 0.800254783268322},
        {"strain": "ERG11", "RR": 2, "GR": 0.661989563202046},
    ],
}

fit_range = np.linspace(0.1, 1.1, 64)
MIC_R_range = np.linspace(1, 5, 32)

def antibiotic_decay(A, t_half):
    ke = np.log(2) / t_half
    return -ke * A

def model(y, t, params, f_A, E_max):
    S, RA, A, E = y
    A = f_A
    Bmax, KG, FIT, Gmin_A, HILL_A, MIC_S, MIC_R = params

    # All immune parameters
    q = 3  # Immune speed of response
    l = 10**-3 # Immune decay rate
    jN = 10**-6 # Immune killing rate
    
    MIC_R = MIC_R * MIC_S # MIC for resistant cells

    # Subpopulations
    dSdt = S*(1-((S+RA)/10**Bmax))*KG*(1- ((1 - Gmin_A/KG)*(A/MIC_S)**HILL_A/((A/MIC_S)**HILL_A - (Gmin_A/KG)))) - jN*S*E
    dRAdt = RA*(1-((S+RA)/10**Bmax))*KG*FIT*(1- ((1 - Gmin_A/(KG*FIT))*(A/MIC_R)**HILL_A/((A/MIC_R)**HILL_A - (Gmin_A/(KG*FIT))))) - jN*RA*E
    dAdt = f_A
    dEdt = q*(E_max - E) - l*E

    return [dSdt, dRAdt, dAdt, dEdt]


fig, ax = plt.subplots(figsize=(11, 8))

E_max_values = [4*10**4, 10**5, 4*10**5] # NP-2, NP-1, NP-3, respectively
titles = ['Extinction boundary NP-2', 'Extinction boundary NP-1', 'Extinction boundary NP-3']
fmt_dict = {4*10**4: 'NP-1', 10**5: 'NP-2', 4*10**5: 'NP-3'}

for idx, E_max in enumerate(E_max_values):
    num_resistant_cells = np.zeros((len(fit_range), len(MIC_R_range)))
    for i in range(len(fit_range)):
        for j in range(len(MIC_R_range)):
            y0 = [S0, RA0, A0, 10**2]
            solution = [y0]
            A_values = [A0]
            dt = t[1] - t[0]

            params = [Bmax, KG, fit_range[i], Gmin_A, HILL_A, MIC_S, MIC_R_range[j]]

            A = A0
            t_half = 24

            for k in range(1, length_experiment):
                if k == 1:
                    A = A0
                elif k % 24 == 0:
                    A += A0
                else:
                    dA = antibiotic_decay(A, t_half)
                    A += dA
                ts = [t[k-1], t[k]]
                y = odeint(model, y0, ts, args=(params, A, E_max))
                if y[1][1] < 1:
                    y[1][1] = 0
                if y[1][0] < 1:
                    y[1][0] = 0
                y0 = y[1]
                solution.append(y0)
                A_values.append(A)

            solution = np.array(solution)
            A_values = np.array(A_values)

            solution[solution < 1] = 0
            if solution[-1][1] == 0:
                num_resistant_cells[i, j] = 0
            else:
                num_resistant_cells[i, j] = np.log10(solution[-1][1])

    if E_max == 10**5:  # Color mapping for NP-1
        cmap = plt.cm.get_cmap('viridis').copy()
        cmap.set_under(color=(232/255, 236/255, 241/255, 0.3))
        cax = ax.contourf(MIC_R_range, fit_range, num_resistant_cells, cmap=cmap, vmin=3, vmax=9, alpha=0.8)
        cbar = fig.colorbar(cax, ax=ax, ticks=[0, 3, 4.5, 6, 7.5, 9])
        cbar.set_label('Log10 (Number of cells)', fontsize=26)
        cbar.ax.tick_params(labelsize=24)
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        # draw extinction boundary for NP-1
        extinction_boundary = ax.contour(MIC_R_range, fit_range, num_resistant_cells, levels=[3], colors='black', linewidths=2.5, alpha=0.8, linestyles='dashed')
        if extinction_boundary.collections:
            ax.clabel(extinction_boundary, inline=True, fontsize=20, fmt={3: fmt_dict[E_max]})
    else:  # Draw line boundary for NP-2 and NP-3
        extinction_boundary = ax.contour(MIC_R_range, fit_range, num_resistant_cells, levels=[3], colors='black', linewidths=2.5, alpha=0.8, linestyles='dashed')
        if extinction_boundary.collections:
            ax.clabel(extinction_boundary, inline=True, fontsize=20, fmt={3: fmt_dict[E_max]})

ax.set_title(f'Clade V', fontsize=37)
ax.set_xlabel('RR (MIC)', fontsize=35)
ax.set_ylabel('GR(µ max)', fontsize=35)

texts = []

for i, strain_data in enumerate(data[str(which_clade)]):
    x = strain_data['RR']
    y = strain_data['GR']
    label = strain_data['strain']
    if label == "ERG11":
        ax.scatter(x, y, color='black', zorder=10, s=120, marker='s', alpha=0.7)
    else:
        ax.scatter(x, y, color='black', zorder=10, s=120, alpha=0.8)

plt.tight_layout()
plt.savefig(f'main_plot/clade_{which_clade}.svg', dpi=700)
plt.savefig(f'main_plot/clade_{which_clade}.png', dpi=700)