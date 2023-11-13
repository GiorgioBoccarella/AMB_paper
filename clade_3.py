import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from adjustText import adjust_text

data = {
    "3": [
        {"strain": "10 (WT)", "RR": 1, "GR": 1},
        #{"strain": "11", "RR": 8, "GR": 0.558140336903881},
        {"strain": "12", "RR": 4, "GR": 0.659616797346785},
        #{"strain": "13", "RR": 8, "GR": 0.880740733915009},
        {"strain": "14", "RR": 2, "GR": 0.640054869512626},
        {"strain": "15", "RR": 4, "GR": 1.06653052789603},
        {"strain": "16", "RR": 2, "GR": 0.789895201102259},
        {"strain": "17", "RR": 4, "GR": 0.999246852192624},
        {"strain": "ERG11", "RR": 2.3, "GR": 0.67},
        {"strain": "ERG3Δ", "RR": 2, "GR": 0.830729324414057},
        {"strain": "ERG6Δ", "RR": 4, "GR": 0.830393583316409},
        {"strain": "NCP1Δ", "RR": 2, "GR": 0.769941284871208},
    ],
}



fit_range = np.linspace(0.1, 1.1, 64)
MIC_R_range = np.linspace(1, 5, 32)
#MIC_R_range2 = np.linspace(11, 64, 1)
#MIC_R_range = np.concatenate(MIC_R_range1)

def antibiotic_decay(A, t_half):
    ke = np.log(2) / t_half
    return -ke * A

def model(y, t, params, f_A, e_max):
    S, RA, A, E = y
    A = f_A
    Bmax, KG, FIT, V, Gmax, Gmin_A, HILL_A,  MIC_S, MIC_R = params

    # all immune parameters from Ankomah and Levin (2014).
    q = 3
    l = 10**-3
    jN = 10**-6
    MIC_R = MIC_R * MIC_S

    # Subpopulations
    dSdt = S*(1-((S+RA)/10**Bmax))*KG*(1- ((Gmax - Gmin_A/KG)*(A/MIC_S)**HILL_A/((A/MIC_S)**HILL_A - (Gmin_A/KG/Gmax)))) - jN*S*E
    dRAdt = RA*(1-((S+RA)/10**Bmax))*KG*FIT*(1- ((Gmax - Gmin_A/(KG*FIT))*(A/MIC_R)**HILL_A/((A/MIC_R)**HILL_A - (Gmin_A/(KG*FIT)/Gmax)))) - jN*RA*E
    dAdt = f_A
    dEdt = q*(e_max - E) - l*E

    return [dSdt, dRAdt, dAdt, dEdt]

which_clade = 3

clades = {
    1: {"KG": 0.219692291518742, "MIC_S": 1},
    3: {"KG": 0.120284472246658, "MIC_S": 1},
    4: {"KG": 0.192553737889521, "MIC_S": 2},
    5: {"KG": 0.176254143936869, "MIC_S": 2}
}

clade = clades[which_clade]

Bmax = 9
KG = clade["KG"]
ke_bac = 0
V = 100
Gmax = 1
Gmin_A = -3
Gmin_B = -1
HILL_A = 4
MIC_S = clade["MIC_S"] 
S0 = 0
RA0 = 1000 
A0 = 1.5

length_experiment = 24 * 8
t = np.linspace(0, length_experiment, length_experiment)

# Create 3 subplots for low, mid, high immune response
fig, axs = plt.subplots(1, 1, figsize=(16, 14))

e_max_values = [10**4]
titles = ['Extinction boundary NP-2', 'Extinction boundary NP-1', 'Extinction boundary NP-3']

for idx, e_max in enumerate(e_max_values):
    num_resistant_cells = np.zeros((len(fit_range), len(MIC_R_range)))
    for i in range(len(fit_range)):
        for j in range(len(MIC_R_range)):
            y0 = [S0, RA0, A0, 10**2]
            solution = [y0]
            A_values = [A0]
            dt = t[1] - t[0]

            params = [Bmax, KG, fit_range[i],  V, Gmax, Gmin_A, HILL_A,  MIC_S, MIC_R_range[j]]

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
                y = odeint(model, y0, ts, args=(params, A, e_max))
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

    if idx == 0:
        cmap = plt.cm.get_cmap('viridis')
        cmap.set_under(color=(232/255, 236/255, 241/255, 0.3))
        ticks = [3, 4.5, 6, 7.5, 9]
        cax = axs.contourf(MIC_R_range, fit_range, num_resistant_cells, cmap=cmap, vmin=ticks[0], vmax=9, alpha=0.8, levels=[0, 3, 4.5, 6, 7.5, 9])
        cbar = fig.colorbar(cax, ax=axs, ticks=ticks)
        # fix the countor levels
        cbar.set_label('Log10 (Number of cells)', fontsize=26)
        cbar.ax.tick_params(labelsize=28)
        axs.tick_params(axis='both', which='major', labelsize=32)
        axs.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        
    # Plot the extinction boundary for all e_max values
    extinction_boundary = axs.contour(MIC_R_range, fit_range, num_resistant_cells, levels=[3], colors='black')


    # Increase font size for axis labels
    axs.set_xlabel('RR (MIC)', fontsize=35)
    axs.set_ylabel('GR(µ)', fontsize=35)

    # Increase font size for title
    axs.set_title(f' Clade III', fontsize=38, fontweight='bold')

    #cbar = fig.colorbar(cax, ax=axs, ticks=[0, 3, 4.5, 6, 7.5, 9])
    # length of the colorbar should indepedent of the plot


    #cbar.set_label('log10(Resistant cells)', fontsize=16)
    cbar.ax.tick_params(labelsize=35)

    axs.tick_params(axis='both', which='major', labelsize=35)
    axs.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    texts = []
    offset = 0.015


    
    
plt.tight_layout()
# export the image as svg so I can move the labels in inkscape
plt.savefig(f'new_plot_supplementary/clade_{which_clade}_immune_response_f{e_max_values[0]}.svg', dpi=700)
plt.savefig(f'new_plot_supplementary/clade_{which_clade}_immune_response_f{e_max_values[0]}.png', dpi=700)

