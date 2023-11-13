# Importing necessary libraries and setting up the initial data and functions
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from adjustText import adjust_text

data = {
    "1": [
        {"strain": "5", "RR": 40.4649161, "GR": 0.3387018},
        {"strain": "8", "RR": 3.80829387, "GR": 0.71951784},
        {"strain": "9", "RR": 2.04763955, "GR": 0.51078035},
        {"strain": "NCP1Δ", "RR": 2.29636007, "GR": 0.41268195},
        {"strain": "2", "RR": 2.42215526, "GR": 0.84020572},
        {"strain": "3", "RR": 2.57425743, "GR": 0.77701439},
        {"strain": "4", "RR": 2.7981059, "GR": 0.83907606},
        {"strain": "6", "RR": 3.45099727, "GR": 0.86052151},
        {"strain": "7", "RR": 3.46821639, "GR": 0.70469188},
        {"strain": "ERG6Δ", "RR": 3.92213134, "GR": 0.66675044},
    ],
    "3": [
        {"strain": "11", "RR": 9.55321765, "GR": 0.44226174},
        {"strain": "12", "RR": 4.91597213, "GR": 0.61750143},
        {"strain": "ERG3Δ", "RR": 2.18335838, "GR": 0.84948045},
        {"strain": "14", "RR": 2.47574805, "GR": 0.52749359},
        {"strain": "16", "RR": 3.91856811, "GR": 0.69496974},
        {"strain": "NCP1Δ", "RR": 3.31648222, "GR": 0.62509194},
        {"strain": "13", "RR": 8.41508403, "GR": 0.57514907},
        {"strain": "ERG11M306I", "RR": 25.5271667, "GR": 0.43294126},
        {"strain": "15", "RR": 2.93619347, "GR": 0.88985395},
        {"strain": "17", "RR": 4.62631507, "GR": 0.84383552},
        {"strain": "ERG6Δ", "RR": 5.73848886, "GR": 0.81320832},
    ],
    "4": [
        {"strain": "25", "RR": 1.9068805, "GR": 0.70929609},
        {"strain": "26", "RR": 1.47077082, "GR": 0.7389433},
        {"strain": "27", "RR": 1.79513709, "GR": 0.75742566},
        {"strain": "28", "RR": 1.70098293, "GR": 0.72595803},
        {"strain": "HMG1R983S", "RR": 1.95861355, "GR": 0.65521852},
        {"strain": "29", "RR": 1.4009312, "GR": 0.80935938},
        {"strain": "ERG10T244N", "RR": 1.32453871, "GR": 0.55351015},
        {"strain": "19", "RR": 2.6057941, "GR": 0.57222081},
        {"strain": "20", "RR": 2.72788412, "GR": 0.800046},
        {"strain": "21", "RR": 2.65080186, "GR": 0.94950802},
        {"strain": "22", "RR": 1.83911019, "GR": 0.67436542},
        {"strain": "23", "RR": 2.42317641, "GR": 0.6389837},
        {"strain": "24", "RR": 2.07966891, "GR": 0.81952012},
        {"strain": "ERG6Δ", "RR": 2.04604242, "GR": 0.70493325},
    ],
    "5": [
        {"strain": "31", "RR": 3.53393295, "GR": 0.77063877},
        {"strain": "32", "RR": 3.06050695, "GR": 0.94851867},
        {"strain": "ERG11D225N", "RR": 2.95693649, "GR": 0.68025386},
    ],
}


fit_range = np.linspace(0.1, 1.1, 42)
MIC_R_range1 = np.linspace(1, 10, 32)
MIC_R_range2 = np.linspace(11, 64, 10)
MIC_R_range = np.concatenate((MIC_R_range1, MIC_R_range2))

def antibiotic_decay(A, t_half):
    ke = np.log(2) / t_half
    return -ke * A

def model(y, t, params, f_A, e_max):
    S, RA, A, E = y
    A = f_A
    Bmax, KG, FIT, V, Gmax, Gmin_A, HILL_A,  MIC_S, MIC_R = params

    q = 1
    l = 10**-3
    jN = 10**-6

    dSdt = S*(1-((S+RA)/10**Bmax))*KG*(1- ((Gmax - Gmin_A/KG)*(A/MIC_S)**HILL_A/((A/MIC_S)**HILL_A - (Gmin_A/KG/Gmax)))) - jN*S*E
    dRAdt = RA*(1-((S+RA)/10**Bmax))*KG*FIT*(1- ((Gmax - Gmin_A/(KG*FIT))*(A/MIC_R)**HILL_A/((A/MIC_R)**HILL_A - (Gmin_A/(KG*FIT)/Gmax)))) - jN*RA*E
    dAdt = f_A
    dEdt = q*(e_max - E) - l*E

    return [dSdt, dRAdt, dAdt, dEdt]

which_clade = 1

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
RA0 = 10**6 
A0 = 1.5

length_experiment = 24 * 8
t = np.linspace(0, length_experiment, length_experiment)

e_max_values = [4*10**4, 10**5, 4*10**5]
titles = ['Neutropenia state 1', 'Neutropenia state 2', 'Neutropenia state 3']

# Proceed to creating the combined plot

fig, ax = plt.subplots(figsize=(16, 12))

# Loop over the different immune responses and plot only the extinction boundary
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

    # Plot only the extinction boundary
    extinction_boundary = ax.contour(MIC_R_range, fit_range, num_resistant_cells, levels=[6])
    
    # For shifting the label towards the end of the line, extract path and use the last point of the path
    for collection in extinction_boundary.collections:
        for path in collection.get_paths():
            vert = path.vertices
            # Using the last point of the path to set the position of the label
            x_label, y_label = vert[-10]
            ax.text(x_label, y_label, titles[idx], color=collection.get_edgecolor()[0], backgroundcolor='white')

# Adjust x-axis to reflect the new range and density
ax.set_xscale('log')

# Set x-ticks manually to reflect actual values
xticks_vals = [1, 2, 4, 8, 16, 32, 64]
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

ax.set_xlabel('RR')
ax.set_ylabel('GR')

# Title with A0 and t_half
ax.set_title(f'A0: {A0}, t_half: {t_half}, clade: {which_clade}')

# Add data points for strains in the chosen clade
texts = []  # create an empty list to store your text objects
offset = 0.01
for i, strain_data in enumerate(data[str(which_clade)]):
    x = strain_data['RR']
    y = strain_data['GR']
    label = strain_data['strain']
    ax.scatter(x, y, color='black', zorder=10, s=70)  # plot the data point
    texts.append(ax.text(x, y+offset, label)) 

# Call adjust_text
adjust_text(texts, 
            expand_points=(1, 1), 
            expand_text=(1, 1), 
            force_points=0.1, 
            ax=ax)

# Adding legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, titles, loc='upper right')

# Save the figure with A0, t_half, clade number, and immune response
plt.tight_layout()
plt_filename = f'final_plot_new/RR_GR_dose_{A0}_t_half_{t_half}_clade_{which_clade}_immune_response_combined.png'
plt.savefig(plt_filename, dpi=300)