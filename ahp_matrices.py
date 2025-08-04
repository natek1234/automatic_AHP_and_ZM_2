import numpy as np
import ahpy


def calculate_Q(m, m_min, m_max):
    if m_min == m_max:
        return 1

    return 1 + ((m - m_min) / (m_max-m_min))*8

def calculate_r_kl(Q_k, Q_l):
    if Q_k >= Q_l:
        return np.round(Q_k / Q_l)
    else:
        return 1 / np.round(Q_l / Q_k)

# METRIC DATA #

# Order: Auger Hopper, Vib Intake Chute, Vib. Bowl Feeder, Front Loader, Vib. Funnel
INTAKE_MODULE_METRICS = [
    [0,0,0,1,0],
    [1,2,2,3,2],
    [3.75,5.25,5.25,5.5,5.25],
    [2,1,1,2,1],
    [0.046,0.068,0.87,0.76,0.19],
    [43.09,39.08,72.36,17.07,23.04],
    [4,2,2,4,2],
    [6.32,6.99,1.68,60,160],
    [3,1,1,4,2]
]

INTAKE_MODULE_OBJ = ['Min', 'Min', 'Min', 'Min', 'Min', 'Min', 'Min', 'Max', 'Min']

# PR-CL, PR-WP, PR-CB, SR-CL, SR-WP, SR-CB, R-R
DEPLOY_MODULE_METRICS = [
    [3,3,3,3,3,3,3],
    [25.43,140.93,140.93,116,231.5,231.5,9],
    [4,4,4,3,3,3,3],
    [3,3,3,2,2,2,2],
    [39,39,45,24,24,30,44],
    [3,3,3,3,3,3,2],
    [3,3,2,4,4,3,5],
    [0.0788,0.0788,0.0788,0.0338,0.0338,0.0338,0.126]
]

DEPLOY_MODULE_OBJ = ['Min', 'Min', 'Min', 'Min', 'Min', 'Min', 'Max', 'Max']

INTERFACE_MODULE_METRICS = [
    [1,1,0,2,0,3,0],
    [2,1,3,3,1,2,5],
    [3.5,3.5,3.5,3.5,0,3.5,0],
    [2,2,2,2,1,1,1],
    [1,1,1,1,1,1,0],
    [0.045,0.05,5,8,1.9,1,3],
    [2,2,2,3,0,4,1],
    [216,6.5,100,1000,300,1000,216],
    [0,0,0,0,1,1,0],
    [2,10,10,10,2,2,4],
    [1,1,4,4,5,2,2]
]

INTERFACE_MODULE_OBJ = ['Min', 'Min', 'Min', 'Min', 'Min', 'Min', 'Min', 'Max', 'Max', 'Max', 'Max']

TRANSPORT_MODULE_METRICS = [
    [4, 5, 3, 2, 4],
    [2, 2, 2, 4, 3],
    [14, 17.5, 14, 7, 14],
    [5, 4, 3, 2, 1], # [4, 5, 4, 2, 4],
    [3, 3, 1, 1, 2],
    [40, 76, 96, 20, 11],
    [4, 5, 4, 3, 4],
    [900, 294, 41, 5, 900],
    [2, 2, 1, 5, 3],
    [1, 1, 1, 1, 1],
    [4, 5, 3, 1, 1],
    [3, 3, 2, 2, 5]
]

TRANSPORT_MODULE_OBJ = ['Min', 'Min', 'Min', 'Max', 'Min', 'Min', 'Min', 'Max', 'Min', 'Min', 'Max', 'Min']

# AHP Process #

# Make RI Matrices #

# Intake Module
intake_criteria_ri_values = {
    "C_I_1": 0.121,
    "C_I_2": 0.104,
    "C_I_3": 0.133,
    "C_I_4": 0.148,
    "C_I_5": 0.103,
    "C_I_6": 0.112,
    "C_I_7": 0.117,
    "C_I_8": 0.0805,
    "C_I_9": 0.0809,
}

# Deploy RCU Module
deploy_criteria_ri_values = {
    "C_I_1": 0.125,
    "C_I_2": 0.159,
    "C_I_3": 0.168,
    "C_I_4": 0.120,
    "C_I_5": 0.118,
    "C_I_6": 0.127,
    "C_I_7": 0.0816,
    "C_I_8": 0.101,
}

# Interface Module
interface_criteria_ri_values = {
    "C_I_1": 0.104,
    "C_I_2": 0.0894,
    "C_I_3": 0.117,
    "C_I_4": 0.127,
    "C_I_5": 0.0917,
    "C_I_6": 0.0873,
    "C_I_7": 0.100,
    "C_I_8": 0.0855,
    "C_I_9": 0.0850,
    "C_I_10": 0.0594,
    "C_I_11": 0.0533   
}

# Transport Module
transport_criteria_ri_values = {
    "C_I_1": 0.0982,
    "C_I_2": 0.0775,
    "C_I_3": 0.105,
    "C_I_4": 0.113,
    "C_I_5": 0.0815,
    "C_I_6": 0.0759,
    "C_I_7": 0.0859,
    "C_I_8": 0.0822,
    "C_I_9": 0.0786,
    "C_I_10": 0.0660,
    "C_I_11": 0.0680,
    "C_I_12": 0.0690    
}

# Configure which module we are computing for
MODULE_METRICS = TRANSPORT_MODULE_METRICS
MODULE_OBJ = TRANSPORT_MODULE_OBJ
criteria_ri_values = transport_criteria_ri_values

# Normalize the RI values to sum to 1
total_ri = sum(criteria_ri_values.values())
criteria_weights = {k: v / total_ri for k, v in criteria_ri_values.items()}

# Create a complete pairwise comparison dictionary for criteria
criteria_comparisons = {}
criteria_list = list(criteria_weights.keys())
for i, crit_i in enumerate(criteria_list):
    for j, crit_j in enumerate(criteria_list):
        if i < j:
            # Relative importance ratio based on RI values
            criteria_comparisons[(crit_i, crit_j)] = criteria_weights[crit_i] / criteria_weights[crit_j]

criteria = ahpy.Compare(name='Criteria', comparisons=criteria_comparisons, precision=5, random_index='saaty')

# Make RP Matrices #

# Automatically loop through all the criteria
all_criteria = {}
for c in range(0,len(MODULE_METRICS)):

    all_criteria['C_I_' + str(c+1)] = {}

    # NOTE: Invert max and min for minimizing objective

    if MODULE_OBJ[c] == 'Min':
        m_min = np.max(MODULE_METRICS[c])
        m_max = np.min(MODULE_METRICS[c])
    else:
        m_min = np.min(MODULE_METRICS[c])
        m_max = np.max(MODULE_METRICS[c])

    # Now create the RP Matrix for each criterion
    for i, crit_i in enumerate(MODULE_METRICS[c]):
        for j, crit_j in enumerate(MODULE_METRICS[c]):
            if i < j:
                all_criteria['C_I_' + str(c+1)][('Cand_' + str(i), 'Cand_' + str(j))] = calculate_r_kl(calculate_Q(crit_i, m_min, m_max), calculate_Q(crit_j, m_min, m_max))

# print(all_criteria['C_I_4'])
# exit()

# Create the AHP comparison objects
comparison_objects = []
for crit in all_criteria.keys():
    comparison_objects.append(ahpy.Compare(crit, all_criteria[crit], precision=5, random_index='saaty'))

compose = ahpy.Compose()

compose.add_comparisons(comparison_objects)

compose.add_comparisons(criteria)


compose.add_hierarchy({'Criteria': list(all_criteria.keys())})

#criteria.add_children(comparison_objects)

#print(comparison_objects[0].report)

#report = comparison_objects[0].report(show=True)

#report = compose.C_I_6.report(show=True, verbose=True)
criteria_report = compose.report(show=True)

# EXTRACT DECISION VALUES #

# Extract global weights (criteria priorities)
global_weights = np.array([criteria_ri_values[f"C_I_{i+1}"] for i in range(len(criteria_ri_values))])

# Normalize the global weights to ensure they sum to 1
global_weights /= global_weights.sum()

# Extract RP values for all criteria and candidates
# Create a matrix where rows correspond to criteria and columns to candidates
rp_matrix = np.zeros((len(criteria_ri_values), len(MODULE_METRICS[0])))

for i, crit in enumerate(all_criteria.keys()):
    # Extract the local weights for each candidate for the current criterion
    rp_values = criteria_report[crit]['elements']['local_weights']
    for j in range(len(MODULE_METRICS[0])):
        rp_matrix[i, j] = rp_values[f"Cand_{j}"]

# Multiply global weights with RP values
# This computes the weighted sum for each candidate across all criteria
weighted_rp_values = global_weights @ rp_matrix

# Display the results
for i, value in enumerate(weighted_rp_values):
    print(f"Candidate {i + 1} Weighted RP Value: {value:.4f}")