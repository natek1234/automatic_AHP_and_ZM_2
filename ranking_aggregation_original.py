import numpy as np
from scipy.stats import norm
import math
import json

np.set_printoptions(precision=5, suppress=True, linewidth=120)

MODULES = ["Intake Lunar Regolith Module", "Deploy Empty RCU Module", "Interface with and Secure RCU Module", "Transport RCU to Desired Pose Module"]

def compute_pij(k_ij, m_ij):
    """Calculate the estimated probability \hat{p}_{ij}."""
    # New order: last `move_count` indices followed by the rest
    return k_ij / m_ij


def compute_z_matrix(pij_matrix):
    """Construct the Z matrix based on the inverse CDF of the normal distribution."""
    n = pij_matrix.shape[0]
    z_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if pij_matrix[i, j] > 0:  # Only fill when pij exists
                if pij_matrix[i, j] >= 0.977:
                    z_matrix[i, j] = -1.995
                elif pij_matrix[i, j] <= 0.023:
                    z_matrix[i, j] = 1.995
                else:
                    z_matrix[i, j] = norm.ppf(1 - pij_matrix[i, j])
                z_matrix[j, i] = -z_matrix[i, j]  # Ensure skew-symmetry

    return z_matrix

def construct_A_B(z_matrix):
    """Construct A and B matrices."""
    n = z_matrix.shape[0]
    B = np.sum(z_matrix, axis=1).reshape(-1, 1)
    
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if z_matrix[i, j] != 0:
                A[i, j] = -1
        A[i, i] = np.sum(z_matrix[i, :] != 0)
    
    return A, B

def construct_W(pij_matrix, m_ij_matrix):
    """Construct the W matrix."""
    n = pij_matrix.shape[0]
    W = np.zeros((n, n))
    cov = np.zeros((n, n))
    J = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if pij_matrix[i, j] > 0:
                p_ij = pij_matrix[i, j]

                if p_ij >= 0.997:
                    p_ij = 0.997
                elif p_ij <= 0.003:
                    p_ij = 0.003

                m_ij = m_ij_matrix[i, j]
                
                # Compute Jacobian
                if p_ij == 0.5:
                    J[i,j] = 2.506628
                else:
                    try:

                        numerator = np.sqrt(2) * (2 * p_ij - 1)
                        log_term = np.log(4) + np.log(p_ij * (1 - p_ij))
                        denominator = p_ij * (1 - p_ij) * np.sqrt(-2 * np.sqrt(2 * np.pi) * log_term)
                        J[i,j] = np.abs(numerator / denominator)
                    except ValueError:
                        # Fall back to a capped approximation if log fails
                        J[i,j] = np.sqrt(2) * (2 * p_ij - 1) / 0.01

                
                if i == j:
                    # Add contribution to W
                    cov[i, i] = ((p_ij * (1-p_ij)) / m_ij)  

    W = np.linalg.inv(J @ cov @ J.T)

    return W

def solve_system(A, B, W):
    """Solve the linear system using generalized least squares."""
    X = -np.linalg.pinv(A.T @ W @ A) @ (A.T @ W @ B)
    return X.flatten()

def ratio_scale_transform(X, mu_t, mu_b):
    """Transform interval scale values to ratio scale."""
    return 100 * (X - mu_b) / (mu_t - mu_b)

def RI_scale_transform(Y, dummy=False):
    """Transform ratio scale values to RI scale."""
    if dummy:
        return Y[:-2] / np.sum(Y[:-2])
    else:
        return Y / np.sum(Y)

def calculate_uncertainty(A, W):
    """Compute uncertainty matrix."""
    cov_X = np.linalg.pinv(A.T @ W @ A)
    return cov_X

def convert_X_to_Y_uncertainty(X, X_uncertainty_matrix):

    J_y = np.zeros_like(X_uncertainty_matrix)

    for i in range(J_y.shape[0]-1):
        J_y[i,i] = 100 / (X[-1] - X[-2])

    J_y[-1,-1] = 0
    J_y[-2,-2] = 0

    for i in range(0, J_y.shape[0]-2):
        J_y[i,-2] = 100 * ((X[i] - X[-1]) / (X[-1] - X[-2])**2)
        J_y[i,-1] = -100 * ((X[i] - X[-2]) / (X[-1] - X[-2])**2)
            
    Y_cov = J_y @ X_uncertainty_matrix @ J_y.T

    return Y_cov

def confidence_intervals(Y, uncertainty_matrix):
    """Compute 95% confidence intervals."""
    std_devs = np.sqrt(np.diag(uncertainty_matrix))
    #return [(y - 2 * std, y + 2 * std) for y, std in zip(Y, std_devs)]
    return std_devs

def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # Only flatten if it's a list
            flat_list.extend(flatten(item))  # Recursively flatten
        else:
            flat_list.append(item)  # Add non-list items directly
    return flat_list

# Consistency with collective ranking
def calculate_p_consistency(expert_rankings, new_ranks, P, m, dummy=False):
    """Compute consistency indicators."""    
    x_j = []

    P_collective = brute_force_decode_rankings_to_P(new_ranks, P, dummy)
    for ranking in expert_rankings:
        if dummy:
            P_expert = brute_force_decode_rankings_to_P(ranking, P, dummy)
            x_j = x_j + [np.sum((P_expert[:-2,:-2] == 1) & (P_collective[:-2,:-2] == 1))]
        else:    
            P_expert = brute_force_decode_rankings_to_P(ranking, P, dummy)
            x_j = x_j + [np.sum((P_expert == 1) & (P_collective == 1))]

    if dummy:
        p = (1 / m) * (np.sum(np.array(x_j) / math.comb(P-2, 2)))
    else:
        p = (1 / m) * (np.sum(np.array(x_j) / math.comb(P, 2)))
    return p

# Inter expert consistency
def calculate_W_consistency(expert_rankings, P, m, dummy=False):

    if dummy:
        p = P - 2
        R = np.zeros((p, 1))
    
        for ranking in expert_rankings:
            for i in range(1, len(ranking)-1):
                R[int(ranking[i][1:])-1] = R[int(ranking[i][1:]) -1] + i # accumulate score
    else:
        p = P
        R = np.zeros((p, 1))
    
        for ranking in expert_rankings:
            for i in range(0, len(ranking)):
                R[int(ranking[i][1:])-1] = R[int(ranking[i][1:]) -1] + (i+1) # accumulate score
    
    W = (12 * np.sum(R**2) - 3 * m**2 * p * (p+1)**2) / (m**2 * p * (p**2 - 1))
    return W

def brute_force_decode_rankings_to_P(rankings, num_criteria, dummy=False):
    """
    Decodes a set of rankings into a comparison matrix P using a brute-force approach.
    Handles incomplete rankings by ensuring the output matrix size matches num_criteria.

    Args:
        rankings (list of lists): A list of rankings where:
                                  - Each element is either an object or a list of objects in the same rank (indifference).
        num_criteria (int): Total number of criteria (unique objects).

    Returns:
        numpy.ndarray: The comparison matrix P of size (num_criteria x num_criteria).
    """
    # Extract unique objects from rankings
    ranked_objects = set()
    for group in rankings:
        if isinstance(group, list):
            ranked_objects.update(group)
        else:
            ranked_objects.add(group)

    # Create a complete list of objects (fill in missing objects)
    if dummy:
        all_objects = [f"O{i+1}" for i in range(num_criteria-2)]
        all_objects = all_objects + ['OZ', 'OM']
    else:
        all_objects = [f"O{i+1}" for i in range(num_criteria)]

    # Assign indices to all objects
    object_indices = {obj: i for i, obj in enumerate(all_objects)}

    # Initialize P matrix
    P = np.zeros((num_criteria, num_criteria))

    # Iterate over all groups in the rankings
    for i, group_i in enumerate(rankings):
        # Ensure group_i is a list for consistency
        if not isinstance(group_i, list):
            group_i = [group_i]

        # Compare group_i with all lower-ranked groups
        for j in range(i + 1, len(rankings)):
            group_j = rankings[j]
            if not isinstance(group_j, list):
                group_j = [group_j]

            # Update P matrix for strict preference
            for obj_i in group_i:
                for obj_j in group_j:
                    if obj_i in object_indices and obj_j in object_indices:
                        P[object_indices[obj_i], object_indices[obj_j]] += 1

        # Handle indifferences within the group
        for obj_i in group_i:
            for obj_j in group_i:
                if obj_i in object_indices and obj_j in object_indices:
                    P[object_indices[obj_i], object_indices[obj_j]] += 0.5

    return P

def rank(scores, P, dummy=False):
    sorted_scores = sorted(scores)
    new_rank_indices = []
    for score in scores:
        new_rank_indices.append(-((np.where(sorted_scores == score)[0][0] + 1).item() - (P+1)))

    new_rank = [''] * P
    for i in range(len(new_rank_indices)):
        if dummy:
            if i < P-2:
                new_rank[new_rank_indices[i]-1] = 'O' + str(i+1)
            elif i == P-2:
                new_rank[new_rank_indices[i]-1] = 'OZ'
            else:
                new_rank[new_rank_indices[i]-1] = 'OM'
        else:
            new_rank[new_rank_indices[i]-1] = 'O' + str(i+1)
    return new_rank, new_rank_indices

# EXAMPLE USAGE #
"""
P = 6  # Number of criteria
m = 5  # Number of experts
expert_rankings = [np.array([0, 1, 2, 3, 4])] * m  # Example expert rankings

expert_ranks = [
    #['OM', ['O1', 'O2', 'O5', 'O6', 'O7'], 'O3', 'O4', ['O8', 'OZ']]
    ['OM', ['O3', 'O4'], 'O1', 'O2', 'OZ'],
    [['O3', 'O4', 'OM'], 'O1', 'O2', 'OZ'],
    [['O4', 'OM'], 'O3', ['O1', 'O2'], 'OZ'],
    ['OM', 'O3', ['O1', 'O4'], ['O2', 'OZ']],
    ['OM', 'O3', 'O4', 'O2', 'O1', 'OZ']
]

num_criteria = 6

k_ij = np.zeros((num_criteria, num_criteria))

for i in range(len(expert_ranks)):
    k_ij = k_ij + brute_force_decode_rankings_to_P(expert_ranks[i], num_criteria, dummy=True)
print(k_ij)

# Example pairwise comparison data
#k_ij = np.array([
#    [2.5, 3.5, 0.0, 0.0, 5.0, 0.0],  # Row O1
#    [1.5, 2.5, 0.0, 0.0, 4.5, 0.0],  # Row O2
#    [0.5, 0.0, 2.5, 3.0, 5.0, 1.0],  # Row O3
#    [4.5, 5.0, 2.0, 2.5, 5.0, 1.0],  # Row O4
#    [5.0, 5.0, 5.0, 5.0, 0.0, 5.0],  # Row Z
#    [5.0, 5.0, 4.5, 4.5, 5.0, 2.5],  # Row M
#])
m_ij = np.full((P, P), m)

# Step-by-step computations
pij_matrix = compute_pij(k_ij, m_ij)

# example
#pij_matrix = np.array([
#    [0.50, 0.70, 0.00, 0.50, 1.00, 0.00],  # Row O1
#    [0.30, 0.50, 0.00, 0.40, 0.90, 0.00],  # Row O2
#    [1.00, 1.00, 0.50, 0.60, 1.00, 0.10],  # Row O3
#    [0.90, 1.00, 0.40, 0.50, 1.00, 0.20],  # Row O4
#    [0.00, 0.00, 0.00, 0.00, 0.50, 0.00],  # Row Z
#    [1.00, 1.00, 0.90, 0.80, 1.00, 0.50],  # Row M
#])
"""

# MAIN FUNCTION #

# Process Rankings #

survey_data_path = ".\survey_output.json"
survey_map_path = ".\survey_map.json"
out_path = "./survey_results_sorted.json"

with open(survey_data_path, 'r') as file:
    survey_data = json.load(file)

with open(survey_map_path, 'r') as file:
    survey_map = json.load(file)

# expert_ranks = [[
#                 "Reliability",
#                 "Longevity",
#                 "Power",
#                 "Exposure to Dust",
#                 "Delivery flow rate",
#                 "Complexity",
#                 "Dust production",
#                 "Regolith compaction",
#                 "Mass"
#             ], 
#             [
#                 "Reliability",
#                 "Regolith compaction",
#                 "Longevity",
#                 "Power",
#                 "Mass",
#                 "Complexity",
#                 "Delivery flow rate",
#                 "Exposure to Dust",
#                 "Dust production"
#             ],
#             [
#                 "Reliability",
#                 "Longevity",
#                 "Complexity",
#                 "Delivery flow rate",
#                 "Regolith compaction",
#                 "Dust production",
#                 "Power",
#                 "Mass",
#                 "Exposure to Dust"
#             ]]

export = False # whether to create output json file

if export:
    export_contents = {}

for q in range(len(MODULES)):
    cur_module = MODULES[q]
    expert_ranks = []
    m = len(survey_data.keys())  # Number of experts
    dummy = True # Whether to include OZ and OM dummy objects

    # Extract one module of rankings from each expert
    for i in range(m):
        expert_id = f'id_'+str(i+1)
        expert_ranks = expert_ranks + [['OM'] + survey_data[expert_id][cur_module][0] + ['OZ']]


    # Encode the rankings in the expected ranking format
    for i in range(len(expert_ranks)):
        for n in range(len(expert_ranks[0])):
            if expert_ranks[i][n] != 'OM' and expert_ranks[i][n] != 'OZ':
                expert_ranks[i][n] = survey_map[cur_module][expert_ranks[i][n]]
        
    P = len(expert_ranks[0])  # Number of criteria
    print(P)

    # Aggregation Algorithm #

    # Form frequency matrix #

    k_ij = np.zeros((P, P))

    for i in range(len(expert_ranks)):
        k_ij = k_ij + brute_force_decode_rankings_to_P(expert_ranks[i], P, dummy=True)

    # Form probability matrix #

    m_ij = np.full((P, P), m)
    pij_matrix = compute_pij(k_ij, m_ij)

    #print(k_ij)
    #print(pij_matrix)

    # Form Z matrix #

    z_matrix = compute_z_matrix(pij_matrix)
    #print(z_matrix)

    # Solve System #

    A, B = construct_A_B(z_matrix)
    #print(A)
    #print(B)

    # ERRORS AFTER THIS LINE - GLS Solution #
    W = construct_W(pij_matrix, m_ij)
    X = solve_system(A, B, W)

    #print(W)
    # print(X)

    # Ratio scaling
    mu_t, mu_b = max(X), min(X)
    Y = ratio_scale_transform(X, mu_t, mu_b).flatten()

    # Convert to RI scale
    RI = RI_scale_transform(Y, dummy)

    # Create new rank
    new_rank, new_rank_indices = rank(Y, P, dummy)

    # Uncertainty
    X_uncertainty_matrix = calculate_uncertainty(A, W)
    # print(X)
    print(np.sqrt(np.diag(X_uncertainty_matrix)))
    # exit()
    Y_uncertainty_matrix = convert_X_to_Y_uncertainty(X, X_uncertainty_matrix)
    intervals = confidence_intervals(Y, Y_uncertainty_matrix)

    # Consistency
    consistency = calculate_p_consistency(expert_ranks, new_rank, P, m, dummy)
    W_consistency = calculate_W_consistency(expert_ranks, P, m, dummy)

    #for testing
    new_intervals = np.zeros((len(new_rank_indices), 1))
    for r in range(len(new_rank_indices)):
        new_intervals[new_rank_indices[r]-1] = intervals[r]

    # Output results
    print("--------------------------------------")
    print(MODULES[q])
    print("--------------------------------------")
    print("W Consistency:", W_consistency)
    print("Ratio Scale:", Y)
    print("RI Scale: ", RI)
    print('New Rank: ', new_rank)
    print(new_rank_indices)
    print("Confidence Intervals:", intervals)
    print("Confidence Intervals Sorted:", new_intervals)
    print("Consistency:", consistency)
    print("\n\n")

    # EXPORT RESULTS #
    if export:
        Y_sorted, RI_sorted, intervals_sorted = zip(*sorted(zip(Y.tolist(), RI.tolist(), intervals.tolist()), reverse=True))
        export_contents[MODULES[q]] = {
            "ratio_scale": Y_sorted,
            "ri_scale": RI_sorted,
            "new_rank": new_rank,
            "confidence_intervals": intervals_sorted,
            "p_consistency": consistency,
            "w_consistency": W_consistency
        }

if export:
    with open(out_path, "w") as file:
        json.dump(export_contents, file, indent=4) 

