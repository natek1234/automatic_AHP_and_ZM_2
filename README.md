# automatic_AHP_and_ZM_2

This tool is made to implement the Analytic Hierarchy Process (AHP) and the ZM_II ranking aggregation algorithm across multiple "modules". Each module has its own ZM_II ranking and AHP evaluation. The tool "best" combination of candidate solutions across the modules. 

This tool does the following:
1. Intakes expert criteria priorities for multiple modules in the form of simple rankings.
2. Aggregates expert priorities to produce a ratio scale output for each module.
3. Converts ratio scale outputs to a normalized AHP Relative Important (RI) scale.
4. Uses these RI values to evaluate candidates across all criteria for each modules.
5. Each candidate is evaluated according to set of metrics for a particular criterion within a given module.
6. These metrics are used to automatically produce AHP Relative Preference (RP) matrices with close to ideal consistency.
7. The RP and RI values for each module are used to produce the AHP Decision Values.
8. The Decision Values are normalized based on the total number of criteria per module.
9. Permutations of the Decision Values for candidates across all modules are summed to find the best combination of candidates.

## Dependencies

- AHPy >= 2.0
- Numpy >= 2.1.2
- Scipy >= 1.14.1

## Work Flow

- Step 1: `ranking_aggregation_original.py`
  - Step 1.1: Define the module names
  - Step 1.2: Use the `survey_map.json` to decode the expert rankings contained in `survey_output.json`
  - Step 1.3: Run the file to output the aggregated rankings to `survey_results_sorted.json`.

- Step 2: `ahp_matrices.py`
  - Step 1.1: Define module metric matrices
  - Step 2.2: Define criteria objectives (minimize or maximize)
  - Step 2.3: Define the RI values for each module
  - Step 2.4: Run the file to produce the the RP values for each candidate solution for a particular module.
  - Step 2.5: Repeat for all modules.
    
- Step 3: `cand_perm.py`
  - Step 3.1: Define the modules and the candidate solution RPs.
  - Step 3.2: Run the file to produce the top 10 combined solutions across the modules.
