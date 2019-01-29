# Experiment Details

The `run_search.py` script allows you to choose any combination of eleven search algorithms
(three uninformed and eight with heuristics) on four air cargo problems. The cargo problem
instances have different numbers of airplanes, cargo items, and airports that increase the
complexity of the domains.

You should run all of the search algorithms on the first two problems and record the following
information for each combination:

- number of actions in the domain
- number of new node expansions
- time to complete the plan search

Use the results from the first two problems to determine whether any of the uninformed search
algorithms should be excluded for problems 3 and 4. You must run at least one uninformed search,
two heuristics with greedy best first search, and two heuristics with A* on problems 3 and 4.

# Report Requirements

Your submission for review must include a report named "report.pdf" that includes all of the
figures (charts or tables) and written responses to the questions below. You may plot multiple
results for the same topic on the same chart or use multiple charts. (Hint: you may see more
details by using log space for one or more dimensions of these charts.)

- Use a table or chart to analyze the number of nodes expanded against number of actions in the domain
- Use a table or chart to analyze the search time against the number of actions in the domain
- Use a table or chart to analyze the length of the plans returned by each algorithm on all search problems

Use your results to answer the following questions:

Which algorithm or algorithms would be most appropriate for planning in a very restricted domain (i.e., one that has only a few actions) and needs to operate in real time?

Which algorithm or algorithms would be most appropriate for planning in very large domains (e.g., planning delivery routes for all UPS drivers in the U.S. on a given day)

Which algorithm or algorithms would be most appropriate for planning problems where it is important to find only optimal plans?
