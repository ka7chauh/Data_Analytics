
Big Picture: What does this code do?
This function tries to automatically find customer segments (rules) like:

“Customers with
income > 8L AND utilization <= 30%
have much higher default rate than average”

It is RuleFit‑like:

Uses trees to discover candidate rules
Uses Lasso (L1) to select only important rules
Filters rules based on size, events, and lift

✅ Final output = interpretable risk segments

Step 1: Select Features and Target
Step 2: Train Gradient Boosting Trees
Handling Imbalance (Very Important)
Step 3: Extract Rules from Trees
Step 4: Remove Duplicate Rules
Step 5: Build Rule Matrix (Very Important Concept)
This creates a matrix like:





























CustomerRule 1Rule 2Rule 3C1100C2011C3110

1 → customer satisfies the rule
0 → customer does not
Step 6: Lasso to Select Important Rules
Step 7: Evaluate Each Selected Rule
