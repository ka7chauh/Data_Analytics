Steps 1: Select All feature based on IV>0.1
Steps 2: With All Feature fit into any GB Model and get importance
Step 3: Get VIF and merge importance,iv,VIF together for features 
Step 4 : using from sklearn.feature_selection import RFE slect top 30/50 etc 
Step 5 : get correlation (>0.6) AND ivf >5 get unique features out
Step 6 : Check Corr again for remaing feature drop (>0.6) it 
Step 7: Repeat till no sets of columns remain where corr >0.6


