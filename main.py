import numpy as np
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(
    [[300.0], [400.0], [400.0], [550.0], [720.0], [850.0], [900.0],[950.0]],
    [300.0, 350.0, 490.0, 500.0, 600.0, 610.0, 700.0, 660.0]
)

# y=0.53096099x + 189.75347155122438
print(reg.coef_) # [0.53096099]
print(reg.intercept_) # 189.75347155122438

# [667.61836402]
print(reg.predict([[900.0]]))
