'''
Purpose is to check if pySR can regress to LJ force equation if the data is clean.
'''
import numpy as np
import matplotlib.pyplot as plt

# Parameters
epsilon = 0.238  # Depth of potential well
sigma = 3.4    # Distance at which potential is zero

# Radial distance range
r = np.linspace(0.5, 3, 100)

# Lennard-Jones Force calculation
F = 48 * epsilon * (sigma**12 / r**13 - sigma**6 / r**7)


# Plotting
plt.scatter(r, F)
plt.show()


from pysr import PySRRegressor

model = PySRRegressor(
    niterations=1000,  # < Increase me for better results
    binary_operators=["+", "*", "-", "/", "pow"],
    unary_operators=[
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(r.reshape(-1, 1), F)
print(model)
