"""## Test SR on synthetic LJ potential dataset"""

using Plots


# Define parameters for the Lennard-Jones potential
sigma = 0.2
epsilon = 10.2

# Define the LJ potential function
function lj_potential(r, sigma, epsilon)
    return 4 * epsilon * ((sigma ./ r).^12 .- (sigma ./ r).^6)
end

# Create a uniformly spaced vector of radial distances
r = range(0.1, stop=3.0, length=1000)  # Avoid zero to prevent division by zero

# Calculate the potential for each distance
V = lj_potential(r, sigma, epsilon)


# Create a scatter plot of the Lennard-Jones potential
scatter(r, V, label="Lennard-Jones Potential", xlabel="Distance (r)", ylabel="Energy (E)", title="Lennard-Jones Potential vs Distance", legend=:topright)




"""## Test SR on GNN edge message dataset"""

using PyCall

# Import the pickle module from Python
@pyimport pickle

# Function to load a pickle file
function load_pickle(filename)
    try
        # Read the entire content of the file as bytes
        bytes = read(filename)  # This reads the file content into a Vector{UInt8}

        # Use pickle.loads() to load the data from bytes
        return pickle.loads(pybytes(bytes))  # Convert bytes to a Python bytes object
    catch e
        println("Error loading pickle file: ", e)
        return nothing  # Return nothing if there was an error
    end
end

# Example usage
#msg_force_dict_pkl_filename = "/content/msg_force_dict_epoch=29-step=270000_edge_msg_constrained_std_trained_over_9k_samples_our_run_5.pkl"
#msg_force_dict_pkl_filename = "/content/msg_force_dict_epoch=29-step=270000_edge_msg_constrained_std_trained_over_9k_samples_our_run_5_on_gamd_dataset.pkl"
#msg_force_dict_pkl_filename = "/content/msg_force_dict_epoch=29-step=135000_edge_msg_constrained_std_trained_over_4.5k_samples_custom_potential.pkl"
msg_force_dict_pkl_filename = "msg_force_dict_epoch=39-step=360000_edge_msg_constrained_std.pkl"
msg_force_dict = load_pickle(msg_force_dict_pkl_filename)

# Displaying the loaded data
if msg_force_dict !== nothing
    println("Loaded data: ", msg_force_dict.keys)
else
    println("Failed to load data.")
end

using LinearAlgebra
using Plots

# Set parameters for Lennard-Jones potential
epsilon = 0.0238  # Depth of the potential well
sigma = 0.98#3.4    # Finite distance at which the potential is zero

@pyimport torch
@pyimport numpy as np

using DataFrames

X = msg_force_dict["radial_distance"].cpu().numpy()


# Calculate Lennard-Jones potential for each radial distance
Y = [
     4.0 * epsilon * ((sigma / norm(X[i, :]))^12 - (sigma / norm(X[i, :]))^6)
     #(-(sigma / norm(X[i, :]))^6)
     #((sigma / norm(X[i, :]))^12)
     #((sigma / norm(X[i, :]))^12 - (sigma / norm(X[i, :]))^6)
    for i in eachindex(axes(X, 1))
]

# Calculate the standard deviation of Y values
std_dev_Y = np.std(Y)
# Print or return the standard deviation
println("Standard Deviation of Y values: ", std_dev_Y)



# Y now contains the Lennard-Jones potentials corresponding to each radial distance


# Plotting Y as a function of X
plot(X, Y, seriestype = :scatter, label = "Lennard-Jones Potential", xlabel = "Radial Distance (X)", ylabel = "Potential (Y)", title = "Lennard-Jones Potential vs Radial Distance", legend = true)


using DataFrames

X = msg_force_dict["radial_distance"].cpu().numpy()


edge_messages_julia = msg_force_dict["edge_messages"].cpu().numpy()
Y = edge_messages_julia[:, 3] # Get all rows and the first column (Julia indexing starts at 1)

# Create a mask
mask = (0 .<= Y) #.& (10.0 .>= X)

# Filter X and Y using the mask
X = X[mask]
Y = Y[mask]

# Calculate the standard deviation of Y values
std_dev_Y = np.std(Y)

# Print or return the standard deviation
println("Standard Deviation of Y values: ", std_dev_Y)

# Plotting Y as a function of X
plot(X, Y, seriestype = :scatter, label = "Edge messages", xlabel = "Radial Distance (X)", ylabel = "Edge messages (Y)", title = "Edge messages vs Radial Distance", legend = true)


"""# Downsample edge messages (to map it to a function)"""

using Statistics
# Define the window size for downsampling
window_size = 0.1  # Adjust this value as needed

# Create arrays to store downsampled results
downsampled_X = Float64[]
downsampled_Y = Float64[]

# Get unique X values for downsampling
unique_X_values = sort(unique(X))

# Iterate over unique X values and compute averages within the window
for x in unique_X_values
    # Find indices of Y values within the window around x
    indices_in_window = findall((X .>= (x - window_size / 2)) .& (X .<= (x + window_size / 2)))

    if !isempty(indices_in_window)
        # Calculate average of Y values in this window
        avg_Y = mean(Y[indices_in_window])

        # Check if x is already in downsampled_X before adding it
        if x in downsampled_X
            continue  # Skip if x is already present
        else
            # Append results to downsampled arrays
            push!(downsampled_X, x)
            push!(downsampled_Y, avg_Y)
        end
    end
end

# Convert results to arrays if needed
X = collect(downsampled_X)
Y = collect(downsampled_Y)

# Print or return the downsampled results
println("Downsampled X: ", X)
println("Downsampled Y: ", Y)

# Plotting downsampled Y as a function of downsampled X
plot(X, Y, seriestype = :scatter, label = "Downsampled Edge messages", xlabel = "Radial Distance (X)", ylabel = "Average Edge messages (Y)", title = "Downsampled Edge messages vs Radial Distance", legend = true)


# Remove rows where downsampled_X is less than 2
mask_final = (X .>= 3.5)

# Filter downsampled arrays using the final mask
X = X[mask_final]
Y = Y[mask_final]

# Print or return the final downsampled results
println("Final Downsampled X: ", X)
println("Final Downsampled Y: ", Y)

# Plotting final downsampled Y as a function of final downsampled X
plot(X, Y, seriestype = :scatter, label = "Final Downsampled Edge messages", xlabel = "Radial Distance (X)", ylabel = "Average Edge messages (Y)", title = "Final Downsampled Edge messages vs Radial Distance", legend = true)
savefig("edge_msg_downsampled_without_temp_exp.png")

X = vec(Float64.(X)) # Convert X to Float64
Y = vec(Float64.(Y))
# Convert X to a DataFrame with a single column
X = DataFrame(radial_distance = X)
X = Dict(:radial_distance => X.radial_distance)

using SymbolicRegression
function lj_potential_structure((; attr_func, rep_func), (rad, ))
  _attr_func = attr_func(rad)^-12
  _rep_func = rep_func(rad)^-6

  out = map((attr_func_i, rep_func_i) -> (attr_func_i - rep_func_i), _attr_func.x, _rep_func.x)
  return ValidVector(out, _attr_func.valid && _rep_func.valid)
end
lj_structure = TemplateStructure{(:attr_func, :rep_func)}(lj_potential_structure)

elementwise_loss = ((x1), (y1)) -> abs(y1 - x1)

using MLJBase

model = SRRegressor(;
    niterations=10000,
    selection_method=SymbolicRegression.MLJInterfaceModule.choose_best,
    binary_operators=(*, /),
    maxsize=25,
    elementwise_loss=elementwise_loss,
    #expression_type=TemplateExpression,
    # Note - this is where we pass custom options to the expression type:
    #expression_options=(; structure = lj_structure),
    batching=true,
)


mach = machine(model, X, Y)
fit!(mach)

report(mach)

r = report(mach)
idx = r.best_idx
best_expr = r.equations[idx]
print("Best equation: ", best_expr)
best_attr = get_contents(best_expr).attr_func
best_rep = get_contents(best_expr).rep_func

print("\nAttr term: ", best_attr)
print("\nRep term: ", best_rep)

y_pred = predict(mach, X)
# Plotting Y as a function of X
plot(y_pred, Y, seriestype = :scatter, label = "GT vs Pred (x-axis)", xlabel = "Pred", ylabel = "GT", title = "GT vs Pred", legend = true)
savefig("pred_vs_gt_without_temp_exp.png")
