from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np

# Define system parameters for argon
num_atoms = 258
sigma = 0.34 * nanometer  # Lennard-Jones sigma for argon (in nanometers)
epsilon = 0.238 * kilocalories_per_mole  # Lennard-Jones epsilon for argon (in kcal/mol)
box_size = 2.0 * nanometer  # Set box size for periodic boundaries

# Initialize the system with periodic boundary conditions
system = System()
argon_mass = 39.948 * amu
for _ in range(num_atoms):
    system.addParticle(argon_mass)

# Set up periodic box
system.setDefaultPeriodicBoxVectors(Vec3(box_size, 0, 0), Vec3(0, box_size, 0), Vec3(0, 0, box_size))

# Define Lennard-Jones potential with periodic boundary conditions (global constants)
lj_potential = '4*epsilon*((sigma/r)^12 - (sigma/r)^6)'  # Using epsilon and sigma as constants in the expression
custom_force = CustomNonbondedForce(lj_potential)

# Add the constants for epsilon and sigma as global parameters
custom_force.addGlobalParameter("epsilon", epsilon)
custom_force.addGlobalParameter("sigma", sigma)

# Add particles to the custom force (matching the system size)
# Since all particles have the same epsilon and sigma, we add them as identical particles
for _ in range(num_atoms):
    custom_force.addParticle([])  # No per-particle parameters since epsilon and sigma are global constants

# Set periodic cutoff for nonbonded interactions
custom_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
custom_force.setCutoffDistance(1.0 * nanometer)
system.addForce(custom_force)

# Generate FCC lattice positions within the box
def generate_fcc_lattice(num_cells, cell_length):
    positions = []
    for i in range(num_cells):
        for j in range(num_cells):
            for k in range(num_cells):
                # FCC lattice positions within each cell
                positions.append((i, j, k))
                positions.append((i + 0.5, j + 0.5, k))
                positions.append((i + 0.5, j, k + 0.5))
                positions.append((i, j + 0.5, k + 0.5))
    positions = np.array(positions) * cell_length
    return positions[:num_atoms]

# Set up FCC lattice
num_cells = 6
cell_length = 0.34 * 2**(1/6)
positions = generate_fcc_lattice(num_cells, cell_length)

# Set positions in the simulation
simulation = Simulation(Modeller(Topology(), system), system, VerletIntegrator(0.001 * picoseconds))
simulation.context.setPositions(positions * nanometer)

# Calculate total potential energy
state = simulation.context.getState(getEnergy=True)
total_energy = state.getPotentialEnergy()
print("Total Potential Energy:", total_energy)

# Calculate pairwise potential energies with minimum image convention
def lennard_jones_potential(r, epsilon, sigma):
    r6 = (sigma / r)**6
    return 4 * epsilon * (r6**2 - r6)

pairwise_energies = []
box_vectors = simulation.context.getState().getPeriodicBoxVectors()

def minimum_image_distance(pos_i, pos_j, box_vectors):
    """Compute minimum image distance between two positions in a periodic box."""
    delta = pos_j - pos_i
    for dim in range(3):
        delta[dim] -= box_vectors[dim][dim] * round(delta[dim] / box_vectors[dim][dim])
    return np.linalg.norm(delta) * nanometer

for i in range(num_atoms):
    for j in range(i + 1, num_atoms):
        pos_i = positions[i] * nanometer
        pos_j = positions[j] * nanometer
        distance = minimum_image_distance(pos_i, pos_j, box_vectors)
        
        if distance < 1.0 * nanometer:
            energy = lennard_jones_potential(distance, epsilon, sigma)
            pairwise_energies.append((i, j, energy))

# Output pairwise potential energies
print("Pairwise Potential Energies (Atom i, Atom j, Energy):")
for (i, j, energy) in pairwise_energies:
    print(f"Atom {i} - Atom {j}: {energy}")
