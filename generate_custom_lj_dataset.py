import logging

import numpy as np


logger = logging.getLogger(__name__)

import time
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

import openmm as mm
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm import unit
from openmm import app

from openmmtools import integrators
from openmmtools import testsystems


def subrandom_particle_positions(nparticles, box_vectors, method='sobol'):
    """Generate a deterministic list of subrandom particle positions.

    Parameters
    ----------
    nparticles : int
        The number of particles.
    box_vectors : openmm.unit.Quantity of (3,3) with units compatible with nanometer
        Periodic box vectors in which particles should lie.
    method : str, optional, default='sobol'
        Method for creating subrandom sequence (one of 'halton' or 'sobol')

    Returns
    -------
    positions : openmm.unit.Quantity of (natoms,3) with units compatible with nanometer
        The particle positions.

    Examples
    --------
    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors)

    Use halton sequence:

    >>> nparticles = 216
    >>> box_vectors = openmm.System().getDefaultPeriodicBoxVectors()
    >>> positions = subrandom_particle_positions(nparticles, box_vectors, method='halton')

    """
    # Create positions array.
    positions = unit.Quantity(np.zeros([nparticles, 3], np.float32), unit.nanometers)

    if method == 'halton':
        # Fill in each dimension.
        primes = [2, 3, 5]  # prime bases for Halton sequence
        for dim in range(3):
            x = halton_sequence(primes[dim], nparticles)
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x * l / l.unit, l.unit)

    elif method == 'sobol':
        # Generate Sobol' sequence.
        from openmmtools import sobol
        ivec = sobol.i4_sobol_generate(3, nparticles, 1)
        x = np.array(ivec, np.float32)
        for dim in range(3):
            l = box_vectors[dim][dim]
            positions[:, dim] = unit.Quantity(x[dim, :] * l / l.unit, l.unit)

    else:
        raise Exception("method '%s' must be 'halton' or 'sobol'" % method)

    return positions



def get_rotation_matrix():
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.random.uniform(-1.0, 1.0, size=(3,)) * np.pi
    print(f'Using angle: {angles}')
    Rx = np.array([[1., 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=np.float32)
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=np.float32)
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]], dtype=np.float32)
    rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))

    return rotation_matrix

def center_positions(pos):
    offset = np.mean(pos, axis=0)
    return pos - offset, offset


BOX_SCALE = 2
DT = 2
for seed in range(10):
    print(f'Running seed: {seed}')
    nparticles = 258

    reduced_density=0.05
    mass=39.9 * unit.amu  # argon
    sigma=3.4 * unit.angstrom  # argon,
    epsilon=0.238 * unit.kilocalories_per_mole  # argon,
    cutoff = 3.0 * sigma
    mass=39.9 * unit.amu
    charge = 0.0 * unit.elementary_charge       
    # Create an empty system object.
    system = openmm.System()

    # Determine volume and periodic box vectors.
    number_density = reduced_density / sigma**3
    volume = nparticles * (number_density ** -1)
    box_edge = volume ** (1. / 3.)
    a = unit.Quantity((box_edge,        0 * unit.angstrom, 0 * unit.angstrom))
    b = unit.Quantity((0 * unit.angstrom, box_edge,        0 * unit.angstrom))
    c = unit.Quantity((0 * unit.angstrom, 0 * unit.angstrom, box_edge))
    system.setDefaultPeriodicBoxVectors(a, b, c) 
    
    # Define Lennard-Jones potential with periodic boundary conditions (global constants)
    lj_potential = '4*epsilon*((sigma/r)^12 - (sigma/r)^6)'  # Using epsilon and sigma as constants in the expression
    custom_force = CustomNonbondedForce(lj_potential)
    # Add the constants for epsilon and sigma as global parameters
    custom_force.addGlobalParameter("epsilon", epsilon)
    custom_force.addGlobalParameter("sigma", sigma)
  
    # Set periodic cutoff for nonbonded interactions    
    custom_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    custom_force.setCutoffDistance(cutoff)
      
      
    for particle_index in range(nparticles):
      system.addParticle(mass)      
      custom_force.addParticle([])
      
    # Add custom bond force to match the original GAMD simulation    
    fluid = testsystems.LennardJonesFluid(nparticles=nparticles, reduced_density=0.50, shift=False,  dispersion_correction=True)
    [topology, system, positions] = fluid.topology, fluid.system, fluid.positions  


     
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            original_nonbonded_force = force
            print("Got nonbonded force.")
                
    for index in range(original_nonbonded_force.getNumExceptions()):
        j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
        custom_force.addExclusion(j, k)
        print("Got exceptions for non-bonded force field.")
    
    ONE_4PI_EPS0 = 138.935456
    eps_solvent = original_nonbonded_force.getReactionFieldDielectric()
    krf = (1/ (cutoff**3)) * (eps_solvent - 1) / (2*eps_solvent + 1)
    crf = (1/ cutoff) * (3* eps_solvent) / (2*eps_solvent + 1)


    energy_expression  = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod*(1/r + krf*r*r - crf);"
    energy_expression += "krf = {:f};".format(krf.value_in_unit(unit.nanometer**-3))
    energy_expression += "crf = {:f};".format(crf.value_in_unit(unit.nanometer**-1))
    energy_expression += "ONE_4PI_EPS0 = {:f};".format(ONE_4PI_EPS0)
    custom_bond_force = mm.CustomBondForce(energy_expression)
    custom_bond_force.addPerBondParameter('chargeprod')
    custom_bond_force.addPerBondParameter('sigma')
    custom_bond_force.addPerBondParameter('epsilon')
    
    for index in range(original_nonbonded_force.getNumExceptions()):
        j, k, chargeprod, sigma, epsilon = original_nonbonded_force.getExceptionParameters(index)
        custom_bond_force.addBond(j, k, [chargeprod, sigma, epsilon])
        print("Added bond to account for exception is nonbonded force field.")
    
    system.addForce(custom_force)
    system.addForce(custom_bond_force)

    # Define initial positions
    positions = subrandom_particle_positions(nparticles, system.getDefaultPeriodicBoxVectors())
            
    # Create topology.
    topology = app.Topology()
    element = app.Element.getBySymbol('Ar')
    chain = topology.addChain()
    for particle in range(system.getNumParticles()):
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)
    topology = topology    
       
    
    R = get_rotation_matrix()
    positions = positions.value_in_unit(unit.angstrom)
    positions, off = center_positions(positions)
    positions = np.matmul(positions, R)
    positions += off
    positions += np.random.randn(positions.shape[0], positions.shape[1]) * 0.005
    positions *= unit.angstrom

    timestep = DT * unit.femtoseconds
    temperature = 100 * unit.kelvin
    chain_length = 10
    friction = 25. / unit.picosecond
    num_mts = 5
    num_yoshidasuzuki = 5

    integrator1 = integrators.NoseHooverChainVelocityVerletIntegrator(system,
                                                                      temperature,
                                                                      friction,
                                                                      timestep, chain_length, num_mts, num_yoshidasuzuki)


    platform = Platform.getPlatformByName("CUDA")
    platformProperties = {'Precision': 'mixed', 'DeviceIndex': '0, 1, 2'}


    simulation = Simulation(topology, system, integrator1, platform, platformProperties)
    
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.minimizeEnergy(tolerance=1*unit.kilojoule/(unit.mole*unit.nanometer))
    simulation.step(1)

    os.makedirs(f'./lj_data_ours/run_6', exist_ok=True)
    stepsPerIter = 50
    totalIter = 1000
    totalSteps = stepsPerIter * totalIter
    dataReporter_gt = StateDataReporter(f'./lj_data_ours/run_6/log_nvt_lj_{seed}.txt', stepsPerIter, totalSteps=totalSteps,
        step=True, time=True, speed=True, progress=True, elapsedTime=True, remainingTime=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True,
                                     separator='\t')
    simulation.reporters.append(dataReporter_gt)

    for t in range(totalIter):
        #if (t+1)%100 == 0:
        #    print(f'Finished {(t+1)*stepsPerIter} steps')
        state = simulation.context.getState(getPositions=True,
                                             getVelocities=True,
                                             getForces=True,
                                             enforcePeriodicBox=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        np.savez(f'./lj_data_ours/run_6/data_{seed}_{t}.npz',
                 pos=pos,
                 vel=vel,
                 forces=force)
        simulation.step(stepsPerIter)

