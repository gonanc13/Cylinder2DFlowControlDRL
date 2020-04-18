"""Resume and use the environment in the configuration version number 2.
   Here you set the environment parameters.
   You can modify, for example, the position of the probes, reward function used, etc.

   Define:
   -geometry parameters
   -mesh parameters

   -profile: Inflow profile
   -mu, rho: flow parameters - dynamic visc, density --> Re

   -simulation_duration: Simulation duration
   -dt: timestep
"""

import sys
import os
import shutil

# Add parent directory to path (to import Env2DCylinder)
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from Env2DCylinder import Env2DCylinder
import numpy as np
from dolfin import Expression
from printind.printind_function import printi, printiv
import math


nb_actuations = 80 # Number of actions (NN actuations) taken per episode

def resume_env(plot=False, # To plot results (Field, controls, lift, drag, rec area) during training
               step=50, # How many steps the control value is kept constant for (Not sure this is it!!!)
               dump=100, # Related to output data
               remesh=False,
               random_start=False,
               single_run=False):  # To do a single run in deterministic mode
    # ---------------------------------------------------------------------------------
    # the configuration version number 1

    # Spatial and time quantitites seem to be non-dimensionalized by 0.1
    # Length scale = D = 0.1
    # Velocity scale = U = 1
    # Time scale = D/U = 0.1

    simulation_duration = 2.0 # Duration of the simulation in seconds. In non-dimensional timw this equates to 20.0
    dt=0.0005 # Dimensional timestep in seconds. In non-dim this equates to 20.0


    root = 'mesh/turek_2d' # Define mesh path root
    if(not os.path.exists('mesh')):
        os.mkdir('mesh')

    jet_angle = 0 # ref angle for jets (deg)

    geometry_params = {'output': '.'.join([root, 'geo']), # output path: mesh/turek_2d.geo
                    'length': 2.2, # domain length (m)
                    'front_distance': 0.05 + 0.15, # dist from cyl center to inlet (m)
                    'bottom_distance': 0.05 + 0.15, # dist from cyl center to bottom wall (m)
                    'jet_radius': 0.05, # cylinder radius (m)
                    'width': 0.41, # domain width (m)
                    'cylinder_size': 0.01, # mesh size close to cyl wall (m)
                    'coarse_size': 0.1, # mesh size close to outflow (m)
                    'coarse_distance': 0.5, # disrance from cyl (center?) where coarsening starts (m)
                    'box_size': 0.05, # mesh size on wall (m)
                    'jet_positions': [90+jet_angle, 270-jet_angle], # position of jets in deg from jet_angle ref
                    'jet_width': 10, # jet width from center of cyl (deg)
                    'clscale': 0.25, # ????
                    'template': '../geometry_2d.template_geo',  # sets rel path of geom template
                    'remesh': remesh} # remesh toggle (from args)

    def profile(mesh, degree): # define inflow profile
        '''
           Time independent inflow profile.
        '''
        bot = mesh.coordinates().min(axis=0)[1] # get bottom boundary of profile
        top = mesh.coordinates().max(axis=0)[1] # get top boundary of profile
        print bot, top
        H = top - bot  # Domain height

        Um = 1.5 # ???

        return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                        '0'), bot=bot, top=top, H=H, Um=Um, degree=degree) # TODO: degree??
        # x[1] is equivalent to y

    flow_params = {'mu': 1E-3, # Dynamic viscosity. This in turn defines the Reynolds number: Re = U * D / mu
                  'rho': 1,  # Density
                  'inflow_profile': profile}  # flow_params['inflow_profile'] stores a reference to the profile function

    solver_params = {'dt': dt}

    # Define probe positions
    list_position_probes = []  # Initialise list of (x,y) np arrays with positions

    # The 9 'columns' of 7 probes downstream of the cylinder
    positions_probes_for_grid_x = [0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

    # The 4 'columns' of 4 probes on top and bottom of the cylinder
    positions_probes_for_grid_x = [-0.025, 0.0, 0.025, 0.05]
    positions_probes_for_grid_y = [-0.15, -0.1, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))

    # The two circles of 36 probes each (one every 10 degrees)
    list_radius_around = [geometry_params['jet_radius'] + 0.02, geometry_params['jet_radius'] + 0.05]
    list_angles_around = np.arange(0, 360, 10)

    for crrt_radius in list_radius_around:
        for crrt_angle in list_angles_around:
            angle_rad = np.pi * crrt_angle / 180.0 # Convert to rad
            list_position_probes.append(np.array([crrt_radius * math.cos(angle_rad), crrt_radius * math.sin(angle_rad)]))

    output_params = {'locations': list_position_probes,  # List of (x,y) np arrays with probe positions
                     'probe_type': 'pressure'  # Set that probes will be measuring the pressure
                     }

    optimization_params = {"num_steps_in_pressure_history": 1,  # ???
                        "min_value_jet_MFR": -1e-2,  # Set min and max Q* for weak actuation... paper says 0.06???
                        "max_value_jet_MFR": 1e-2,
                        "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80),  # Parameter to smooth??? Gives 0.1 with usual params
                        "zero_net_Qs": True,  # True if Q1 + Q2 = 0
                        "random_start": random_start}  # Should we random start??? How does this work??

    # See later what these inspection params do (??)
    inspection_params = {"plot": plot,
                        "step": step,
                        "dump": dump,
                        "range_pressure_plot": [-2.0, 1],
                        "range_drag_plot": [-0.175, -0.13],
                        "range_lift_plot": [-0.2, +0.2],
                        "line_drag": -0.1595,
                        "line_lift": 0,
                        "show_all_at_reset": False,
                        "single_run":single_run
                        }

    reward_function = 'drag_plain_lift'  # Choose reward function

    verbose = 0  # ?

    number_steps_execution = int((simulation_duration/dt)/nb_actuations)

    # ---------------------------------------------------------------------------------
    # do the initialization

    # If we remesh, we sim with no control until a well-developed unsteady wake is obtained. That state is saved and
    # used as a start for each subsequent learning episode.

    # We set the value of n-iter (no. iterations to calculate converged initial state) depending on if we remesh
    if(remesh):
        n_iter = int(5.0 / dt)  # 5 seconds / dt --> Number of iterations
        if(os.path.exists('mesh')):
            shutil.rmtree('mesh')   # If previous mesh directory exists, we delete it
        os.mkdir('mesh')  # Create new empty mesh directory
        print("Make converge initial state for {} iterations".format(n_iter))  # Log
    else:
        n_iter = None



    #Processing the name of the simulation (to be used in outputs)
    simu_name = 'Simu'  # 'root' of the name

    if (geometry_params["jet_positions"][0] - 90) != 0:
        next_param = 'A' + str(geometry_params["jet_positions"][0] - 90)
        simu_name = '_'.join([simu_name, next_param])     # e.g: if jet_pos[0] = 100 --> simu_name += '_A10'
    if geometry_params["cylinder_size"] != 0.01:
        next_param = 'M' + str(geometry_params["cylinder_size"])[2:]
        simu_name = '_'.join([simu_name, next_param])     # e.g: if cyl_size (mesh) = 0.025 --> simu_name += '_M25'
    if optimization_params["max_value_jet_MFR"] != 0.01:
        next_param = 'maxF' + str(optimization_params["max_value_jet_MFR"])[2:]
        simu_name = '_'.join([simu_name, next_param])   # e.g: if max_MFR = 0.09 --> simu_name += '_maxF9'
    if nb_actuations != 80:
        next_param = 'NbAct' + str(nb_actuations)
        simu_name = '_'.join([simu_name, next_param])   # e.g: if max_MFR = 100 --> simu_name += '_NbAct100'

    # Now add a ref to the reward function used to the name.
    next_param = 'drag'
    if reward_function == 'recirculation_area':
        next_param = 'area'
    if reward_function == 'max_recirculation_area':
        next_param = 'max_area'
    elif reward_function == 'drag':
        next_param = 'last_drag'
    elif reward_function == 'max_plain_drag':
        next_param = 'max_plain_drag'
    elif reward_function == 'drag_plain_lift':
        next_param = 'lift'
    elif reward_function == 'drag_avg_abs_lift':
        next_param = 'avgAbsLift'
    simu_name = '_'.join([simu_name, next_param])

    # Pass parameters to the Environment class
    env_2d_cylinder = Env2DCylinder(path_root=root,
                                    geometry_params=geometry_params,
                                    flow_params=flow_params,
                                    solver_params=solver_params,
                                    output_params=output_params,
                                    optimization_params=optimization_params,
                                    inspection_params=inspection_params,
                                    n_iter_make_ready=n_iter,  # We recalculate if necessary
                                    verbose=verbose,
                                    reward_function=reward_function,
                                    number_steps_execution=number_steps_execution,
                                    simu_name = simu_name)

    return(env_2d_cylinder)  # resume_env() returns instance of Environment object
