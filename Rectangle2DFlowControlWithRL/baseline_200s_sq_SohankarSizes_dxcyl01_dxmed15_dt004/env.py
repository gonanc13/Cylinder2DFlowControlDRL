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
               step=50, # If verbose > 0, every "step" steps print (step,Qs,probe values, drag, lift, recArea) to screen (see def output_data)
               dump=100, # If not False, we generate output.csv with episode averages and
               remesh=False,
               random_start=False,
               single_run=False):  # To do a single run in deterministic mode
    # ---------------------------------------------------------------------------------
    # the configuration version number 1

    # Spatial and time quantitites seem to be non-dimensionalized by 0.1

    simulation_duration = 20.0 # Duration of the simulation in seconds. In non-dimensional time this equates to 20.0 (***)
    dt=0.004 # Dimensional timestep in seconds. In non-dim this equates to 0.005 (***)

    root = 'mesh/turek_2d' # Define mesh path root
    if(not os.path.exists('mesh')):
        os.mkdir('mesh')

    geometry_params = {'output': '.'.join([root, 'geo']), # output path: mesh/turek_2d.geo
                    'clscale': 1, # mesh size scaling ratio
                    'template': '../rectangle_2d.template_geo', # sets rel path of geom template
                    'remesh': remesh, # remesh toggle (from resume_env args))
                    'jets_toggle': 1,  # toggle Jets --> 0 : No jets, 1: Yes jets
                    'jet_width': 0.1,  # Jet Width  (***)
                    'height_cylinder': 1,  # Cylinder Height (***)
                    'ar': 1.0,  # Cylinder Aspect Ratio
                    'cylinder_y_shift': 0,  # Cylinder Center Shift from Centerline, Positive UP  (***)
                    'x_upstream': 10,  # Domain Upstream Length (from left-most rect point)  (***)
                    'x_downstream': 26,  # Domain Downstream Length (from right-most rect point)  (***)
                    'height_domain': 20,  # Domain Height  (***)
                    'mesh_size_cylinder': 0.01,  # Mesh Size on Cylinder Walls
                    'mesh_size_medium': 0.15,  # Medium mesh size (at boundary where coarsening starts
                    'mesh_size_coarse': 1,  # Coarse mesh Size Close to Domain boundaries outside wake
                    'coarse_y_distance_top_bot': 4,  # y-distance from center where mesh coarsening starts
                    'coarse_x_distance_left_from_LE': 2.5}  # x-distance from upstream face where mesh coarsening starts


    profile = Expression(('1','0'), degree=2)

    flow_params = {'mu': 1E-2, # Dynamic viscosity. This in turn defines the Reynolds number: Re = U * D / mu  (***)
                  'rho': 1,  # Density
                  'inflow_profile': profile}  # flow_params['inflow_profile'] stores a reference to the profile function

    solver_params = {'dt': dt}

    # Define probe positions  (***)
    list_position_probes = []  # Initialise list of (x,y) np arrays with positions

    cyl_height = geometry_params['height_cylinder']
    ar = geometry_params['ar']
    cyl_length = ar * cyl_height

    # The 9 'columns' of 7 probes downstream of the cylinder
    positions_probes_x_dist_from_right = [0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    positions_probes_for_grid_x = [x + cyl_length/2 for x in positions_probes_x_dist_from_right]
    positions_probes_for_grid_y = [-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

    # The 4 'columns' of 4 probes on top and bottom of the cylinder
    positions_probes_for_grid_x = [-cyl_length/4, 0.0, cyl_length/4, cyl_length/2]
    positions_probes_for_grid_y = [-0.15, -0.1, 0.1, 0.15]

    for crrt_x in positions_probes_for_grid_x:
        for crrt_y in positions_probes_for_grid_y:
            list_position_probes.append(np.array([crrt_x, crrt_y]))


    # The two rectangles of 36 probes each (for now, 10 probes on each side --> 36 total as corners are shared)
    # TODO: Make the distribution even as AR changes (scalable)
    # TODO: Current way leads to corner duplicate probes!!!!

    for offset in [0.02, 0.05]:
        dist_probes_x = (cyl_length + offset * 2) / 9  # Dist between probes on top and bottom sides of periferic
        dist_probes_y = (cyl_height + offset * 2) / 9  # Dist between probes on left and right sides of periferic
        left_side_periferic_x = -cyl_length / 2 - offset  # x coord of left side of periferic
        bot_side_periferic_y = -cyl_height / 2 - offset # y coord of bot side of periferic

        # Define top and bot probes
        positions_probes_for_grid_x = [left_side_periferic_x + dist_probes_x * i for i in range(10)]
        positions_probes_for_grid_y = [bot_side_periferic_y, cyl_height / 2 + offset]

        for crrt_x in positions_probes_for_grid_x:
            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

        # Define left and right probes
        positions_probes_for_grid_x = [left_side_periferic_x, cyl_length / 2 + offset]
        positions_probes_for_grid_y = [bot_side_periferic_y + dist_probes_y * i for i in range(10)]

        for crrt_x in positions_probes_for_grid_x:
            for crrt_y in positions_probes_for_grid_y:
                list_position_probes.append(np.array([crrt_x, crrt_y]))  # Append (x,y) pair array

    output_params = {'locations': list_position_probes,  # List of (x,y) np arrays with probe positions
                     'probe_type': 'pressure'  # Set that probes will be measuring the pressure
                     }

    optimization_params = {"num_steps_in_pressure_history": 1, # Number of steps that constitute an environment state (state shape = this * len(locations))
                        "min_value_jet_MFR": -1e-2,  # Set min and max Q* for weak actuation... paper says 0.06???
                        "max_value_jet_MFR": 1e-2,
                        "smooth_control": (nb_actuations/dt)*(0.1*0.0005/80),  # parameter alpha to smooth out control. Gives 0.1 with usual params
                        "zero_net_Qs": True,  # True if Q1 + Q2 = 0
                        "random_start": random_start}  # Should we random start??? How does this work??

    # See later what these inspection params do (??)
    inspection_params = {"plot": plot,
                        "step": step,
                        "dump": dump,
                        "range_pressure_plot": [-2.0, 1], # ylim for pressure plot
                        "range_drag_plot": [-0.175, -0.13],  # ylim for drag plot
                        "range_lift_plot": [-0.2, +0.2],  # ylim for lift plot
                        "line_drag": -0.1595,  # Mean drag without control (TODO)
                        "line_lift": 0,
                        "show_all_at_reset": False,
                        "single_run": single_run
                        }

    reward_function = 'drag_plain_lift'  # Choose reward function

    verbose = 0  # For detailed output (see Env2DRectangle)

    number_steps_execution = int((simulation_duration/dt)/nb_actuations)  # Number of steps over which NN control output is kept constant

    # ---------------------------------------------------------------------------------
    # do the initialization

    # If we remesh, we sim with no control until a well-developed unsteady wake is obtained. That state is saved and
    # used as a start for each subsequent learning episode.

    # We set the value of n-iter (no. iterations to calculate converged initial state) depending on if we remesh
    if(remesh):
        n_iter = int(200.0 / dt)  # 20 seconds / dt --> Number of iterations
        if(os.path.exists('mesh')):
            shutil.rmtree('mesh')   # If previous mesh directory exists, we delete it
        os.mkdir('mesh')  # Create new empty mesh directory
        print("Make converge initial state for {} iterations".format(n_iter))  # Log
    else:
        n_iter = None



    #Processing the name of the simulation (to be used in outputs)
    simu_name = 'Simu'  # 'root' of the name

    if geometry_params["mesh_size_cylinder"] != 0.01:
        next_param = 'M' + str(geometry_params["mesh_size_cylinder"])[2:]
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
                                    simu_name=simu_name)

    return(env_2d_cylinder)  # resume_env() returns instance of Environment object
