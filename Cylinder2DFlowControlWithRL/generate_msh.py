import os, subprocess
from numpy import deg2rad
from printind.printind_function import printiv

# Args will be the geometry params (or cmd line args when called from there)
# dim is domain dimension (2 for 2D) for meshing
def generate_mesh(args, template='geometry_2d.template_geo', dim=2):
    '''Modify template according to args (geom_params) and make gmsh generate the mesh'''

    assert os.path.exists(template)  # Raise an error if no template
    args = args.copy()

    printiv(template)

    with open(template, 'r') as f: old = f.readlines()  # Read template and save each line as an item of the list 'old''

    # Chop the file to replace the jet positions

    # Lambda defines an anonymous function --> lambda arguments : expression
    # .startswith() method returns True if a string starts with the specified prefix
    # map(function, iterable) returns a list of the outputs of applying the function to each element of the iterable
    # .index() returns the index of an element in a list
    split = map(lambda s: s.startswith('DefineConstant'), old).index(True) # --> Return line number of DefineConstant

    # .pop() returns an item from the dict and removes it
    jet_positions = deg2rad(map(float, args.pop('jet_positions')))  # Convert list of jet pos to float and then to rad
    jet_positions = 'jet_positions[] = {%s};\n' % (', '.join(map(str, jet_positions))) # Create new jet_pos line for .geo
    body = ''.join([jet_positions] + old[split:]) # generate new .geo body with input jet pos

    output = args.pop('output')  # -> output = 'mesh/turek_2d.geo'
    printiv(output)

    # os.path.splitext() splits the path name into a pair root and ext
    if not output:
        output = template
    assert os.path.splitext(output)[1] == '.geo'  # return error if output doesnt have .geo extension

    with open(output, 'w') as f: f.write(body)  # write new body to target ('output') .geo file

    args['jet_width'] = deg2rad(args['jet_width'])  # Convert jet width to rad

    scale = args.pop('clscale')  # Mesh size scaling ratio

    cmd = 'gmsh -0 %s ' % output   # Create cmd string to output unrolled geometry

    list_geometric_parameters = ['width', 'jet_radius', 'jet_width', 'box_size', 'length',
                                 'bottom_distance', 'cylinder_size', 'front_distance',
                                 'coarse_distance', 'coarse_size']


    constants = " "

    for crrt_param in list_geometric_parameters:
        constants = constants + " -setnumber " + crrt_param + " " + str(args[crrt_param])  # Create cmd string to set params from DefineConstants

    # Create unrolled model with the geometry_params set
    subprocess.call(cmd + constants, shell=True)  # run the command to create unrolled

    # Assert that 'mesh/turek_2d.geo'_unrolled exists
    unrolled = '_'.join([output, 'unrolled'])
    assert os.path.exists(unrolled)

    return subprocess.call(['gmsh -%d -clscale %g %s' % (dim, scale, unrolled)], shell=True)
    # generate 2d mesh --> turek_2d.msh?
    # -clscale <float> : Set global mesh element size scaling factor
    # -2: Perform 2D mesh generation, then exit

# -------------------------------------------------------------------

if __name__ == '__main__':   # This is only run when this file is executed as a script, not when imported

    import argparse, sys #, petsc4py
    from math import pi

    # The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv.
    # generate argument help output for when "python generate_mesh.py -help" is called from the command line:

    parser = argparse.ArgumentParser(description='Generate msh file from GMSH',  # Text to display before the argument help
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # adds information about default values to each of the argument help messages
    # Optional output geo file
    parser.add_argument('-output', default='', type=str, help='A geofile for writing out geometry')
    # Geometry
    parser.add_argument('-length', default=200, type=float,
                        help='Channel length')
    parser.add_argument('-front_distance', default=40, type=float,
                        help='Cylinder center distance to inlet')

    parser.add_argument('-bottom_distance', default=40, type=float,
                        help='Cylinder center distance from bottom wall')
    parser.add_argument('-jet_radius', default=10, type=float,
                        help='Cylinder radius')
    parser.add_argument('-width', default=80, type=float,
                        help='Channel width')
    parser.add_argument('-cylinder_size', default=0.25, type=float,
                        help='Mesh size on cylinder')
    parser.add_argument('-box_size', default=5, type=float,
                        help='Mesh size on wall')
    # Jet parameters
    parser.add_argument('-jet_positions', nargs='+', default=[0, 60, 120, 180, 240, 300],
                        help='Angles of jet center points')
    parser.add_argument('-jet_width', default=10, type=float,
                        help='Jet width in degrees')

    # Refine geometry
    parser.add_argument('-clscale', default=1, type=float,
                        help='Scale the mesh size relative to give')

    args = parser.parse_args()

    # Using geometry_2d.geo to produce geometry_2d.msh
    sys.exit(generate_mesh(args.__dict__))

    # FIXME: inflow profile
    # FIXME: test with turek's benchmark

    # IDEAS: More accureate non-linearity handling
    #        Consider splitting such that we solve for scalar components
