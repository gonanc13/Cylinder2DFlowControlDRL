import os, subprocess
from numpy import deg2rad
from printind.printind_function import printiv

# Args will be the geometry params (or cmd line args when called from there)
# dim is domain dimension (2 for 2D) for meshing
def generate_mesh(args, template='rectangle_2d.template_geo', dim=2):
    '''Modify template according to args (geom_params) and make gmsh generate the mesh'''

    assert os.path.exists(template)  # Raise an error if no template
    args = args.copy()
    printiv(template)

    output = args.pop('output')  # -> output = 'mesh/turek_2d.geo'
    printiv(output)

    # os.path.splitext() splits the path name into a pair root and ext
    if not output:
        output = template
    assert os.path.splitext(output)[1] == '.geo'  # return error if output doesnt have .geo extension

    scale = args.pop('clscale')  # Get mesh size scaling ratio

    cmd = 'gmsh -0 %s ' % output   # Create cmd string to output unrolled geometry

    list_geometric_parameters = ['jets_toggle', 'jet_width', 'height_cylinder', 'ar', 'cylinder_y_shift',
                                 'x_upstream', 'x_downstream', 'height_domain',
                                 'mesh_size_cylinder', 'mesh_size_wall', 'mesh_size_coarse', 'coarse_distance']

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
    parser.add_argument('-jets_toggle', default=1, type=bool,
                        help='toggle Jets --> 0 : No jets, 1: Yes jets')
    parser.add_argument('-jet_width', default=0.1, type=float,
                        help='Jet Width')
    parser.add_argument('-height_cylinder', default=40, type=float,
                        help='Cylinder Height')
    parser.add_argument('-ar', default=1, type=float,
                        help='Cylinder Aspect Ratio')
    parser.add_argument('-cylinder_y_shift', default=80, type=float,
                        help='Cylinder Center Shift from Centerline, Positive UP')
    parser.add_argument('-x_upstream', default=0.25, type=float,
                        help='Domain Upstream Length (from left-most rect point)')
    parser.add_argument('-x_downstream', default=5, type=float,
                        help='Domain Downstream Length (from right-most rect point)')
    parser.add_argument('-height_domain', nargs='+', default=[0, 60, 120, 180, 240, 300],
                        help='Domain Height')
    parser.add_argument('-mesh_size_cylinder', default=10, type=float,
                        help='Mesh Size on Cylinder Walls')
    parser.add_argument('-mesh_size_wall', default=10, type=float,
                        help='Mesh Size on Channel Walls')
    parser.add_argument('-mesh_size_coarse', default=10, type=float,
                        help='Mesh Size Close to Outflow')
    parser.add_argument('-coarse_distance', default=10, type=float,
                        help='Distance From Cylinder Right-Most Point Where Coarsening Starts')


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
