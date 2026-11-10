"""
Note:
In python, a 3d array has a shape of (1, 1, 1) is equivalent to a 1-meter cube.
"""
import argparse
import random
import sys
import os
from typing import *
from math import degrees
import csv
import time
import json

# Blender packages
import bpy
import bmesh


def create_cylinder(radius, height, vertices=32, mass=100.0):
    """
    This function creates a cylinder in the scene, then the cylinder was set into rigid body
    :param radius: radius of the cylinder
    :param height: height of the cylinder
    :param vertices: number of vertices in the cylinder
    :param mass: mass of the cylinder
    """
    for obj in bpy.data.objects:
        if 'Cylinder' in obj.name:
            print('\tERR: A cylinder has already been added.')
            return
    bpy.ops.object.select_all(action='DESELECT')  # Deselect all objects
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=height)  # Add cylinder to the scene
    cylinder = bpy.context.active_object  # Ensure the active object is a mesh
    cylinder.location.z += height / 2  # Move the cylinder up
    bpy.ops.object.mode_set(mode='EDIT')  # Switch to Edit Mode
    bpy.ops.mesh.select_all(action='DESELECT')  # Deselect all elements
    bpy.context.tool_settings.mesh_select_mode = (False, False, True)  # Switch to face select mode
    mesh = bmesh.from_edit_mesh(cylinder.data)  # Make sure using the mesh from edit mode
    mesh.faces.ensure_lookup_table()  # Ensure the mesh is up-to-date
    face = mesh.faces[-4]  # Access the top face
    face.select = True  # Select the top face
    bpy.ops.mesh.delete(type='FACE')  # Delete the selected face
    bpy.ops.object.mode_set(mode='OBJECT')  # Switch back to Object Mode when done
    bpy.ops.rigidbody.object_add()  # Set the rigid body properties
    cylinder.rigid_body.type = 'PASSIVE'  # Set the rigid body type to Active
    cylinder.rigid_body.collision_shape = 'MESH'  # Set collision type to Mesh
    cylinder.rigid_body.mass = mass  # Set mass of the object


def add_particle_meshes(path: str, cylinder_radius: float, cylinder_height: float, mesh_list_path: str = None,
                        start_idx: int = 0, n_imports: int = 0, batch: int = 1, seed: Optional[int] = None) -> List[str]:
    """
    Add a given number of particle meshes to the blender scene
    :param mesh_list_path: a dedicated list of meshes involved in the simulation
    :param path: path to a list of particles
    :param start_idx: start to import the particles at the index
    :param n_imports: number of imports, defaults to 0 means import all
    :param batch: n-th import batch
    :param seed: the seed for random shuffle, if seed=None, don't shuffle
    :return: added meshes name
    """
    particle_meshes = sorted(os.listdir(path))
    if mesh_list_path is not None:
        with open(mesh_list_path) as file:
            print("Using custom list of particles")
            reader = csv.reader(file)
            filtered_meshes = set([f'{row[0]}.obj' for row in reader])
            particle_meshes = sorted(list(set(particle_meshes).intersection(filtered_meshes)))
            print("Finished updating list of particles")

    if seed is not None:
        random.seed(seed)
        random.shuffle(particle_meshes)

    n_particles = len(particle_meshes)
    if n_particles == 0:
        print(f'FATAL ERR: No particle meshes found at {path}', file=sys.stderr)
        sys.exit(0)
    n_imports = n_particles if n_imports <= 0 else n_imports
    start_idx = max(min(start_idx, n_particles - 1), 0)
    end_idx = min(start_idx + n_imports - 1, n_particles - 1)

    # Check if the current batch of particles is added
    for obj in bpy.data.objects:
        if obj.name.startswith(f'{batch:02d}'):
            print('\tERR: Current batch has already been added.')
            return []
    print("Start importing meshes")
    # Add particles to the blender scene
    new_obj_names = []
    for import_idx in range(start_idx, end_idx + 1):
        mesh_name = particle_meshes[import_idx - 1]
        if mesh_name.endswith('.obj'):
            blender_obj_name = f'{batch:02d}_{mesh_name[:-4]}'
            bpy.ops.wm.obj_import(filepath=os.path.join(path, mesh_name))
            imported_object = bpy.context.selected_objects[0]
            err_msg = ''
            dim_x, dim_y, dim_z = imported_object.dimensions.x, imported_object.dimensions.y, imported_object.dimensions.z
            if max(max(dim_x, dim_y), dim_z) > 2 * cylinder_radius:
                err_msg += f'Particle size exceeds cylinder diameter. (x:{dim_x:.1f} y:{dim_y:.1f} z: {dim_z:.1f}>{cylinder_radius * 2:.1f}).\n'
            if max(max(dim_x, dim_y), dim_z) > 2 * cylinder_height:
                err_msg += f'Particle size exceeds cylinder diameter. (x:{dim_x:.1f} y:{dim_y:.1f} z: {dim_z:.1f}>{cylinder_height * 2:.1f}).\n'
            if len(err_msg) > 0:
                print(f'FATAL ERR: Cylinder size check failed.\n{err_msg}', file=sys.stderr)
                sys.exit(1)
            imported_object.name = blender_obj_name
            imported_object.data.name = blender_obj_name
            new_obj_names.append(blender_obj_name)
    return new_obj_names


def update_particle_meshes_properties(obj_names: List[str], start_frame: int, end_frame: int, z_offset: float = 50.0):
    """
    Update properties of a list of objects by its name.
    Changed properties:
    - set object origin: geometry center
    - set object rigid body
    - set object mass: x * y * z / 5.0
    - set object rotation: no rotation
    - set object location: lastly updated location + 0.5 * object z dimension
    - set object scale and movement key frame at `frame`
    :param obj_names: name of a list of objects to update
    :param start_frame: the start frame to update the key framed properties
    :param end_frame: the end frame to update the key framed properties
    :param z_offset: placement of the initial particle in z axis
    """
    if len(obj_names) == 0:
        print('\nERR: No objects to update.')
        return
    obj_location_z = z_offset
    obj_z_padding = 5.0
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        # Approximate the mass by its size
        obj.rigid_body.mass = obj.dimensions.x * obj.dimensions.y * obj.dimensions.z / 5.0
        obj.rotation_euler = (0.0, 0.0, 0.0)
        obj.location = (0.0, 0.0, obj_location_z + obj.dimensions.z / 2)
        obj_location_z += obj.dimensions.z + obj_z_padding
        # Set start key frame properties
        # - Disables movement
        # - Sets collision collection to batch index
        # - Enables movement
        # - Set collision collection to 0
        obj.rigid_body.enabled = False
        obj.keyframe_insert(data_path='rigid_body.enabled', frame=start_frame)
        obj.rigid_body.collision_collections[0] = False
        obj.rigid_body.collision_collections[int(obj_name[:2])] = True
        obj.keyframe_insert(data_path='rigid_body.collision_collections', frame=start_frame)
        obj.rigid_body.collision_collections[int(obj_name[:2])] = False
        obj.rigid_body.collision_collections[0] = True
        obj.keyframe_insert(data_path='rigid_body.collision_collections', frame=start_frame + 1)
        obj.rigid_body.enabled = True
        obj.keyframe_insert(data_path='rigid_body.enabled', frame=start_frame + 1)
        # Set end key frame properties
        # - Enables movement
        # - Disables movement
        obj.rigid_body.enabled = True
        obj.keyframe_insert(data_path="rigid_body.enabled", frame=end_frame)
        obj.rigid_body.enabled = False
        obj.keyframe_insert(data_path="rigid_body.enabled", frame=end_frame + 1)
        obj.select_set(False)


def init_z_offset(obj_names: List[str]) -> float:
    """
    Initialises the z-offset by finding the maximum height of the objects
    :param obj_names: name of a list of objects to find the init z offset
    :return: the maximum size along z axis
    """
    max_z = 0.0
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        max_z = max(max_z, obj.dimensions.z)
    return max_z


def check_cylinder_size(obj_names: List[str], cylinder_radius: float, cylinder_height: float):
    """
    Check if the size of the cylinder can accommodate the particles
    :param obj_names: name of a list of objects
    :param cylinder_radius: radius of the cylinder
    :param cylinder_height: height of the cylinder
    """
    max_x, max_y, max_z = 0.0, 0.0, 0.0
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        max_x = max(max_x, obj.dimensions.x)
        max_y = max(max_y, obj.dimensions.y)
        max_z = max(max_z, obj.dimensions.z)
    max_dim = max(max_x, max(max_y, max_z))
    err_msg = ''
    if max_dim > 2 * cylinder_radius:
        err_msg += f'Particle size exceeds cylinder diameter. ({max_dim:.1f}>{cylinder_radius * 2:.1f}).\n'
    if max_dim > cylinder_height:
        err_msg += f'Particle size exceeds cylinder height. ({max_dim:.1f}>{cylinder_height:.1f})\n'
    if len(err_msg) > 0:
        print(f'FATAL ERR: Cylinder size check failed.\n{err_msg}'
              f'\nMax x: {max_x}, Max y: {max_y}, Max z: {max_z}', file=sys.stderr)
        sys.exit(1)
    print('Cylinder size check passed.')


def update_z_offset(obj_names: List[str], old_z_offset: float):
    """
    Update the z offset based on a list of object names.
    Similar to `init_z_offset`.
    :param obj_names: name of a list of objects to find the new z offset
    :param old_z_offset: old z offset
    :return: updated z offset
    """
    new_z_offset = old_z_offset
    for obj_name in obj_names:
        obj = bpy.data.objects.get(obj_name)
        obj_loc = obj.matrix_world.to_translation().z
        vertex = obj_loc + (obj.dimensions.x ** 2 + obj.dimensions.y ** 2 + obj.dimensions.z ** 2) ** 0.5 * 0.5
        new_z_offset = max(new_z_offset, vertex)
    return new_z_offset


def simulate(start_frame=0, end_frame=1000, log_period=100):
    """
    Simulate n frames.
    :param start_frame: render start frame
    :param end_frame: render end frame
    """
    bpy.context.scene.frame_set(start_frame)
    cnt = 1
    for frame in range(start_frame, end_frame + 1):
        if cnt % log_period == 0:
            print(f'Simulation in progress: {cnt / (end_frame - start_frame) * 100:.1f}% ({cnt}/{end_frame - start_frame})')
        cnt += 1
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()


def export_simulation(args):
    """
    Export the simulated result.
    :param export_path: export path
    """
    export_time = int(time.time())
    export_csv = os.path.join(args.out_dir, f'simulation_result_{export_time}.csv')
    export_json = os.path.join(args.out_dir, f'simulation_result_{export_time}.json')
    with open(export_csv, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['particle', 'rx', 'ry', 'rz', 'tx', 'ty', 'tz'])
        for obj in bpy.context.scene.objects:
            if "_" in obj.name:
                obj.select_set(True)
                rotation_euler = obj.matrix_world.to_euler('XYZ')
                degs = list(map(degrees, rotation_euler))
                trans = obj.matrix_world.to_translation()
                row_content = [obj.name] + degs + [trans.x, trans.y, trans.z]
                writer.writerow(row_content)
                obj.select_set(False)
    with open(export_json, 'w') as file:
        metadata = {
            'Input directory':     args.in_dir,
            'Mesh list directory': args.mesh_list_dir,
            'Output directory':    args.out_dir,
            'Cylinder radius':     args.radius,
            'Cylinder height':     args.height,
            'NO. imports':         args.n_imports,
            'NO. batches':         args.batches,
            'Random seed':         args.seed,
            'Render frames':       args.frames
        }
        json.dump(metadata, file, indent=4)
    return f'simulation_result_{export_time}'


def _parse_args():
    class ArgumentParserForBlender(argparse.ArgumentParser):
        """
        Source:
        > https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
        This class is identical to its superclass, except for the parse_args
        method (see docstring). It resolves the ambiguity generated when calling
        Blender from the CLI with a python script, and both Blender and the script
        have arguments. E.g., the following call will make Blender crash because
        it will try to process the script's -a and -b flags:

        To bypass this issue this class uses the fact that Blender will ignore all
        arguments given after a double-dash ('--'). The approach is that all
        arguments before '--' go to Blender, arguments after go to the script.
        The following calls work fine:
        """
        @staticmethod
        def _get_argv_after_doubledash():
            """
            Given the sys.argv as a list of strings, this method returns the
            sublist right after the '--' element (if present, otherwise returns
            an empty list).
            """
            try:
                idx = sys.argv.index("--")
                return sys.argv[idx + 1:]  # the list after '--'
            except ValueError:  # '--' not in the list:
                return ""

        # overrides superclass
        def parse_args(self, **kwargs):
            """
            This method is expected to behave identically as in the superclass,
            except that the sys.argv list will be pre-processed using
            _get_argv_after_doubledash before. See the docstring of the class for
            usage examples and details.
            """
            return super().parse_args(args=self._get_argv_after_doubledash())

    # Parse arguments
    parser = ArgumentParserForBlender(description='A program to run particle simulation in blender')
    parser.add_argument('-i', '--in_dir', type=str, metavar='path', help='input path')
    parser.add_argument('-m', '--mesh_list_dir', metavar='path', help='dir to a dedicated list of meshes')
    parser.add_argument('-o', '--out_dir', type=str, metavar='path', help='output path')
    parser.add_argument('-r', '--radius', type=float, metavar='float', default=100,
                        help='cylinder radius')
    parser.add_argument('-c', '--height', type=float, metavar='float', default=1000,
                        help='cylinder height')
    parser.add_argument('-n', '--n_imports', type=int, metavar='int', default=10,
                        help='number of particles to import per batch')
    parser.add_argument('-s', '--seed', type=int, metavar='int', default=None,
                        help='random seed controls how the simulation will be performed')
    parser.add_argument('-b', '--batches', type=int, metavar='int', default=1,
                        help='number of batch of import, default=1')
    parser.add_argument('-f', '--frames', type=int, metavar='int', default=1000,
                        help='number of frames in rendering, default=1000')
    args = parser.parse_args()
    return args


def main():
    """
    Workflow:
    1. Create cylinder
    2. Start simulation cycles:
        - add particle meshes
        - if n == 0
            -> initialise z offset with added meshes
          else
            -> update z offset
        - set particle properties
        - start render
        - freeze particles
    3. Export result
    """
    args = _parse_args()
    if args.batches > 18:
        print('FATAL ERR: Due to Blender restriction, a maximum of 18 batches can be added', file=sys.stderr)
        sys.exit(1)

    print(f'Simulation Settings\n'
          f'Input directory:     {args.in_dir}\n'
          f'Mesh list directory: {args.mesh_list_dir}\n'
          f'Output directory:    {args.out_dir}\n'
          f'Cylinder radius:     {args.radius}\n'
          f'Cylinder height:     {args.height}\n'
          f'NO. imports:         {args.n_imports}\n'
          f'NO. batches:         {args.batches}\n'
          f'Random seed:         {args.seed}\n'
          f'Render frames:       {args.frames}')
    # Blender scene setup
    scene = bpy.context.scene
    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.point_cache.frame_end = args.batches * args.frames + 50
    scene.rigidbody_world.substeps_per_frame = 50
    bpy.context.scene.frame_set(0)
    bpy.context.view_layer.update()

    # Simulation workflow starts
    create_cylinder(radius=args.radius, height=args.height)
    added_meshes, z_offset = [], 0
    for batch in range(1, args.batches + 1):
        print(f'\nStart batch {batch}/{args.batches} simulation.')
        start_frame = (batch - 1) * args.frames
        end_frame = batch * args.frames
        # Add particle to simulation scene
        if batch == 1:
            added_meshes = add_particle_meshes(path=args.in_dir, cylinder_radius=args.radius, cylinder_height=args.height,
                                               mesh_list_path=args.mesh_list_dir, start_idx=1,
                                               n_imports=args.n_imports, batch=batch, seed=args.seed)
            z_offset = init_z_offset(added_meshes) + 20
        else:
            z_offset = update_z_offset(added_meshes, z_offset)
            added_meshes = add_particle_meshes(path=args.in_dir, cylinder_radius=args.radius, cylinder_height=args.height,
                                               mesh_list_path=args.mesh_list_dir, start_idx=1,
                                               n_imports=args.n_imports, batch=batch, seed=args.seed)
        # Update particle's properties
        print(f'Finished import {args.n_imports} particle meshes.')
        update_particle_meshes_properties(added_meshes,
                                          start_frame=start_frame, end_frame=end_frame,
                                          z_offset=z_offset)
        print(f'Finished updating {args.n_imports} meshes.')
        print(f'Start simulating {args.frames} frames.')
        # Start simulation
        start = time.time()
        simulate(start_frame, end_frame)
        end = time.time()
        print(f'Finished simulating {args.frames} frames, simulation speed: {args.frames / (end - start):.1f} FPS.')
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = export_simulation(args)
    out_csv, out_json = os.path.join(args.out_dir, out_file + '.csv'), os.path.join(args.out_dir, out_file + '.json')
    print(f'Finished simulation. Result saved in:\n{out_csv}\nMetadata saved in:\n{out_json}\nProgram exit.')


"""
Example usage::
blender --background --python ./simulation.py -- -i /path/to/data/mesh/toy/ -o /path/to/data/simulation_results/ -n 10 -b 4 -f 500
"""
if __name__ == "__main__":
    main()
