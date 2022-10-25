import numpy as np
import os
import sys
import bpy
import config.blender_config as bconfig
from typing import *
import bmesh
from mathutils import Matrix, Vector, Euler, Quaternion
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


def random_unit_vectors(num_points: int, ndim: int = 3) -> np.ndarray:
    """
    Randomly generate unit vectors of dimension ndim

    :param num_points: number of vectors to generate
    :param ndim: dimension of vectors
    :returns: array of random unit vectors each of length ndim
    """
    vec = np.random.randn(ndim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def init_blender(configuration: bconfig.BlenderConfig) -> bpy.types.Scene:
    """ Initialize blender scene by deleting demo objects and applying settings from configuration

    :param configuration: A set of blender configurations to apply to the current scene
    :returns: the current active scene
    """
    context = bpy.context
    scene = context.scene
    for c in scene.collection.children:
        bpy.data.collections.remove(c, do_unlink=True)
    set_blender_data(scene, configuration)
    return scene


def set_blender_data(item: Any, config: Union[dict, bconfig.BlenderConfig, Any]) -> None:
    """
    This function recursively sets all settings in a blender object assuming the dictionary entries are
    structured and named correctly.

    :param item: This should be a blender Object or Scene, etc.
    :param config: a dictionary-like object with blender settings
    """
    if hasattr(config, '__getitem__'):
        for key in config:
            if hasattr(config[key], '__getitem__') and not isinstance(config[key], str):
                set_blender_data(getattr(item, key), config[key])
            else:
                setattr(item, key, config[key])


def _recursive_rename(name: str, count: int = 0) -> str:
    """
    a hacked-together helper function to call  when importing STL files to rename all existing copies and
    return a name that is safe to use. This will be unnecessary if we can figure out how to get the handle
    for the STL directly on import...

    :param name: expected name of the STL
    :param count: internal counter for this function.
    :return: a safe name to rename the imported STL.
    """
    if count == 0:
        desired_name = f'{name}_{count:03d}'
    else:
        desired_name = name
    existing = bpy.data.objects.get(desired_name)
    if existing is not None:
        _recursive_rename(f'{desired_name[:-4]}_{count:03d}', count+1)
    elif count > 0:
        previous_existing = bpy.data.objects.get(f'{desired_name[:-4]}_{count-2:03d}')
        previous_existing.name = f'{desired_name[:-4]}_{count-1:03d}'
    return desired_name


def import_stl(stl_file: str,
               center: bool = False,
               smooth_shading: bool = True,
               collection: bpy.types.Collection = None) -> bpy.types.Object:
    """
    import an STL into blender

    :param stl_file: path to the STL file
    :param center: whether to move the stl's center of volume to the origin
    :param smooth_shading: sets the model to be displayed as smooth
    :param collection: a collection to put the object in
    :return: the blender Object for the STL mesh object
    """
    object_name = os.path.splitext(os.path.basename(stl_file))[0]
    new_obj_name = _recursive_rename(object_name)
    bpy.ops.import_mesh.stl(filepath=stl_file)
    obj = bpy.data.objects[object_name]
    obj.name = f'{new_obj_name}'

    if collection is not None:
        bpy.context.collection.objects.unlink(obj)
        collection.objects.link(obj)

    if center:
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        obj.location = [0, 0, 0]
        apply_transformations(obj, location=True, rotation=False, scale=False)

    if smooth_shading:
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def scale_mesh_volume(obj: bpy.types.Object, volume: float) -> None:
    """
    scale a blender object by volume to a given volume

    :param volume: the desired volume.
    :param obj: the blender object to scale
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = bm.calc_volume()
    scaling_factor = (volume / vol) ** (1 / 3)
    obj.scale = Vector([scaling_factor] * 3)
    apply_transformations(obj, location=False, rotation=False, scale=True)


def apply_transformations(obj: bpy.types.Object,
                          location=True,
                          rotation=True,
                          scale=True) -> None:
    """
    Set current transformations as permanent for the given object. maintains the object's current physical position and
    resets transformations to default. (i.e. location is now the global origin)
    This affects all children of the object as well.
    https://blender.stackexchange.com/questions/159538/how-to-apply-all-transformations-to-an-object-at-low-level

    :param obj: The object to set
    :param location: reset the location
    :param rotation: reset the rotation
    :param scale: reset the scale
    """
    # obj.data.transform(obj.matrix_world)
    # obj.data.update()
    # matrix = Matrix.Identity(4)
    # obj.matrix_world = matrix
    mb = obj.matrix_basis
    I = Matrix()
    loc, rot, _scale = mb.decompose()

    # rotation
    T = Matrix.Translation(loc)
    # R = rot.to_matrix().to_4x4()
    R = mb.to_3x3().normalized().to_4x4()
    S = Matrix.Diagonal(_scale).to_4x4()

    transform = [I, I, I]
    basis = [T, R, S]

    def swap(i):
        transform[i], basis[i] = basis[i], transform[i]

    if location:
        swap(0)
    if rotation:
        swap(1)
    if scale:
        swap(2)

    M = transform[0] @ transform[1] @ transform[2]
    if hasattr(obj.data, "transform"):
        obj.data.transform(M)
    for c in obj.children:  # TODO: make this recursive
        c.matrix_local = M @ c.matrix_local

    obj.matrix_basis = basis[0] @ basis[1] @ basis[2]


def apply_surface_displacement():
    pass


def new_material(name: str) -> bpy.types.Material:
    """ A helper function for creating new materials in Blender.

    :param name: the name for the new material.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat


def add_surface_lighting(stl_file: str,
                         collection: bpy.types.Collection = None,
                         parent_object: bpy.types.Object = None,
                         scaling_factor: float = 1,
                         euler_rotation: List[float] = None,
                         emission_color: List[float] = None,
                         emission_strength: float = 100) \
        -> Tuple[bpy.types.Object, bpy.types.Node]:
    """
    Add an STL surface to the collection and make it light up.
    TODO: add ability to specify material settings

    :param stl_file: path to stl surface
    :param parent_object: optional parent object to assign this light to
    :param collection: the collection to link this object to. defaults to active one
    :param scaling_factor: uniform scaling factor to apply to the STL model
    :param euler_rotation: initial rotation to apply to STL model in degrees
    :param emission_strength: high bright the light is
    :param emission_color: the color of the light. defaults to [1, 1, 1, 1]
    :return: the added light object, the emission shader node
    """
    if euler_rotation is None:
        euler_rotation = [0, 0, 0]
    if emission_color is None:
        emission_color = [1, 1, 1, 1]
    bpy.ops.import_mesh.stl(filepath=stl_file)
    stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_file))[0]]
    stl_object.scale = Vector([scaling_factor, scaling_factor, scaling_factor])
    stl_object.rotation_euler = Vector(np.radians(euler_rotation))
    if collection is not None:
        bpy.context.scene.collection.objects.unlink(stl_object)
        collection.objects.link(stl_object)

    if parent_object:
        stl_object.parent = parent_object
    mat = new_material('light_emission')
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeEmission')
    mixer = nodes.new(type='ShaderNodeMixShader')
    backfacing = nodes.new(type='ShaderNodeNewGeometry')
    nodes["Emission"].inputs[0].default_value = emission_color
    nodes["Emission"].inputs[1].default_value = emission_strength
    links.new(backfacing.outputs[6], mixer.inputs[0])
    links.new(shader.outputs[0], mixer.inputs[1])
    links.new(mixer.outputs[0], output.inputs[0])
    stl_object.data.materials.append(mat)
    return stl_object, shader


def add_tumor_particle_nodegroup(stl_file: str,
                                 amount: float = 10,
                                 volume_max: float = 0.1,
                                 scaling_range: List[float] = None,
                                 rotation_range: List[float] = None) \
        -> bpy.types.NodeGroup:
    """
    Creates node group that scatters instances of the mesh in the stl-file over the targeted object.

    :param stl_file: path to the stl file to be used as tumor particle
    :param amount: controls amount of particles added
    :param volume_max: volume of object referenced for the instances
    :param scaling_range: range in which scaling of instances varies
    :param rotation_range: range in which rotation of instances varies
    :return: particle scattering node group
    """
    if scaling_range is None:
        scaling_range = [0.1, 1]
    if rotation_range is None:
        rotation_range = [0, 360]
    # create reference object from .stl-file
    particle_ref_object = import_stl(str(stl_file), center=True)
    scale_mesh_volume(particle_ref_object, volume_max)
    # apply transforms
    apply_transformations(particle_ref_object)
    particle_ref_object.hide_render = True
    # set up node group
    particle_nodegroup = bpy.data.node_groups.new('particle-nodes', type='GeometryNodeTree')
    nodes = particle_nodegroup.nodes
    links = particle_nodegroup.links
    # create nodes
    group_in = nodes.new('NodeGroupInput')
    points_on_faces = nodes.new('GeometryNodeDistributePointsOnFaces')
    face_area = nodes.new('GeometryNodeInputMeshFaceArea')
    attribute_statistic = nodes.new('GeometryNodeAttributeStatistic')
    div = nodes.new('ShaderNodeMath')
    obj_info = nodes.new('GeometryNodeObjectInfo')
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    rotation_vector = nodes.new('FunctionNodeRandomValue')
    scaling_int = nodes.new('FunctionNodeRandomValue')
    join_geo = nodes.new('GeometryNodeJoinGeometry')
    group_out = nodes.new('NodeGroupOutput')
    # set default parameters
    rotation_vector.data_type = 'FLOAT_VECTOR'
    div.operation = 'DIVIDE'
    div.inputs[0].default_value = amount
    rotation_vector.inputs.data.inputs['Min'].default_value = [np.deg2rad(rotation_range[0])]*3
    rotation_vector.inputs.data.inputs['Max'].default_value = [np.deg2rad(rotation_range[1])]*3
    scaling_int.inputs.data.inputs[2].default_value = scaling_range[0]
    scaling_int.inputs.data.inputs[3].default_value = scaling_range[1]
    obj_info.inputs['Object'].default_value = particle_ref_object
    # link nodes
    links.new(group_in.outputs[0], points_on_faces.inputs['Mesh'])  # 0->'Geometry'
    links.new(group_in.outputs[0], attribute_statistic.inputs['Geometry'])
    links.new(face_area.outputs['Area'], attribute_statistic.inputs['Attribute'])
    links.new(attribute_statistic.outputs['Sum'], div.inputs[1]) # 1 -> Denominator
    links.new(div.outputs['Value'], points_on_faces.inputs['Density'])
    links.new(group_in.outputs[0], join_geo.inputs['Geometry'])
    links.new(obj_info.outputs['Geometry'], instance_on_points.inputs['Instance'])
    links.new(points_on_faces.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(scaling_int.outputs[1], instance_on_points.inputs['Scale'])  # 1 ->'Value' single value random float
    # node has no output 0
    links.new(rotation_vector.outputs['Value'], instance_on_points.inputs['Rotation'])
    links.new(instance_on_points.outputs['Instances'], join_geo.inputs['Geometry'])
    links.new(join_geo.outputs['Geometry'], group_out.inputs[0])
    return particle_nodegroup


def add_diverticulum_nodegroup(  amount: float = 2,
                                 subdivisions_sphere: int = 4,
                                 radius_sphere_range: List[float] = None,
                                 radius_opening_range: List[float] = None) \
        -> bpy.types.NodeGroup:
    """
    Creates node group that scatters instances of the mesh in the stl-file over the targeted object.

    :param amount: controls amount of particles added
    :param subdivisions_sphere: mesh detail of the sphere
    :param radius_sphere_range: range in which the radii of the sphere instances vary in m
    :param radius_opening_range: rough control of the size of the opening, only values in a certain range make sense
    :return: diverticulum node group
    """
    if radius_sphere_range is None:
         radius_sphere_range = [0.001, 0.020]
    if radius_opening_range is None:
        radius_opening_range = [0.003, 0.008]
    # create reference object from .stl-file

    # set up node group
    diverticulum_nodegroup = bpy.data.node_groups.new('diverticulum-nodes', type='GeometryNodeTree')
    nodes = diverticulum_nodegroup.nodes
    links = diverticulum_nodegroup.links

    # MAIN NODE CHAIN: scatter spheres (of constant radius) across target object and unite target mesh and
    # spheres with a boolean union
    group_in = nodes.new('NodeGroupInput')
    # scatter points over mesh surface
    points_on_faces = nodes.new('GeometryNodeDistributePointsOnFaces')
    links.new(group_in.outputs[0], points_on_faces.inputs['Mesh'])  # 0->'Geometry'
    # create reference sphere
    ico_sphere = nodes.new('GeometryNodeMeshIcoSphere')
    ico_sphere.inputs['Radius'].default_value = 1
    ico_sphere.inputs['Subdivisions'].default_value = subdivisions_sphere
    # link sphere to points
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    links.new(points_on_faces.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(ico_sphere.outputs['Mesh'], instance_on_points.inputs['Instance'])
    # currently the spheres' center is located on the surface-points. This is changed down the line using
    # this node.
    translate_instances = nodes.new('GeometryNodeTranslateInstances')
    translate_instances.inputs['Local Space'].default_value = False
    links.new(instance_on_points.outputs['Instances'], translate_instances.inputs['Instances'])
    # delete regions where the spheres' mesh reach into target's mesh and vice versa
    mesh_boolean = nodes.new('GeometryNodeMeshBoolean')
    mesh_boolean.operation = 'UNION'
    links.new(group_in.outputs[0], mesh_boolean.inputs[1])
    links.new(translate_instances.outputs['Instances'], mesh_boolean.inputs[1])
    # output
    group_out = nodes.new('NodeGroupOutput')
    links.new(mesh_boolean.outputs['Mesh'], group_out.inputs[0])

    # SUBGROUPS: additional groups of nodes which modify the behavior of the main node chain
    # GROUP 1: make amount of instances invariant to the surface area of the target object by calculating the surface
    # area of the target and feeding the inverse into the density of the randomly distributed points
    face_area = nodes.new('GeometryNodeInputMeshFaceArea')
    attribute_statistic = nodes.new('GeometryNodeAttributeStatistic')
    div_by_surface_area = nodes.new('ShaderNodeMath')
    div_by_surface_area.operation = 'DIVIDE'
    div_by_surface_area.inputs[0].default_value = amount
    links.new(group_in.outputs[0], attribute_statistic.inputs['Geometry'])
    links.new(face_area.outputs['Area'], attribute_statistic.inputs['Attribute'])
    links.new(attribute_statistic.outputs['Sum'], div_by_surface_area.inputs[1])  # 1 -> Denominator
    links.new(div_by_surface_area.outputs['Value'], points_on_faces.inputs['Density'])
    # GROUP 2: Randomize the radius of the spheres
    random_radius_sphere = nodes.new('FunctionNodeRandomValue')
    random_radius_sphere.inputs.data.inputs[2].default_value = radius_sphere_range[0]
    random_radius_sphere.inputs.data.inputs[3].default_value = radius_sphere_range[1]
    links.new(random_radius_sphere.outputs[1], instance_on_points.inputs['Scale'])
    # GROUP 3: Translate the spheres along the normal of the target's surface, away from the surface in order to mimic
    # the placement of a diverticulum. The size of the opening is controlled by how far the sphere is translated
    # away from the surface. The value for the radius of the created opening/window is only an estimate based on the
    # assumption that the target's surface is flat.

    # determine the direction of translation (normalize normal of target surface)
    normalize = nodes.new('ShaderNodeVectorMath')
    normalize.operation = 'NORMALIZE'
    links.new(points_on_faces.outputs['Normal'], normalize.inputs['Vector'])

    # randomize the radius of the opening
    random_radius_opening = nodes.new('FunctionNodeRandomValue')
    random_radius_opening.inputs.data.inputs[2].default_value = radius_opening_range[0]
    random_radius_opening.inputs.data.inputs[3].default_value = radius_opening_range[1]

    # determine the distance d of translation (some Trigonometry)
    # arcsin(r2) (r2 = radius of the opening)
    arcsine = nodes.new('ShaderNodeMath')
    arcsine.operation = 'ARCSINE'
    links.new(random_radius_opening.outputs[1], arcsine.inputs[0])

    # arcsin(r2)/r3 (r3 = radius of the sphere
    div_by_radius_sphere = nodes.new('ShaderNodeMath')
    div_by_radius_sphere.operation = 'DIVIDE'
    div_by_radius_sphere.use_clamp = True  # get rid of the periodicity, due to this only certain ranges for
    # radius_opening actually change something
    links.new(arcsine.outputs['Value'], div_by_radius_sphere.inputs[0])  # 0->Numerator
    links.new(random_radius_sphere.outputs[1], div_by_radius_sphere.inputs[1])

    # cos(arcsin(r2)/r3)
    cosine = nodes.new('ShaderNodeMath')
    cosine.operation = 'COSINE'
    links.new(div_by_radius_sphere.outputs['Value'], cosine.inputs[0])

    # d = cos(arcsin(r2)/r3) * r3
    mult_by_radius_sphere = nodes.new('ShaderNodeMath')
    mult_by_radius_sphere.operation = 'MULTIPLY'
    links.new(random_radius_sphere.outputs[1], mult_by_radius_sphere.inputs[0])
    links.new(cosine.outputs['Value'], mult_by_radius_sphere.inputs[1])

    # multiply translation direction and distance to receive final translation Vector
    mult_direction_and_distance = nodes.new('ShaderNodeVectorMath')
    mult_direction_and_distance.operation = 'MULTIPLY'
    links.new(normalize.outputs['Vector'], mult_direction_and_distance.inputs[0])
    links.new(mult_by_radius_sphere.outputs['Value'], mult_direction_and_distance.inputs[1])

    # apply translation via Translate Instances node from earlier
    links.new(mult_direction_and_distance.outputs['Vector'], translate_instances.inputs['Translation'])

    return diverticulum_nodegroup


def add_render_output_nodes(scene: bpy.types.Scene,
                            color: bool = True,
                            depth: bool = True,
                            normals: bool = False,
                            view_layer: str = "ViewLayer") -> List[bpy.types.Node]:
    """
    Modify the graph of a scene's node tree to include color and depth outputs

    :param color: whether to include color as output.
    :param depth: whether to include the depth as output.
    :param normals: whether to include the normals as output.
    :param scene: the scene to create the rendering for.
    :param view_layer: the view layer in the scene to enable the rendering passes for.
    :returns: image node, depth node, normals node  <- will be None if option not enabled.
    """
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    return_list = [None, None, None]
    links = tree.links
    if depth:
        scene.view_layers[view_layer].use_pass_z = True
        # create depth output node
        depth_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_node.format.file_format = "OPEN_EXR"
        links.new(rl.outputs['Depth'], depth_node.inputs['Image'])  # link Z to output
        return_list[1] = depth_node

    if color:
        # create image output node
        img_node = tree.nodes.new('CompositorNodeOutputFile')
        img_node.format.file_format = "PNG"
        links.new(rl.outputs['Image'], img_node.inputs['Image'])  # link image to output
        return_list[0] = img_node

    if normals:
        scene.view_layers[view_layer].use_pass_normal = True
        # create normals output node
        normal_node = tree.nodes.new('CompositorNodeOutputFile')
        normal_node.format.file_format = "OPEN_EXR"
        links.new(rl.outputs['Normal'], normal_node.inputs['Image'])  # link Z to output
        return_list[2] = normal_node

    return return_list


def add_shrinkwrap_constraint(obj: bpy.types.Object,
                              config: bconfig.ShrinkwrapConfig = bconfig.ShrinkwrapConfig()) -> bpy.types.Constraint:
    """
    adds a shrinkwrap constraint to the provided object and sets the values provided in config.

    :param obj: a blender object
    :param config: the configuration for the shrinkwrap
    :return: the constraint handle
    """
    shrinkwrap_constr = obj.constraints.new("SHRINKWRAP")
    set_blender_data(shrinkwrap_constr, config)
    return shrinkwrap_constr


def set_gpu_rendering_preferences(gpu: int = -1) -> None:
    """
    Set GPU resources to use for rendering. This function only works for CUDA GPUs or Macs

    :param gpu: the GPU ID to use. if -1, uses all GPUs available.
    """
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA' if (sys.platform != 'darwin') else 'METAL'
    for dev in prefs.devices:
        if dev.type == "CPU":
            dev.use = True
    gpu_num = 0
    for dev in prefs.devices:
        if dev.type != "CPU":
            if gpu == -1:
                dev.use = True
            else:
                dev.use = gpu_num == gpu
            gpu_num += 1


def extract_system_arguments() -> Tuple[List[str], bool]:
    """
    Provides the user passed arguments that come after "--" when running a python script through blender.
    Also provides whether the script was called as headless

    :return: parsed_args, headless
    """
    idx = 0
    try:
        idx = sys.argv.index("--")
        cli_arguments = True
    except ValueError:
        cli_arguments = False
    arg_string = sys.argv[idx + 1:] if cli_arguments else ""

    gui_enabled = False
    try:
        gui_enabled = bool(sys.argv.index('-b'))
    except ValueError:
        pass

    return arg_string, not gui_enabled


def convert_norm_exr_2_cam(file: str, camera: bpy.types.Camera, output_file: str = None) -> None:
    """
    Load an exr file that represents global normals for surfaces and convert them to the camera's coordinates.
    If output_file is unspecified, then it will overwrite the existing exr file.

    :param file: normals file to convert
    :param camera: blender camera to convert normals to
    :param output_file: optional output file for the converted exr file
    """
    cam_world_matrix: Matrix = camera.matrix_world.copy()
    cam_world_matrix.invert()
    cam_world_numpy = np.asarray(cam_world_matrix.to_3x3())
    normals = cv2.imread(file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    shape = normals.shape
    transformed_norms = np.reshape(cam_world_numpy @ np.expand_dims(np.reshape(normals, (int(shape[0] * shape[1]), 3)), -1), shape)
    filename = file if output_file is None else output_file
    cv2.imwrite(filename, transformed_norms.astype(np.float32))


def clear_all_keyframes() -> None:
    """
    Erases all keyframes from all objects
    """
    for obj in bpy.data.objects:
        if obj.animation_data:  # Check for presence of animation data.
            obj.animation_data.action = None


def add_smoothing_modifier(obj: bpy.types.Object, config: bconfig.SmoothModConfig) -> None:
    modifier = obj.modifiers.new(type='SMOOTH', name='smoothing')
    set_blender_data(modifier, config)