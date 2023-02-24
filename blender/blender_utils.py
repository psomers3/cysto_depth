import numpy as np
import os
import sys
import bpy
import config.blender_config as bconfig
from typing import *
import bmesh
from mathutils import Matrix, Vector


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


def init_blender(configuration: bconfig.BlenderConfig) -> Tuple[bpy.types.Scene, bpy.types.ViewLayer]:
    """ Initialize blender scene by deleting demo objects and applying settings from configuration

    :param configuration: A set of blender configurations to apply to the current scene
    :returns: the current active scene and view_layer
    """
    context = bpy.context
    scene = context.scene
    view_layer = context.view_layer
    for c in scene.collection.children:
        bpy.data.collections.remove(c, do_unlink=True)
    set_blender_data(scene, configuration)
    return scene, view_layer


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
        _recursive_rename(f'{desired_name[:-4]}_{count:03d}', count + 1)
    elif count > 0:
        previous_existing = bpy.data.objects.get(f'{desired_name[:-4]}_{count - 2:03d}')
        previous_existing.name = f'{desired_name[:-4]}_{count - 1:03d}'
    return desired_name


def import_stl(stl_file: str,
               center: bool = False,
               smooth_shading: bool = True,
               collection: bpy.types.Collection = None,
               flip_normals: bool = False) -> bpy.types.Object:
    """
    import an STL into blender

    :param stl_file: path to the STL file
    :param center: whether to move the stl's center of volume to the origin
    :param smooth_shading: sets the model to be displayed as smooth
    :param collection: a collection to put the object in
    :param flip_normals: flip all the normals
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

    if flip_normals:
        flip_mesh_normals(obj)
    obj.select_set(False)
    return obj


def flip_mesh_normals(obj: bpy.types.Object) -> None:
    """
    Flip all the normals on an object's mesh
    :param obj: blender object that is a mesh
    :return:
    """
    bm = bmesh.new()
    me = obj.data
    bm.from_mesh(me)
    for f in bm.faces:
        f.normal_flip()
    bm.normal_update()  # not sure if req'd
    bm.to_mesh(me)
    me.update()


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
                          scale=True) -> Matrix:
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
    M, basis = get_transformation(obj, location, rotation, scale)
    if hasattr(obj.data, "transform"):
        obj.data.transform(M)
    for c in obj.children:  # TODO: make this recursive
        c.matrix_local = M @ c.matrix_local

    obj.matrix_basis = basis[0] @ basis[1] @ basis[2]

    return M


def get_transformation(obj: bpy.types.Object, location=True, rotation=True, scale=True) -> Tuple[Matrix, List[Matrix]]:
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

    return M, basis


def new_material(name: str) -> bpy.types.Material:
    """ A helper function for creating new materials in Blender.

    :param name: the name for the new material.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if mat.node_tree is not None:
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
    apply_transformations(stl_object)
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
                                 rotation_mode: str = 'random',
                                 rotation_range: List[float] = None,
                                 collection: bpy.types.Collection = None) \
        -> Tuple[bpy.types.NodeGroup, bpy.types.Object]:
    """
    Creates node group that scatters instances of the mesh in the stl-file over the targeted object.
    :param stl_file: path to the stl file to be used as tumor particle
    :param amount: controls amount of particles added
    :param volume_max: volume of object referenced for the instances
    :param scaling_range: range in which scaling of instances varies
    :param rotation_mode: determines how the instances are rotated, default: 'random'
                            -'random': random rotation, range given by rotation_range
                            -'align_to_surface': the z-axis of the particle model is aligned to the normal of the surface
    :param rotation_range: range in which rotation of instances varies, only has effect if
    :return: particle scattering node group
    """
    if scaling_range is None:
        scaling_range = [0.1, 1]
    if rotation_range is None:
        rotation_range = [0, 360]
    # create reference object from .stl-file
    particle_ref_object = import_stl(str(stl_file), center=True, flip_normals=True)
    scale_mesh_volume(particle_ref_object, volume_max)
    # apply transforms
    apply_transformations(particle_ref_object)
    particle_ref_object.hide_render = True
    particle_ref_object.hide_viewport = True
    if collection is not None:
        collection.objects.link(particle_ref_object)
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
    mirror_z_axis = nodes.new('FunctionNodeRotateEuler')
    rotation_vector = nodes.new('FunctionNodeRandomValue')
    scaling_int = nodes.new('FunctionNodeRandomValue')
    join_geo = nodes.new('GeometryNodeJoinGeometry')
    group_out = nodes.new('NodeGroupOutput')
    # set default parameters
    rotation_vector.data_type = 'FLOAT_VECTOR'
    div.operation = 'DIVIDE'
    div.inputs[0].default_value = amount
    mirror_z_axis.type = 'EULER'
    mirror_z_axis.space = 'LOCAL'
    mirror_z_axis.inputs.data.inputs[1].default_value = (3.141593, 0, 0)
    rotation_vector.inputs.data.inputs['Min'].default_value = [np.deg2rad(rotation_range[0])] * 3
    rotation_vector.inputs.data.inputs['Max'].default_value = [np.deg2rad(rotation_range[1])] * 3
    scaling_int.inputs.data.inputs[2].default_value = scaling_range[0]
    scaling_int.inputs.data.inputs[3].default_value = scaling_range[1]
    obj_info.inputs['Object'].default_value = particle_ref_object
    # link nodes
    links.new(group_in.outputs[0], points_on_faces.inputs['Mesh'])  # 0->'Geometry'
    links.new(group_in.outputs[0], attribute_statistic.inputs['Geometry'])
    links.new(face_area.outputs['Area'], attribute_statistic.inputs['Attribute'])
    links.new(attribute_statistic.outputs['Sum'], div.inputs[1])  # 1 -> Denominator
    links.new(div.outputs['Value'], points_on_faces.inputs['Density'])
    links.new(group_in.outputs[0], join_geo.inputs['Geometry'])
    links.new(obj_info.outputs['Geometry'], instance_on_points.inputs['Instance'])
    links.new(points_on_faces.outputs['Points'], instance_on_points.inputs['Points'])
    links.new(scaling_int.outputs[1], instance_on_points.inputs['Scale'])  # 1 ->'Value' single value random float
    # node has no output 0
    if rotation_mode == 'align_to_surface':
        links.new(points_on_faces.outputs['Rotation'], mirror_z_axis.inputs['Rotation'])
        links.new(mirror_z_axis.outputs['Rotation'], instance_on_points.inputs['Rotation'])
    else:
        links.new(rotation_vector.outputs['Value'], instance_on_points.inputs['Rotation'])

    links.new(instance_on_points.outputs['Instances'], join_geo.inputs['Geometry'])
    links.new(join_geo.outputs['Geometry'], group_out.inputs[0])
    return particle_nodegroup, particle_ref_object


def add_diverticulum_nodegroup(amount: float = 2,
                               subdivisions_sphere: int = 4,
                               radius_sphere_range: List[float] = None,
                               translation_range: List[float] = None) \
        -> bpy.types.NodeGroup:
    """
    Creates node group that introduces diverticuli, by scattering instances of a sphere on the surface of the targeted
    object. The spheres are then translated by a random distance along the target surface's normal. Spheres and target
    mesh are combined via boolean union.
    :param amount: controls amount of particles added
    :param subdivisions_sphere: mesh detail of the sphere
    :param radius_sphere_range: range in which the radii of the sphere instances vary in m
    :param translation_range: range for random translation of the spheres in 'radius of translated sphere'. Values larger than 0 -> outward
     translation, values <0 -> inward translation. Default range = [-0.7, 0.7]
    :return: diverticulum node group
    """
    if radius_sphere_range is None:
        radius_sphere_range = [0.001, 0.020]
    if translation_range is None:
        radius_opening_range = [-0.7, 0.7]
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
    # turn instances into own meshes
    realize_instances = nodes.new('GeometryNodeRealizeInstances')
    links.new(mesh_boolean.outputs['Mesh'], realize_instances.inputs[0])
    # output
    group_out = nodes.new('NodeGroupOutput')
    links.new(realize_instances.outputs['Geometry'], group_out.inputs[0])

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

    # randomize the translation factor for the spheres
    random_translation_factor = nodes.new('FunctionNodeRandomValue')
    random_translation_factor.inputs.data.inputs[2].default_value = translation_range[0]
    random_translation_factor.inputs.data.inputs[3].default_value = translation_range[1]

    # calculate translation distance for spheres
    mult_by_radius_sphere = nodes.new('ShaderNodeMath')
    mult_by_radius_sphere.operation = 'MULTIPLY'
    links.new(random_radius_sphere.outputs[1], mult_by_radius_sphere.inputs[0])
    links.new(random_translation_factor.outputs[1], mult_by_radius_sphere.inputs[1])

    # multiply translation direction and distance to receive final translation vector
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
                            custom_normals_label: str = "",
                            custom_depth_label: str = "",
                            view_layer: str = "ViewLayer") -> List[bpy.types.Node]:
    """
    Modify the graph of a scene's node tree to include color and depth outputs

    :param color: whether to include color as output.
    :param depth: whether to include the depth as output.
    :param normals: whether to include the normals as output.
    :param custom_normals_label: a user-defined AOV output to use for the normals. If empty, defaults to the usual
                                 normals pass.
    :param custom_depth_label: a user-defined AOV output to use for the normals. If empty, defaults to the usual
                                 normals pass.
    :param scene: the scene to create the rendering for.
    :param view_layer: the view layer in the scene to enable the rendering passes for.
    :returns: image node, depth node, normals node  <- will be None if option not enabled.
    """
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.scene = scene
    return_list = [None, None, None]
    links = tree.links
    if depth:
        scene.view_layers[view_layer].use_pass_z = True
        # create depth output node
        depth_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_node.format.file_format = "OPEN_EXR"
        depth_label = 'Depth'
        if custom_depth_label:
            depth_label = custom_depth_label
            aov = scene.view_layers[view_layer].aovs.add()
            aov.name = 'raw_depth'
        links.new(rl.outputs[depth_label], depth_node.inputs['Image'])
        return_list[1] = depth_node
        depth_node.mute = True
        depth_node.name = 'depth_out'

    if color:
        # create image output node
        img_node = tree.nodes.new('CompositorNodeOutputFile')
        img_node.format.file_format = "PNG"
        links.new(rl.outputs['Image'], img_node.inputs['Image'])  # link image to output
        return_list[0] = img_node
        img_node.name = 'color_out'

    if normals:
        scene.view_layers[view_layer].use_pass_normal = True
        # create normals output node
        normal_node = tree.nodes.new('CompositorNodeOutputFile')
        normal_node.format.file_format = "OPEN_EXR"
        normals_label = "Normal"
        if custom_normals_label:
            normals_label = custom_normals_label
            aov = scene.view_layers[view_layer].aovs.add()
            aov.name = 'raw_normals'

        links.new(rl.outputs[normals_label], normal_node.inputs['Image'])
        return_list[2] = normal_node
        normal_node.mute = True
        normal_node.name = 'normals_out'

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


def set_gpu_rendering_preferences(gpu: int = -1, verbose: bool = True, device_type: str = 'OPTIX') -> None:
    """
    Set GPU resources to use for rendering. This function only works for CUDA GPUs or Macs

    :param gpu: the GPU ID to use. if -1, uses all GPUs available.
    :param verbose: whether to print the devices found.
    :param device_type: which device type for cycles rendering. METAL for apple silicon and either OPTIX, CUDA, OPENCL 
                        for other systems.
    """
    gpu_types = ['OPTIX', 'CUDA', 'METAL', 'OPENCL']
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = device_type
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = True
    gpu_num = 0
    for dev in prefs.devices:
        if dev.type in gpu_types:
            if gpu == -1:
                dev.use = True
            else:
                if dev.type == device_type:
                    dev.use = gpu_num == gpu
                    gpu_num += 1
                else:
                    dev.use = False
        if verbose:
            print(f'name: {dev.name}, type: {dev.type}, use: {dev.use}')


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


def clear_all_keyframes() -> None:
    """
    Erases all keyframes from all objects
    """
    for obj in bpy.data.objects:
        if obj.animation_data:  # Check for presence of animation data.
            obj.animation_data.action = None


def add_subdivision_modifier(obj: bpy.types.Object, config: bconfig.SubdivisionModConfig) -> None:
    """
    Add a surface subdivision modifier to the object.

    :param obj: the blender object to apply the subdivision to
    :param config: configuration of properties for the modifier
    """
    modifier = obj.modifiers.new(type='SUBSURF', name='smoothing')
    set_blender_data(modifier, config)


def add_resection_loop(config: bconfig.ResectionLoopConfig,
                       collection: bpy.types.Collection = None,
                       parent: bpy.types.Object = None) \
        -> Tuple[bpy.types.Object, bpy.types.Object, bpy.types.Object, bpy.types.Object, List[bpy.types.Object]]:
    """
    Adds a cutting loop to the endoscope.
    :param collection: collection to add the objects to.
    :param config: the configuration settings for adding the loop
    :param parent: an optional blender object to set as the parent (i.e. the camera)
    :return: tuple with objects containing the resection loop items, wire, insulation, extension direction,
             and no-clip-points.
    """

    wire = import_stl(config.wire_stl, flip_normals=True)
    insulation = import_stl(config.insulation_stl, flip_normals=True)
    wire.scale = Vector([config.scaling_factor] * 3)
    insulation.scale = Vector([config.scaling_factor] * 3)
    wire.rotation_euler = Vector(np.radians(config.euler_rotation))
    insulation.rotation_euler = Vector(np.radians(config.euler_rotation))
    transform = apply_transformations(wire)
    apply_transformations(insulation)
    # retrieve, format and transform direction and no-clip-points
    direction = np.array(config.extension_direction)
    direction = direction.transpose()
    direction = np.append(direction, [1], axis=0)
    loop_extension_direction = np.array(transform) @ direction
    points = np.array(config.no_clip_points)
    points = points.transpose()
    points = np.append(points, [[1] * points.shape[1]], axis=0)
    loop_no_clip_points = np.array(transform) @ points
    resection_loop = bpy.data.objects.new('resection_loop', None)
    wire.parent = resection_loop
    insulation.parent = resection_loop
    wire_material = new_material('reflective_metal')
    nodes = wire_material.node_tree.nodes
    links = wire_material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.inputs[0].default_value = config.wire_base_color  # Base Color
    shader.inputs[6].default_value = config.wire_metallic  # Metallic
    shader.inputs[9].default_value = config.wire_roughness  # Roughness
    shader.inputs[10].default_value = config.wire_anisotropic  # Anisotropic
    links.new(shader.outputs[0], output.inputs[0])
    wire.data.materials.append(wire_material)

    if parent:
        resection_loop.parent = parent
    if collection is not None:
        bpy.context.scene.collection.objects.unlink(wire)
        bpy.context.scene.collection.objects.unlink(insulation)
        collection.objects.link(wire)
        collection.objects.link(insulation)
        collection.objects.link(resection_loop)
    return resection_loop, wire, insulation, loop_extension_direction, loop_no_clip_points


def add_raw_depth_to_material(mat:bpy.types.Material) -> None:
    """
        Add an AOV output to the given material that will forward the raw z-depth in camera space to the rendering
        compositor. The AOV output is called "raw_depth"
        :param mat: blender material to foward normals for
        """
    cam = mat.node_tree.nodes.new("ShaderNodeCameraData")
    aov = mat.node_tree.nodes.new("ShaderNodeOutputAOV")
    aov.name = "raw_depth"
    mat.node_tree.links.new(cam.outputs['View Z Depth'], aov.inputs['Color'])


def add_depth_to_all_materials() -> None:
    """
    Add normals AOV to every registered material
    """
    for material in bpy.data.materials:
        if material is not None:
            material.use_nodes = True
            add_raw_depth_to_material(material)


def add_raw_normals_to_material(mat: bpy.types.Material) -> None:
    """
    Add an AOV output to the given material that will forward the raw normals in camera space to the rendering
    compositor. The AOV output is called "raw_normals"
    :param mat: blender material to foward normals for
    """
    geo = mat.node_tree.nodes.new("ShaderNodeNewGeometry")
    cam_transform = mat.node_tree.nodes.new("ShaderNodeVectorTransform")
    cam_transform.vector_type = 'NORMAL'
    cam_transform.convert_to = 'CAMERA'
    aov = mat.node_tree.nodes.new("ShaderNodeOutputAOV")
    aov.name = "raw_normals"
    mat.node_tree.links.new(geo.outputs['Normal'], cam_transform.inputs['Vector'])
    mat.node_tree.links.new(cam_transform.outputs['Vector'], aov.inputs['Color'])


def add_normals_to_all_materials() -> None:
    """
    Add normals AOV to every registered material
    """
    for material in bpy.data.materials:
        if material is not None:
            material.use_nodes = True
            add_raw_normals_to_material(material)


def is_inside(p, obj: bpy.types.Object, normals_reversed: bool = False):
    _, point, normal, face = obj.closest_point_on_mesh(p)
    p2 = point-p
    v = p2.dot(normal)
    if normals_reversed:
        return v < 0.0
    else:
        return v > 0.0


def check_image_in_body(cam: bpy.types.Camera, obj: bpy.types.Object, scene: bpy.types.Scene) -> bool:
    """
    Return True only if all 4 camera corners are within the body.
    :param cam:
    :param obj:
    :param scene:
    :return:
    """
    frame_local = cam.data.view_frame(scene=scene)
    frame_corners_global = [cam.matrix_world @ corner for corner in frame_local]
    for corner in frame_corners_global:
        if is_inside(corner, obj, True):
            continue
        else:
            return False
    return True


def update_bladder_material(config: bconfig.BladderMaterialConfig, material_name: str) -> None:
    """
    A function to update the values in the bladder material.
    :param config:
    :param material_name:
    """
    mat = bpy.data.materials[material_name]
    mat.node_tree.nodes['Volume Absorption'].inputs['Density'].default_value = config.volume_absorbtion_density
    mat.node_tree.nodes['Volume Scatter'].inputs['Density'].default_value = config.volume_scatter_density
    mat.node_tree.nodes['Volume Scatter'].inputs['Anisotropy'].default_value = config.volume_scatter_anisotropy
