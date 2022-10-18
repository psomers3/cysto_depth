from http.cookiejar import DefaultCookiePolicy
from tokenize import String
import bpy
import numpy as np
from bpy.props import IntProperty, FloatProperty
import sys
import os
import bmesh
import argparse
import sys
from mathutils import Matrix, Vector



obj_map = {}

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def init(frames_per_sim, randomize_view=True ):
    scene = init_scene()
    # bpy.ops.object.light_add(type='POINT')
    depth_node, img_node = init_render_settings(scene)
    init_animation(frames_per_sim, randomize_view)
    return depth_node, img_node
    
def init_render_settings(scene, engine = "CYCLES"):
    # Set render resolution
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.image_settings.color_mode = 'RGB'
    if engine == 'CYCLES':
        scene.render.engine = 'CYCLES'
        scene.cycles.adaptive_min_samples = 64
        scene.cycles.adaptive_max_samples = 128
        scene.cycles.denoiser = 'OPTIX'
    elif engine == 'EEVEE':
        scene.render.engine = 'BLENDER_EEVEE'

    # Set up rendering of depth map:
    bpy.context.scene.use_nodes = True
    
    tree = bpy.context.scene.node_tree
    clear_current_render_graph(tree)
    return create_render_graph(tree)

def init_animation(frames_per_sim, randomize_view):

    random_points = sample_spherical(frames_per_sim)
    max_angle = 90
    dirs = np.random.uniform(low = np.radians(-max_angle), high = np.radians(max_angle), size = (frames_per_sim,3))
    if 'args' in obj_map:
        args = obj_map['args']
        distances = np.random.uniform(low = args.distance_min, high = args.distance_max, size = frames_per_sim)
        light_energies = np.random.uniform(low = args.light_energy_min, high = args.light_energy_max, size = frames_per_sim)
        spotsizes = np.random.uniform(low = np.radians(args.spotsize_min), high = np.radians(args.spotsize_max), size = frames_per_sim)
    empty = obj_map["empty"]
    
    for idx, point in enumerate(random_points.T):
        empty.location = point
        empty.keyframe_insert(data_path="location", frame=idx+1)
        if randomize_view:
            empty.rotation_euler = dirs[idx,:] # set y orientation
            empty.keyframe_insert(data_path="rotation_euler", frame= idx+1)
            if 'args' in obj_map:
                if distances is not None:
                    obj_map['shrinkwrap'].distance = distances[idx]
                    obj_map['shrinkwrap'].keyframe_insert(data_path="distance", frame= idx+1)
                if light_energies is not None:
                    obj_map['light_data'].energy = light_energies[idx]
                    obj_map['light_data'].keyframe_insert(data_path = "energy",frame = idx+1)
                if spotsizes is not None:
                    obj_map['light_data'].spot_size = spotsizes[idx]
                    obj_map['light_data'].keyframe_insert(data_path = "spot_size",frame = idx+1)
        bpy.context.scene.frame_end = frames_per_sim
        
        
    # Follow Path from start to end
#    path_constraint.offset_factor = 0
#    path_constraint.keyframe_insert(data_path="offset_factor", frame=1)
#    path_constraint.offset_factor = 1
#    path_constraint.keyframe_insert(data_path="offset_factor", frame=frames_per_sim)
#    bpy.context.scene.frame_end = frames_per_sim
#    # Randomize Direction
#    if randomize_view:
#        dirs = np.random.uniform(low = np.radians(-180), high = np.radians(180), size = (frames_per_sim,3))
#        for idx, dir in enumerate(dirs):
#            camera_object.rotation_euler = dir # set y orientation
#            camera_object.keyframe_insert(data_path="rotation_euler", frame= idx+1)
    
    
def clear_current_render_graph(tree):
    # clear default/old nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
        
def create_render_graph(tree):
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')
       
    # create depth output node
    depth_node = tree.nodes.new('CompositorNodeOutputFile')  
    depth_node.format.file_format = "OPEN_EXR"
    
    # create image output node
    img_node = tree.nodes.new('CompositorNodeOutputFile')  
    img_node.format.file_format = "PNG"
    
    # Links
    links = tree.links
    links.new(rl.outputs['Depth'], depth_node.inputs['Image']) # link Z to output
    links.new(rl.outputs['Image'], img_node.inputs['Image']) # link image to output
    return depth_node, img_node

def init_scene():
    scene = bpy.data.scenes["Scene"]
    scene.unit_settings.length_unit = 'MILLIMETERS'
    
    master_collection = bpy.context.scene.collection
    # disable all collections that are still active
    for collection in master_collection.children: 
        collection.hide_render= True
    endo_collection = create_endoscope(scene)
    master_collection.children.link(endo_collection) # add the generated collection to the master collection
    return scene
    
def create_endoscope(scene, deleteExisting=True):
    # Delete existing endoscope
    if deleteExisting:
        for collection in bpy.context.scene.collection.children:
            if collection.name.lower().startswith("endoscope"):
                for obj in collection.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(collection)
    # create collection for endoscope   
    collection = bpy.data.collections.new("Endoscope")
    camera_data = bpy.data.cameras.new(name='Camera_Data')
    camera_data.clip_start = 0.001 # 10mm 
    camera_data.clip_end = 0.1 # 1m
    camera_data.sensor_width = 1 # 1mm
    camera_data.lens_unit= "FOV"
    camera_data.angle= np.radians(110) # 110 degrees field of view
    camera_object = bpy.data.objects.new('Camera', camera_data)
    obj_map["camera"] = camera_object
    camera_object.location=(0,0,0)
    camera_object.rotation_euler = (0,0,0)
    collection.objects.link(camera_object)
    scene.camera = camera_object
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="Light_Data", type='SPOT')
    light_data.energy = 0.003 # 1mW
    light_data.shadow_soft_size = 0.001 # set radius of Light Source (5mm)
    light_data.spot_blend = 0.5 # smoothness of spotlight edges
    light_data.spot_size = np.radians(120) #
    obj_map['light_data'] = light_data

    # create new object with our light datablock
    light_left = bpy.data.objects.new(name="light_left", object_data=light_data)
    
    # create new object with our light datablock
    light_right = bpy.data.objects.new(name="light_right", object_data=light_data)
    light_offset = 0.001
    light_left.location = (-light_offset,0,0)
    light_right.location = (light_offset,0,0)
    light_left.parent = camera_object
    light_right.parent = camera_object

    # link light object
    collection.objects.link(light_right)
    collection.objects.link(light_left)
    
    # make it active 
    # bpy.context.view_layer.objects.active = light_object
    
    path_tracker = bpy.data.objects.new( "empty", None )
    obj_map["empty"] = path_tracker
    collection.objects.link(path_tracker)
    camera_object.parent = path_tracker
    shrinkwrap_constr = path_tracker.constraints.new("SHRINKWRAP")
    obj_map["shrinkwrap"] = shrinkwrap_constr
    shrinkwrap_constr.wrap_mode  = "INSIDE"
    shrinkwrap_constr.distance = 0.02
    shrinkwrap_constr.shrinkwrap_type = "NEAREST_SURFACE"
    shrinkwrap_constr.use_track_normal = True
    shrinkwrap_constr.track_axis = "TRACK_NEGATIVE_Z"
    return collection

    
def render():
    # render
    bpy.ops.render.render(animation=True)

def getSimulationCollections():
    master_collection = bpy.context.scene.collection
    simulation_collections = {}
    for collection in master_collection.children:
        if collection.name.lower().startswith("sim-labels"):
            simulation_collections[collection.name] = collection
    return simulation_collections

def getMeshAndPathFromCollection(sim_collection):
    curve = None
    for object in sim_collection.objects:
        if object.type == 'CURVE': 
            curve = object
        elif object.type == 'MESH':
            mesh = object
    return mesh,curve

def setAnimationTarget(sim_collection):
    mesh, curve = getMeshAndPathFromCollection(sim_collection)
    if curve == None:
        curve = bpy.data.objects["DefaultPath"]
    # attach camera to curve
    # path_constraint.target = curve
    shrinkwrap_const = obj_map["shrinkwrap"]
    shrinkwrap_const.target= mesh


def renderAnimation(sim_collection, img_node, depth_node, screenshot_folder):
    depth_node.base_path = screenshot_folder + "/"+ sim_collection.name
    img_node.base_path = screenshot_folder + "/" + sim_collection.name
    sim_collection.hide_render= False
    render()
    sim_collection.hide_render= True     

def importMeshes():
    import_collection = bpy.data.collections["Import"]
    default_collection = bpy.data.collections["Default"]
    # deselect all objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    mesh_obs = import_collection.objects
    # move all objects to center
    bpy.ops.object.origin_set(
            {"object" : mesh_obs[0],
            "selected_objects" : mesh_obs,
            "selected_editable_objects" : mesh_obs,
            }
    )
    # set shade smooth for all
    bpy.ops.object.shade_smooth(
            {"object" : mesh_obs[0],
            "selected_objects" : mesh_obs,
            "selected_editable_objects" : mesh_obs,
            }
    )
    # scale all to 400ml
    for obj in mesh_obs:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        vol = bm.calc_volume()
        print("Current Volume: {}".format(vol))
        obj.select_set(True)
        
        bpy.ops.mesh.print3d_scale_to_volume(volume= 0.0004, volume_init = vol)
        # apply so that scale is at [1 1 1] again
        # location and rotation is necessary somehow
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)
    
    for obj in mesh_obs:
        print("UV Projecting {}".format(obj.name))
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT') # for all faces
        # bpy.ops.uv.smart_project(angle_limit=66, scale_to_bounds = True, correct_aspect = True)
        bpy.ops.uv.sphere_project(scale_to_bounds = True, correct_aspect = True)
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')  
        bpy.context.view_layer.objects.active = None
        
    for obj in mesh_obs:
        print("Creating Modifiers {}".format(obj.name))
        obj.active_material = bpy.data.materials["DefaultMaterial"]
        obj.modifiers.new("displace","DISPLACE")
        obj.modifiers[0].texture_coords = "UV"
        obj.modifiers[0].strength = 0.01
        obj.modifiers[0].texture = bpy.data.textures["Texture"]
        obj.modifiers[0].mid_level = 0.1
        obj.modifiers.new("subsurf","LAPLACIANSMOOTH")    
        obj.modifiers["subsurf"].iterations = 2     
        obj.modifiers.new(name='particles',type='PARTICLE_SYSTEM')
        obj.particle_systems[0].settings.type = "HAIR"
        obj.particle_systems[0].settings.count = 20
        obj.particle_systems[0].settings.render_type = "OBJECT"
        obj.particle_systems[0].settings.instance_object = bpy.data.objects["DefaultParticle"]
        obj.particle_systems[0].settings.size_random = 0.8
        
    for obj in mesh_obs:
        print("Saving to collection {}".format(obj.name))
        target_collection_name = "sim-"+obj.name
        try: 
            target_collection = bpy.data.collections[target_collection_name]
        except KeyError:
            target_collection = bpy.data.collections.new(target_collection_name)
            target_collection.hide_render = True
        bpy.context.scene.collection.children.link(target_collection)
        # hide in viewport
        obj_map["viewlayer"].layer_collection.children[target_collection.name].hide_viewport = True
        import_collection.objects.unlink(obj)
        target_collection.objects.link(obj) 
        
def setModifiersEnabled(collection, particles_enabled, displacement_enabled,mesh_material, particle_material):
    mesh,curve = getMeshAndPathFromCollection(collection)
    if mesh_material:
        mesh.active_material = bpy.data.materials[mesh_material]
    if particle_material:
        bpy.data.objects["DefaultParticle"].active_material = bpy.data.materials[particle_material]
    mesh.modifiers['particles'].show_render = particles_enabled
    mesh.modifiers['displace'].show_render = displacement_enabled
    mesh.modifiers['subsurf'].show_render = displacement_enabled
    
def main():
    try:
        idx = sys.argv.index("--")
        cli_arguments = True
    except: 
        cli_arguments= False
        
    if cli_arguments: 
        arg_string = sys.argv[idx+1:]
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", type=str,choices=["import", "render","test"])
        parser.add_argument("frames", type= int)
        parser.add_argument("--light-energy-min", dest="light_energy_min", type= float, default = 0.002)
        parser.add_argument("--light-energy-max", dest="light_energy_max", type= float, default = 0.005)
        parser.add_argument("--spotsize-min", dest = "spotsize_min", type = float, default = 90)
        parser.add_argument("--spotsize-max", dest = "spotsize_max", type = float, default = 130)
        parser.add_argument("--distance-min", dest="distance_min", type= float, default = 0.005)
        parser.add_argument("--distance-max", dest="distance_max", type= float, default = 0.03)
        parser.add_argument("--material",type=str,default = "DefaultMaterial")
        parser.add_argument("--particle-material",type=str,default = "DefaultParticleMaterial")
        parser.add_argument('--enable-particles', dest='enable_particles', action='store_true')
        parser.add_argument('--enable-displace', dest='enable_displace', action='store_true')
        parser.set_defaults(gen_gan_data=False,enable_particles=False, enable_displace = False)
        args = parser.parse_args(arg_string)
        obj_map['args'] = args
        mode = args.mode
        frames_per_sim = args.frames
        enable_particles = args.enable_particles
        enable_displace = args.enable_displace
        material=args.material
        particle_material = args.particle_material
        randomize_view = True
    else: 
        mode = "render-only"
        selected_collection = "sim-labels-135_bladder_smooth"
        frames_per_sim = 10
        enable_particles = True
        enable_displace = True
        material= "DefaultMaterial"
        particle_material = "DefaultParticleMaterial"
        enable_particles = True
        enable_displace = True
        if mode == "test":
            randomize_view = True
        else:
            randomize_view = False
    obj_map["viewlayer"] = bpy.context.scene.view_layers['ViewLayer']
    
    if mode == "render" or mode == "test":
        depth_node, img_node = init(frames_per_sim, randomize_view)
    sim_collections = getSimulationCollections()
    if mode == "render":
        screenshot_folder = "/graphics/scratch/students/zahnjoha/DepthMapsFromEndoVideos/BladderDepthEstimation/datasets/depth_data"
        screenshot_folder = screenshot_folder +"_"+ material + "_" + args.particle_material
        if enable_particles:
            screenshot_folder = screenshot_folder + "_par"
        if enable_displace:
            screenshot_folder = screenshot_folder + "_dis"
            
        try:
            for sim_collection in sim_collections.values():
                setAnimationTarget(sim_collection)
                setModifiersEnabled(sim_collection,enable_particles,enable_displace,material,args.particle_material)
                renderAnimation(sim_collection, img_node, depth_node, screenshot_folder)
        except KeyboardInterrupt:
            pass
            
    elif mode == "test":
        sim_collection = sim_collections[selected_collection]
        setAnimationTarget(sim_collection)
        setModifiersEnabled(sim_collection,enable_particles,enable_displace, material, particle_material)
        
    elif mode =="render-only":
        screenshot_folder = "C:\tmp"
        init_render_settings(bpy.data.scenes['Scene'], "CYCLES")
        sim_collection = sim_collections[selected_collection]
        sim_collection.hide_render= False
        render()
        sim_collection.hide_render= True  
    elif mode == "import":
        importMeshes()
    
main()