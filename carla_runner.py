import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
sys.path.append("D:\\Carla\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.12-py3.7-win-amd64.egg")
sys.path.append("D:\\Carla\\WindowsNoEditor\\PythonAPI\\carla")
import carla

IM_WIDTH = 1920
IM_HEIGHT = 1080
_random_seed = 2000
_rng = np.random.RandomState(_random_seed)

# def process_img(image):
#     i = np.array(image.raw_data) 
#     # print(dir(image))
#     print(i.shape)
#     i2 = i.reshape(IM_HEIGHT,IM_WIDTH,4)
#     i3 = i2[:,:,:3]
    

    

#     obj_list=   {"unlabeled": np.array([0, 0, 0]), "building": np.array([0, 0, 1]), "fence": np.array([0, 0, 2]), "other": np.array([0, 0, 3]), "pedestrian": np.array([0, 0, 4])
#                 , "pole": np.array([0, 0, 5]), "roadline": np.array([0, 0, 6]), "road": np.array([0, 0, 7]), "sidewalk": np.array([0, 0, 8]), "vegetation": np.array([0, 0, 9])
#                 , "vehicles": np.array([0, 0, 10]), "wall": np.array([0, 0, 11]), "signs": np.array([0, 0, 12]), "sky": np.array([0, 0, 13]), "ground": np.array([0, 0, 14])
#                 , "bridge": np.array([0, 0, 15]), "railtrack": np.array([0, 0, 16]), "guardrail": np.array([0, 0, 17]), "trafficlight": np.array([0, 0, 18])
#                 , "static": np.array([0, 0, 19]), "dynamic": np.array([0, 0, 20]), "water": np.array([0, 0, 21]), "terrain": np.array([0, 0, 22]) }

#     target_list= {"unlabeled": np.array([0, 0, 0]), "building": np.array([70, 70, 70]), "fence": np.array([100, 40, 40]), "other": np.array([55, 90, 80]), "pedestrian": np.array([220, 20, 60])
#                 , "pole": np.array([153, 153, 153]), "roadline": np.array([157, 234, 50]), "road": np.array([128, 64, 128]), "sidewalk": np.array([244, 35, 232]), "vegetation": np.array([107, 142, 35])
#                 , "vehicles": np.array([0, 0, 142]), "wall": np.array([102, 102, 156]), "signs": np.array([220, 220, 0]), "sky": np.array([70, 130, 180]), "ground": np.array([81, 0, 81])
#                 , "bridge": np.array([150, 100, 100]), "railtrack": np.array([230, 150, 140]), "guardrail": np.array([180, 165, 180]), "trafficlight": np.array([250, 170, 30])
#                 , "static": np.array([110, 190, 160]), "dynamic": np.array([170, 120, 50]), "water": np.array([45, 60, 150]), "terrain": np.array([145, 170, 100]) }

#     # for ij in np.ndindex(i3.shape[:2]):
#     #     b=i3[ij]

    
#     #     langit=np.array([0, 0, 13])
#     #     if np.array_equal(b,langit):
#     #         i3[ij]= [70, 130, 180]
    
#     for ij in np.ndindex(i3.shape[:2]):

#         b=i3[ij]

#         for key in obj_list:
#             c = obj_list[key]
#             if np.array_equal(b,c):
#                 i3[ij]= target_list[key]

#     print(i3)
#     cv2.imshow("",i3)
#     cv2.waitKey(1)
#     return i3/225.0
cur_phase = 0
rgb_phase = 0
sem_phase = 0

#use cv2

def listenerrgb(image):
    global cur_phase, rgb_phase
    if rgb_phase < cur_phase:
        i = np.array(image.raw_data)
        Path('images_rgb/'+ var_name).mkdir(parents=True, exist_ok=True)
        fname = f"images_rgb/{var_name}/{var_name}_{rgb_phase:05d}.jpg"
        im = i.reshape((1080,1920,4))[:,:,:3]
        cv2.imwrite(fname, im)
        rgb_phase += 1
    

# #use save_to_disk
# def listenerrgb(image):
            
#     # with open("world_frames/dc0_frame_{:05d}.bmp".format(image.frame), "wb") as f:
#     #     f.write(image.raw_data)
#     # file_name = Path("D:/Carla/WindowsNoEditor/world_frames/test1.jpg")
#     # if not file_name.is_file():
#     global cur_phase, rgb_phase
#     if rgb_phase < cur_phase:
#         image.save_to_disk(f"images_rgb/{var_name}_{rgb_phase:05d}.jpg")
#         rgb_phase += 1
        
def listenersem(image):
            
    # with open("world_frames/dc0_frame_{:05d}.bmp".format(image.frame), "wb") as f:
    #     f.write(image.raw_data)
    # file_name = Path("D:/Carla/WindowsNoEditor/world_frames/sem_test1.jpg")
    # if not file_name.is_file():
    global cur_phase, sem_phase
    if sem_phase < cur_phase:
        Path('images_sem/'+ var_name).mkdir(parents=True, exist_ok=True)
        image.save_to_disk(f"images_sem/{var_name}/{var_name}_{sem_phase:05d}.jpg",carla.ColorConverter.CityScapesPalette)
        sem_phase += 1

actor_list = []

try:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town01")
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    # set up ego
    bp = blueprint_library.filter("vehicle.lincoln.mkz_2017")[0]
    

    # spawn_point = random.choice(world.get_map().get_spawn_points())
    # Town1
    spawn_point = carla.Transform(
            carla.Location(1.61,200,0.5),
            carla.Rotation(0,270,0))

    # Town3
    # spawn_point = carla.Transform(
    #         carla.Location(140,-201,3),
    #         carla.Rotation(0,180,0))
    
    vehicle = world.spawn_actor(bp, spawn_point)
    # vehicle.set_autopilot(True)
    time.sleep(2)
    actor_list.append(vehicle)

    # set up pedestrian

    distance_x = 0
    distance_y = 3

    bp = blueprint_library.filter("walker.*")
    bp = _rng.choice(bp)
    
    ego_y = vehicle.get_transform().location.y
    ego_x = vehicle.get_transform().location.x
    ego_yaw = vehicle.get_transform().rotation.yaw
    print(ego_yaw)
    spawn_point_pd= carla.Transform(
            carla.Location(x = ego_x - distance_x, y = ego_y - distance_y, z=0.1),
            carla.Rotation(pitch=0,yaw=ego_yaw-90,roll=0))
    pedestrian = world.spawn_actor(bp, spawn_point_pd)
    
    actor_list.append(pedestrian)

    # p_ingestor1 = mp.Process(target=ingestor, args=(imq,))

    # p_ingestor1.start()

    # print(p_ingestor1)

    # set up rgb cam
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov","90")

    spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(sensor)
    sensor.listen(lambda data: listenerrgb(data))
    
    # set up semantic cam
    sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    sem_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    sem_bp.set_attribute("fov","90")

    spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
    sensor_sem = world.spawn_actor(sem_bp, spawn_point, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(sensor_sem)
    sensor_sem.listen(lambda data: listenersem(data))

    # change viewpoint
    transform = carla.Transform(carla.Location(x=ego_x, y=ego_y, z=6.0), carla.Rotation(pitch=-10.0, yaw=180.0, roll=0.000000))
    world.get_spectator().set_transform(transform)
   

   # set variation
    variation_folder = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','variation_file')
    list_variation = glob.glob(os.path.join(variation_folder, '*.csv'))
    variation_file = os.path.join(variation_folder, 'ydist_variation1.csv')

    for v in list_variation:

        # choose specific variation file
        if v == variation_file:
            variation=pd.read_csv(v)
            # drop unnamed columns
            if 'Unnamed: 0' in list(variation.columns):
                variation=variation.drop(columns=['Unnamed: 0'])
            # dataframe to list
            # param_list = variation["y_distance"].tolist()
            param_list = variation.values.tolist()
            var_name= Path(v).stem
        else:
            continue

        # # open all variation files
        # variation=pd.read_csv(v)
        # # drop unnamed columns
        # if 'Unnamed: 0' in list(variation.columns):
        #     variation=variation.drop(columns=['Unnamed: 0'])
        # # dataframe to list
        # # param_list = variation["y_distance"].tolist()
        # param_list = variation.values.tolist()
        # var_name= Path(v).stem
        print(v)
        # positions = [*range(5, 5 + 10)]
        for p in param_list:
            dy=p[0]
            dx=0
            px, py = ego_x - dx, ego_y - dy
            pedestrian.set_location(carla.Location(x = px, y = py, z=0.3))

            # demorgan's !(a & b) = !a | !b
            for _ in range(4):
                _  = world.wait_for_tick()
            # print("AAA", cnt, pedestrian.get_transform().location)

            cur_phase += 1
            while not (rgb_phase == sem_phase == cur_phase):
                _  = world.wait_for_tick()
            # print("BBB", cnt, pedestrian.get_transform().location)

        cur_phase = 0
        rgb_phase = 0
        sem_phase = 0
        print('Finished for {}'.format(v))
    # imq.close()
    # p_ingestor1.join()

finally:
    for actor in actor_list:
        actor.destroy()
    print("cleaned up!")