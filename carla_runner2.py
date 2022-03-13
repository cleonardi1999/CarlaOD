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
import time
from PIL import Image
import requests
import io
from tqdm import tqdm
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

def make_rgb_listener(observed, obs):
    def calc_bb(image):
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        r = requests.post("http://127.0.0.1:8080/predictions/fastrcnn", data=buf.getvalue())
        
        def unpack(obj):
            score = obj['score']
            [(name, box)] = [(n, b) for n, b in obj.items() if n != "score"]
            return name, box, score

        return [(n, b, s) for n, b, s in [unpack(obj) for obj in r.json()] if n in observed]

    def listenerrgb(image):
        global cur_phase, rgb_phase
        if rgb_phase < cur_phase:
            i = np.array(image.raw_data)
            Path('images_rgb/'+ var_name).mkdir(parents=True, exist_ok=True)
            # fname = f"images_rgb/{var_name}/{var_name}_{rgb_phase:05d}.jpg"
            im = Image.fromarray(i.reshape((1080,1920,4))[:,:,:3])
            nbs = calc_bb(im)
            n = [n for n, _, _ in nbs]
            b = [b for _, b, _ in nbs]
            s = [s for _, _, s in nbs]
            obs.append([f"{var_name}_{rgb_phase:05d}", n, b, s])
            rgb_phase += 1
    return listenerrgb
    

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
        
def make_sem_listener(observed, obs):
    obj_list= {"unlabeled": 0, "building": 1, "fence": 2, "other": 3, "person": 4
        , "pole": 5, "roadline": 6, "road": 7, "sidewalk":8, "vegetation": 9
        , "car":10, "wall": 11, "signs": 12, "sky": 13, "ground": 14
        , "bridge": 15, "railtrack": 16, "guardrail": 17, "trafficlight": 18
        , "static": 19, "dynamic": 20, "water": 21, "terrain": 22 }

    def calc_bb(im_):
        boxes = []
        boxes_name = []

        for key in obj_list:
            if key in observed:
                col_filter = obj_list[key]
                f = im_ == col_filter
                n, m = f.shape
                top = np.any(f, axis=1).argmax()
                bottom = (n-1) - np.any(f, axis=1)[::-1].argmax()
                left = np.any(f, axis=0).argmax()
                right = (m-1) - np.any(f, axis=0)[::-1].argmax()
                box = [left,top,right,bottom]
                if box != [0,0,m-1,n-1]:
                    boxes.append(box)
                    boxes_name.append(key)
        return boxes_name, boxes
    
    def listenersem(image):
        # with open("world_frames/dc0_frame_{:05d}.bmp".format(image.frame), "wb") as f:
        #     f.write(image.raw_data)
        # file_name = Path("D:/Carla/WindowsNoEditor/world_frames/sem_test1.jpg")
        # if not file_name.is_file():
        global cur_phase, sem_phase
        if sem_phase < cur_phase:
            Path('images_sem/'+ var_name).mkdir(parents=True, exist_ok=True)
            # image.save_to_disk(f"images_sem/{var_name}/{var_name}_{sem_phase:05d}.jpg",carla.ColorConverter.CityScapesPalette)
            obs.append([f"{var_name}_{sem_phase:05d}", *calc_bb(np.array(image.raw_data).reshape((1080,1920,4))[:,:,2])])
            sem_phase += 1

    return listenersem

actor_list = []

try:
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(60.0)
    world = client.load_world("Town03")
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    # set up ego
    bp = blueprint_library.filter("vehicle.lincoln.mkz_2017")[0]
    time_start = time.time()

    # spawn_point = random.choice(world.get_map().get_spawn_points())
    # Town1
    # spawn_point = carla.Transform(
    #         carla.Location(1.61,200,0.5),
    #         carla.Rotation(0,270,0))

    # Town3
    # spawn_point = carla.Transform(
    #         carla.Location(140,-201,3),
    #         carla.Rotation(0,180,0))
    
    spawn_point = carla.Transform(
            carla.Location(-20,201,3),
            carla.Rotation(0,180,0))    
    vehicle = world.spawn_actor(bp, spawn_point)
    # vehicle.set_autopilot(True)
    time.sleep(2)
    actor_list.append(vehicle)

    # set up pedestrian

    distance_x = 4
    distance_y = 7

    bp = blueprint_library.filter("walker.*")
    bp = _rng.choice(bp)
    
    ego_y = vehicle.get_transform().location.y
    ego_x = vehicle.get_transform().location.x
    ego_yaw = vehicle.get_transform().rotation.yaw
    print(ego_yaw)
    spawn_point_pd= carla.Transform(
            carla.Location(x = ego_x - distance_x, y = ego_y - distance_y, z=1),
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

    rgb_observations = []
    sensor.listen(make_rgb_listener({"person"}, rgb_observations))
    
    # set up semantic cam
    sem_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    sem_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    sem_bp.set_attribute("fov","90")

    spawn_point = carla.Transform(carla.Location(x=0.8, z=1.7))
    sensor_sem = world.spawn_actor(sem_bp, spawn_point, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(sensor_sem)

    sem_observations = []
    sensor_sem.listen(make_sem_listener({"car", "person"}, sem_observations))

    # change viewpoint
    transform = carla.Transform(carla.Location(x=ego_x, y=ego_y, z=6.0), carla.Rotation(pitch=-10.0, yaw=180.0, roll=0.000000))
    world.get_spectator().set_transform(transform)
   

   # set variation
    variation_folder = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','variation_file')
    list_variation = glob.glob(os.path.join(variation_folder, '*.csv'))
    variation_file = os.path.join(variation_folder, 'xydist_variation4.csv')

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
        for p in tqdm(param_list):
            dy=p[0]
            dx=p[1]
            # rain=p[2]
            px, py = ego_x - dx, ego_y - dy
            pedestrian.set_location(carla.Location(x = px, y = py, z=1))
            # weather = carla.WeatherParameters(cloudiness=10.000000, precipitation=rain, 
            #                         precipitation_deposits=0.000000, wind_intensity=10.000000, sun_azimuth_angle=180.000000, sun_altitude_angle=30.000000, 
            #                         fog_density=0.000000, fog_distance=0.750000, fog_falloff=0.100000, wetness=0.000000, scattering_intensity=1.000000, 
            #                         mie_scattering_scale=0.030000, rayleigh_scattering_scale=0.033100)
            # world.set_weather(weather)
            # print(world.get_weather())
            # demorgan's !(a & b) = !a | !b
            for _ in range(16):
                _  = world.wait_for_tick()
            # print("AAA", cnt, pedestrian.get_transform().location)

            cur_phase += 1
            while not (rgb_phase == sem_phase == cur_phase):
                _  = world.wait_for_tick()
            # print("BBB", cnt, pedestrian.get_transform().location)

        cur_phase = 0
        rgb_phase = 0
        sem_phase = 0
    # imq.close()
    # p_ingestor1.join()
        csv_name_rgb = var_name + '.csv'
        csv_path_rgb = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','Bbox_rgb',csv_name_rgb)
        csv_name_sem = var_name + '.csv'
        csv_path_sem = os.path.join('d:',os.sep,'Carla','WindowsNoEditor','Bbox_sem',csv_name_sem)
        pd.DataFrame(rgb_observations, columns=["var_name", "Type", "BBox", "Score"]).set_index("var_name").to_csv(csv_path_rgb)
        pd.DataFrame(sem_observations, columns=["var_name", "Type", "BBox"]).set_index("var_name").to_csv(csv_path_sem)
        rgb_observations.clear()
        sem_observations.clear()

        print('Finished for {}'.format(v))
finally:
    for actor in actor_list:
        actor.destroy()
    time_end=time.time()
    print("cleaned up!, total duration:", time_end-time_start)