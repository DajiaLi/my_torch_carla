import glob
import os
import time
import sys
import numpy as np
import cv2
import math
from config import dqn_setting, ddpg_setting

try:
    sys.path.append(glob.glob('E:/carla/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = dqn_setting.WIDTH
IM_HEIGHT = dqn_setting.HEIGHT
SHOW_PREVIEW = False  ## for debugging purpose
SECONDS_PER_EPISODE = dqn_setting.SECONDS_PER_EPISODE
ACTION_SLEEP_TIME = dqn_setting.ACTION_SLEEP_TIME

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = dqn_setting.STEER_AMT
    SPEED_AMT = dqn_setting.SPEED_AMT
    SPEED_AMT_DDPG = ddpg_setting.SPEED_AMT_DDPG
    DSEIRED_DIS = 5
    im_width = None
    im_height = None
    front_camera = None

    def __init__(self, im_width=IM_WIDTH, im_height=IM_HEIGHT, startlocation="default", action_type="discrete", reward_type="discrete"):
        self.startlocation = startlocation
        self.reward_type = reward_type
        self.action_type = action_type
        if self.action_type == "discrete":
            self.SPEED_THRESHOLD = dqn_setting.SPEED_THRESHOLD
        elif self.action_type == "continuous":
            self.SPEED_THRESHOLD = ddpg_setting.SPEED_THRESHOLD
        self.client = carla.Client(host='127.0.0.1', port=2000)
        self.client.set_timeout(2.0)
        self.world = self.client.load_world('Town03')
        self.blueprint_library = self.world.get_blueprint_library()

        weather = carla.WeatherParameters(sun_altitude_angle=90.0)
        self.world.set_weather(weather)

        self.model_3 = self.blueprint_library.filter("model3")[0]

        CarEnv.im_width = im_width
        CarEnv.im_height = im_height
        self.waypoints = None
        self.filtered_waypoints = None
        self.actor_list = None
        self.collision_hist = None
        self.vehicle = None
        self.transform = None
        self.colsensor = None
        self.rgb_cam = None
        self.sensor = None

    def reset(self):
        # 销毁之前所有的actor
        if self.actor_list is not None and len(self.actor_list) > 0:
            for actor in self.actor_list:
                actor.destroy()

        self.collision_hist = []
        self.actor_list = []

        # self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
        #
        # self.filtered_waypoints = []  ## chaned
        # for self.waypoint in self.waypoints:
        #     if (self.waypoint.road_id == 10):
        #         self.filtered_waypoints.append(self.waypoint)
        #
        # self.spawn_point = self.filtered_waypoints[1].transform
        # self.spawn_point.location.z += 2
        # self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints

        # 在平地的测试
        # self.transform = self.world.get_map().get_spawn_points()[43]
        # self.transform.rotation.yaw = 90
        # self.transform.location.x -= 3.5
        # print("self.transform:", self.transform)
        self.transform = self.get_start_location()
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2, y=-0.4, z=1.07))  # 相机对于车的相对位置
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        time.sleep(4)  # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:  ## return the observation
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        # # 测试用
        # places = self.world.get_map().get_spawn_points()
        # for i in range(len(places)):
        #     self.world.debug.draw_string(places[i].location, str(i), draw_shadow=False,
        #                                  color=carla.Color(r=0, g=255, b=0), life_time=400,
        #                                  persistent_lines=True)

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((CarEnv.im_height, CarEnv.im_width, 4))
        i3 = i2[:, :, :3]
        rgb = i3[:, :, ::-1]  # BGR转为RGB
        if CarEnv.SHOW_CAM:
            cv2.imshow("", rgb)
            cv2.waitKey(1)
        self.front_camera = rgb  ## remember to scale this down between 0 and 1 for CNN input purpose

    def step(self, action):
        '''
        For now let's just pass steer left, straight, right
        0, 1, 2
        '''

        reward = 0
        done = None

        if self.action_type == "discrete":
            if action == 0:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0 * self.SPEED_AMT, steer=-1.0 * self.STEER_AMT))
            if action == 1:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0 * self.SPEED_AMT, steer=0.0))
            if action == 2:
                self.vehicle.apply_control(carla.VehicleControl(throttle=1.0 * self.SPEED_AMT, steer=1.0 * self.STEER_AMT))
        elif self.action_type == "continuous":
            self.vehicle.apply_control(carla.VehicleControl(throttle=min(self.SPEED_AMT_DDPG + float(action[1]), 1.0), steer=float(action[0])))
        # 采取行动后需要进一步观察， 60表示FPS
        time.sleep(ACTION_SLEEP_TIME)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        if len(self.collision_hist) != 0:
            done = True
            reward = 1
        elif self.reward_type == "discrete":
            reward = -1 if kmh < self.SPEED_THRESHOLD else 1
        elif self.reward_type == "linear":
            reward = kmh * (1 - (-1)) / 50 - 1
            # print(reward)

        if self.episode_start + SECONDS_PER_EPISODE < time.time():  ## when to stop
            done = True

        return self.front_camera, reward, done, None

    def get_dis(self):
        # print("start:",self.transform)
        trans = self.vehicle.get_transform()
        # print("end:",trans)
        dis = self.transform.location.distance(self.vehicle.get_transform().location)
        # print("dis:", dis)
        return dis

    def get_start_location(self):
        if self.startlocation == "default":
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
            self.filtered_waypoints = []  ## chaned
            for waypoint in self.waypoints:
                if (waypoint.road_id == 10):
                    self.filtered_waypoints.append(waypoint)
            self.transform = self.filtered_waypoints[1].transform
            self.transform.location.z += 2
        elif self.startlocation == "default2":
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
            self.filtered_waypoints = []  ## chaned
            for waypoint in self.waypoints:
                if (waypoint.road_id == 10):
                    self.filtered_waypoints.append(waypoint)
            self.transform = self.filtered_waypoints[1].transform
            self.transform.location.x += 2.5
            self.transform.location.y += 2.5
            self.transform.location.z += 2
        elif self.startlocation == "default3":
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
            self.filtered_waypoints = []  ## chaned
            for waypoint in self.waypoints:
                if (waypoint.road_id == 10):
                    self.filtered_waypoints.append(waypoint)
            self.transform = self.filtered_waypoints[1].transform
            self.transform.location.x -= 8
            self.transform.location.y += 25
            self.transform.location.z += 2
            self.transform.rotation.yaw = -90
        elif self.startlocation == "point43":
            self.transform = self.world.get_map().get_spawn_points()[43]
            self.transform.rotation.yaw = 0
            self.transform.location.x -= 3.5
        elif self.startlocation == "point3":
            self.transform = self.world.get_map().get_spawn_points()[3]
        return self.transform




if __name__ == '__main__':
    env = CarEnv(startlocation="default2")
    env.reset()
