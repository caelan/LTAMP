#!/usr/bin/env python

import time
import pybullet as p
import copy

from perception_tools.perception import Perception
from perception_tools.common import NAME_TEMPLATE, get_body_urdf
from plan_tools.common import get_color
from pybullet_tools.utils import create_box, ClientSaver, set_point, wait_for_user, pairwise_collision, remove_body, \
    user_input, Mesh, mesh_from_points, set_pose, load_pybullet, create_mesh, apply_alpha
from collections import namedtuple

from ltamp_perception.msg import Belief
from image_geometry import PinholeCameraModel
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import CameraInfo

ONYX_SCALE_HEIGHT = 0.025

##################################################

BodyInfo = namedtuple('BodyInfo', ['type', 'color', 'pose', 'name'])

def convert_ros_position(position):
    return [position.x, position.y, position.z]

def convert_ros_orientation(orientation):
    return [orientation.x, orientation.y, orientation.z, orientation.w]

def convert_ros_pose(pose):
    return convert_ros_position(pose.position), convert_ros_orientation(pose.orientation)

##################################################


# This perception module uses pybullet 
class ROSPerception(Perception):
    # TODO: could just unify the perception classes since they both do about the same thing
    def __init__(self, world, use_voxels=False):
        """
        __init__
        :param belief_manager: belief manager
        :param surfaces: dictionary of surface ids
        :param objects: dictionary of object ids
        :param pr2: pr2 model
        :param client: pybullet client
        """

        super(ROSPerception, self).__init__(world)
        # Launches the detection and belief construction nodes
        import rospy
        rospy.loginfo("LAUNCHING NODE")

        # Initializing lists for belief management
        self.data = None
        self.voxels = []
        self.use_voxels = use_voxels

        # Dictionaries for sim ids for different objects
        #self.sim_items_ids = {} # e.g. - {greenblock: [3, 4], bluecup: [7]}
        #self.sim_surfaces_ids = {} # e.g. {table: [0, 1]}
        # TODO: use these to keep the name constant
        self.items = []
        self.surfaces = []

        self.num_observations = 0
        self.updated = False
        self.updated_simulation = False

        self.info_from_name = {}

        # Sets up simulator - using pybullet
        #self.p = belief_manager
        #self.holding_liquid = [] # TODO: detecting holding liquid?

        self.camera_info = None
        self.cam_model = PinholeCameraModel()
        self.sub_kinect = rospy.Subscriber(
            "/head_mount_kinect/rgb/camera_info", CameraInfo, self.calibration_cb, queue_size=1)  # 'depth' | 'ir
        self.max_range = rospy.get_param(
            "/head_mount_kinect/disparity_depth/max_range")

        self.markers_pub = rospy.Publisher(
            '~markers', MarkerArray, queue_size=1)

        rospy.Subscriber("/ltamp_perception/belief", Belief, self._obj_belief_cb)

        # TODO: average over a few observations

    def calibration_cb(self, data):
        if self.camera_info is not None:
            return
        # https://github.mit.edu/caelan/stripstream/blob/master/scripts/openrave/run_belief_tamp.py#L193

        # header:
        #   seq: 955
        #   stamp:
        #     secs: 1547592314
        #     nsecs: 343205039
        #   frame_id: head_mount_kinect_rgb_optical_frame
        # height: 480
        # width: 640
        # distortion_model: plumb_bob
        # D: [0.0, 0.0, 0.0, 0.0, 0.0]
        # K: [525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0]
        # R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        # P: [525.0, 0.0, 319.5, 0.0, 0.0, 525.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0]
        # binning_x: 0
        # binning_y: 0
        # roi:
        #   x_offset: 0
        #   y_offset: 0
        #   height: 0
        #   width: 0
        #   do_rectify: False
        #print(data)
        # https://github.mit.edu/caelan/ROS/blob/master/tensorflow_octomap.py
        # http://docs.ros.org/diamondback/api/image_geometry/html/python/
        # TODO: also camera intrinsics in an openrave xml
        # /usr/local/share/openrave-0.9/robots/pr2-beta-sim.robot.xml
        self.camera_info = data
        self.cam_model.fromCameraInfo(self.camera_info)
        #print(self.cam_model.intrinsicMatrix())
        # [[525.    0.  319.5   0. ]
        # [  0.  525.  239.5   0. ]
        # [  0.    0.    1.    0. ]]
        #print('Kinect projection matrix:')
        #print(self.cam_model.projectionMatrix())
        #print(self.cam_model.rotationMatrix())

    def process_data(self, data):
        # Items & Surfaces are sorted by increasing y coordinate
        type_count = {}
        self.items = []
        for item in sorted(data.objects, key=lambda i: i.pose.pose.position.y):
            ty = item.name
            type_count[ty] = type_count.get(ty, 0) + 1
            name = NAME_TEMPLATE.format(ty, type_count[ty])
            pose = convert_ros_pose(item.pose.pose)
            self.items.append(BodyInfo(ty, item.color, pose, name))

        self.surfaces = []
        for surface in sorted(data.surfaces, key=lambda s: s.table.pose.position.y):
            # TODO: use truncated
            ty = surface.name
            type_count[ty] = type_count.get(ty, 0) + 1
            name = NAME_TEMPLATE.format(ty, type_count[ty])
            convex_hull = list(map(convert_ros_position, surface.table.convex_hull))
            mesh = mesh_from_points(convex_hull)
            pose = convert_ros_pose(surface.table.pose)
            #self.surfaces.append(BodyInfo(ty, surface.color, pose, name))
            self.surfaces.append(BodyInfo(mesh, surface.color, pose, name))

        num_voxels = 0
        self.voxels = []
        for marker in data.octree:
            assert marker.header.frame_id == 'base_link'
            #assert marker.pose == unit_point()
            # TODO: ensure these are in the correct frame
            extents = (marker.scale.x, marker.scale.y, marker.scale.z)
            centers = [(p.x, p.y, p.z) for p in marker.points]
            num_voxels += len(centers)
            if centers:
                self.voxels.append((extents, centers))

        #print('Surfaces: {} | Items: {} | Voxels: {}'.format(
        #    len(self.surfaces), len(self.items), num_voxels))

    def _obj_belief_cb(self, data):
        """
        Callback function that keeps the belief state updated
        :param data: belief messages from detection node
        """
        if self.updated_simulation:
            return
        if data is None:
            return
        self.process_data(data)
        #self.data = data
        self.updated = True
        self.num_observations += 1

    # waits for the next update to perception
    # returns True if updated, false if timed out
    # time out is the max wait time in seconds
    # interval is how long it waits before checking again
    def wait_for_update(self, timeout=10, interval=0.1):
        end = time.time() + timeout
        self.updated = False
        while (time.time() < end) and not self.updated:
            time.sleep(interval)
        return self.updated

    def _update_items(self):
        with ClientSaver(self.client):
            for name in list(self.sim_items):
                remove_body(self.get_body(name))
                del self.sim_bodies[name]
                del self.sim_items[name]
                if name in self.info_from_name:
                    del self.info_from_name[name]

            for item in self.items:
                self.sim_items[item.name] = None
                self.info_from_name[item.name] = item
                self.load_body(item.name, pose=item.pose, fixed_base=False)

    def _update_surfaces(self):
        with ClientSaver(self.client):
            for name in list(self.sim_surfaces):
                remove_body(self.get_body(name))
                del self.sim_bodies[name]
                del self.sim_surfaces[name]
                if name in self.info_from_name:
                    del self.info_from_name[name]

            for surface in self.surfaces:
                self.sim_surfaces[surface.name] = None
                self.info_from_name[surface.name] = surface
                if isinstance(surface.type, Mesh):
                    #color = surface.color
                    color = apply_alpha(get_color('tan'), 1.0)
                    # TODO: bottom is still visualizing even with under=False
                    body = create_mesh(surface.type, under=True, color=color)
                    self.sim_bodies[surface.name] = body
                    set_pose(body, surface.pose)
                else:
                    self.load_body(surface.name, pose=surface.pose, fixed_base=True)

    def _update_voxels(self):
        if not self.use_voxels:
            return False
        voxel_time = time.time()
        boxes = []
        with ClientSaver(self.client):
            voxel_distance = 1e-2 # 0.
            detected = list(self.sim_bodies.values())
            #color = (1, 0, 0, 0.5)
            color = None
            for extent, centers in self.voxels:
                box = create_box(*extent, color=color)
                keep_centers = []
                for center in centers:
                    set_point(box, center)
                    if not any(pairwise_collision(body, box, max_distance=voxel_distance) for body in detected):
                        keep_centers.append(center)
                        #wait_for_user()
                remove_body(box)
                for center in keep_centers:
                    box = create_box(*extent, color=(1, 0, 0, 0.5))
                    set_point(box, center)
                    boxes.append(box)
        print('Created {} voxels in {:3f} seconds'.format(len(boxes), time.time() - voxel_time))
        return True

    def update_simulation(self):
        """
        Updates the robot's belief representation in simulation
        """
        self.updated_simulation = True
        self._update_items()
        self._update_surfaces()
        self._update_voxels()
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        return True
