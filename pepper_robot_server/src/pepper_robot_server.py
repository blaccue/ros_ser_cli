#!/usr/bin/env python

import rospy
import pybullet as p # PyBullet physics simulation
import pybullet_data # PyBullet data for models
from sensor_msgs.msg import JointState, Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import numpy as np 
import cv2 # OpenCV for image processing
from cv_bridge import CvBridge # OpenCV to ROS Image conversion
import threading  # Threading for concurrency
from stable_baselines3 import PPO  # RL algorithm
from stable_baselines3.common.env_util import make_vec_env  # RL environment
import torch  # PyTorch for neural network computation
import mmap  # For shared memory
import struct  # For packing/unpacking data

class PepperRobotServer:
    def __init__(self):
        rospy.init_node('pepper_robot_server', anonymous=True)
        
        self.command_sub = rospy.Subscriber('/robot/command', String, self.command_callback)
        self.vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.vel_callback)
        self.sensor_sub = rospy.Subscriber('/sensor_topic', LaserScan, self.sensor_callback)
        
        self.status_pub = rospy.Publisher('/robot/status', String, queue_size=10)
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.camera_pub = rospy.Publisher('/camera/rgb/image_raw', Image, queue_size=10)
        self.laser_pub = rospy.Publisher('/scan', LaserScan, queue_size=10)
        
        # Setup PyBullet simulation
        self.setup_simulation()
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        self.lock = threading.Lock()

        # Shared memory setup
        self.shm = mmap.mmap(-1, 1024)
        
        # GPU Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup PPO model
        self.env = make_vec_env('MinitaurBulletEnv-v0', n_envs=1)
        self.model = PPO('MlpPolicy', self.env, verbose=1)
        self.model.learn(total_timesteps=10000)

    def setup_simulation(self):
        # Connect to PyBullet and load robot model
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF("pepper_description/urdf/pepper.urdf", useFixedBase=True)
        p.setGravity(0, 0, -9.8)
        self.camera_setup()
        self.load_objects()

    def camera_setup(self):
        # Setup camera in PyBullet
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1, 1, 1],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
        )

    def load_objects(self):
        # Load dynamic and soft body objects into the simulation
        
        # Dynamic object (e.g., a rigid box)
        self.dynamic_object_id = p.loadURDF("r2d2.urdf", [0, 0, 1])
        
        # Soft body object (e.g., a deformable sphere)
        self.soft_body_id = p.loadSoftBody("soft.obj", basePosition=[1, 1, 1], scale=0.5)
        p.setRealTimeSimulation(1)

    def command_callback(self, msg):
        # Handle commands to the robot
        with self.lock:
            command = msg.data
            if command == 'move':
                self.move_arm()
            elif command == 'grasp':
                self.grasp(command)
            self.status_pub.publish(f"Executed command: {command}")

    def vel_callback(self, msg):
        with self.lock:
            linear = msg.linear
            angular = msg.angular
            p.resetBaseVelocity(self.robot_id, [linear.x, linear.y, linear.z], [angular.x, angular.y, angular.z])

    def sensor_callback(self, data):
        # Handle sensor data and write to shared memory
        self.write_to_shm(data.ranges[0])

    def write_to_shm(self, data):
        # Write data to shared memory
        self.shm.seek(0)
        self.shm.write(struct.pack('d', data))

    def read_from_shm(self):
        # Read data from shared memory
        self.shm.seek(0)
        return struct.unpack('d', self.shm.read(8))[0]

    def move_arm(self):
        # Some arm movement
        joint_indices = [p.getJointInfo(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] == p.JOINT_REVOLUTE]
        for _ in range(1000):
            p.stepSimulation()  # Step the PyBullet simulation
            for joint_index in joint_indices:
                p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=0.5)
            self.publish_states()

    def grasp(self, command):
         # Some grasping of objects
        gripper_indices = [p.getJointInfo(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id)) if 'gripper' in p.getJointInfo(self.robot_id, i)[1].decode('UTF-8')]
        if 'soft' in command:
            target_position = 0.3  # Less force for soft objects
        else:
            target_position = 0.1  # More force for hard objects

        for _ in range(100):
            p.stepSimulation()  # Step the PyBullet simulation
            for joint_index in gripper_indices:
                p.setJointMotorControl2(self.robot_id, joint_index, p.POSITION_CONTROL, targetPosition=target_position)
            self.publish_states()

    def publish_states(self):
        self.publish_joint_states()
        self.publish_odom()
        self.publish_camera_image()
        self.publish_laser_scan()

    def publish_joint_states(self):
        joint_states = JointState()
        joint_positions = [p.getJointState(self.robot_id, i)[0] for i in range(p.getNumJoints(self.robot_id))]
        joint_states.position = joint_positions
        self.joint_pub.publish(joint_states)

    def publish_odom(self):
        odom = Odometry()
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = position
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = orientation
        self.odom_pub.publish(odom)

    def publish_camera_image(self):
        width, height, rgbImg, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix
        )
        rgbImg = np.reshape(rgbImg, (480, 640, 4))
        rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGBA2RGB)
        image_msg = self.bridge.cv2_to_imgmsg(rgbImg, "rgb8")
        self.camera_pub.publish(image_msg)

    def publish_laser_scan(self):
        laser_scan = LaserScan()
        laser_scan.angle_min = -1.57
        laser_scan.angle_max = 1.57
        laser_scan.range_min = 0.1
        laser_scan.range_max = 30.0
        laser_scan.ranges = [1.0] * 100
        self.laser_pub.publish(laser_scan)

    def run(self):
        while not rospy.is_shutdown():
            # RL model takes action
            observation = self.env.reset()  # Get initial observation
            action, _states = self.model.predict(observation, deterministic=True)
            self.env.step(action)  # Take action in the environment

            self.publish_states()
            self.rate.sleep()

if __name__ == '__main__':
    server = PepperRobotServer()
    server.run()