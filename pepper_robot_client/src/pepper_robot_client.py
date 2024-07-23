#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import threading
import mmap
import struct

class PepperRobotClient:
    def __init__(self):
        rospy.init_node('pepper_robot_client')
        self.command_pub = rospy.Publisher('/robot/command', String, queue_size=10)
        self.status_sub = rospy.Subscriber('/robot/status', String, self.status_callback)

        self.lock = threading.Lock()
        self.shm = mmap.mmap(-1, 1024)

    def status_callback(self, msg):
        rospy.loginfo(f"Status: {msg.data}")

    def send_command(self, command):
        self.command_pub.publish(command)

    def read_sensor_data(self):
        with self.lock:
            try:
                self.shm.seek(0)
                return struct.unpack('d', self.shm.read(8))[0]
            except Exception as e:
                rospy.logerr(f"Failed to read sensor data from shared memory: {e}")
                return None

    def run(self):
        while not rospy.is_shutdown():
            command = input("Enter command (move, grasp hard, grasp soft): ")
            self.send_command(command)
            sensor_data = self.read_sensor_data()
            rospy.loginfo(f"Sensor data: {sensor_data}")

if __name__ == '__main__':
    client = PepperRobotClient()
    client.run()