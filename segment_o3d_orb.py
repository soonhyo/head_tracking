import rospy
import open3d as o3d
import open3d.core as o3c
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from collections import deque
import tf.transformations as tf_trans

class PointCloudSLAM:
    def __init__(self):
        rospy.init_node('pointcloud_slam_node', anonymous=True)

        self.bridge = CvBridge()

        self.camera_name = rospy.get_param('~camera_name', 'camera')

        self.depth_sub = message_filters.Subscriber(f'{self.camera_name}/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber(f'{self.camera_name}/color/image_rect_color', Image)
        self.info_sub = message_filters.Subscriber(f'{self.camera_name}/aligned_depth_to_color/camera_info', CameraInfo)
        self.pose_sub = rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.pose_callback)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], 10, 0.5)
        self.ts.registerCallback(self.callback)

        self.integrated_pointcloud_pub = rospy.Publisher('integrated_pointcloud', PointCloud2, queue_size=1)

        self.device = o3c.Device("CUDA:0")
        self.voxel_size = 1.5 / 256
        self.block_resolution = 4
        self.block_count = 5000
        self.depth_scale = 1000.0
        self.depth_max = 1.0
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device
        )

        self.frame_count = 5
        self.depth_queue = deque(maxlen=self.frame_count)
        self.color_queue = deque(maxlen=self.frame_count)
        self.intrinsic = None
        self.extrinsic = np.eye(4)
        self.global_pcd = o3d.geometry.PointCloud()
        self.latest_extrinsic = np.eye(4)  # To store the latest pose from ORB-SLAM3

    def pose_callback(self, pose_msg):
        self.latest_extrinsic = self.pose_to_matrix(pose_msg.pose)

    def pose_to_matrix(self, pose):
        """Convert geometry_msgs/Pose to a 4x4 transformation matrix."""
        trans = [pose.position.x, pose.position.y, pose.position.z]
        rot = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        matrix = tf_trans.quaternion_matrix(rot)
        matrix[:3, 3] = trans
        return matrix

    def callback(self, depth_msg, color_msg, info_msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

            if not depth_image.any():
                print("No depth image")
                return

            self.depth_queue.append(depth_image)
            self.color_queue.append(color_image)

            if len(self.depth_queue) == self.frame_count:
                if self.intrinsic is None:
                    intrinsic = o3d.camera.PinholeCameraIntrinsic()
                    intrinsic.set_intrinsics(info_msg.width, info_msg.height, info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5])
                    self.intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

                self.integrate()
        except CvBridgeError as e:
            rospy.logerr(e)

    def integrate(self):
        for depth_image, color_image in zip(self.depth_queue, self.color_queue):
            depth_o3d = o3d.t.geometry.Image(depth_image).to(self.device)
            color_o3d = o3d.t.geometry.Image(color_image).to(self.device)

            extrinsic = o3d.core.Tensor(self.latest_extrinsic, o3d.core.Dtype.Float64)

            try:
                frustum_block_coords = self.vbg.compute_unique_block_coordinates(
                    depth_o3d, self.intrinsic, extrinsic, self.depth_scale, self.depth_max)

                self.vbg.integrate(frustum_block_coords, depth_o3d, color_o3d, self.intrinsic,
                                   self.intrinsic, extrinsic, self.depth_scale, self.depth_max)
            except RuntimeError as e:
                rospy.logerr(f"Integration failed: {e}")
                return

            pcd = self.vbg.extract_point_cloud().to_legacy()

            if not pcd.is_empty():
                if self.global_pcd.is_empty():
                    self.global_pcd = pcd
                else:
                    self.global_pcd += pcd

        if not self.global_pcd.is_empty():
            self.publish_pointcloud(self.global_pcd)

    def publish_pointcloud(self, pcd):
        points = np.asarray(pcd.points)
        if points.shape[0] == 0:
            rospy.logwarn("No points in point cloud to publish")
            return
        colors = np.asarray(pcd.colors)
        r, g, b = (colors * 255).astype(np.uint8).T
        rgba = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b) | (0xFF << 24)
        points = np.concatenate((points, rgba.reshape(-1, 1)), axis=1, dtype=object)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"

        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('rgba', 12, pc2.PointField.UINT32, 1)]

        points_list = [tuple(point) for point in points]
        cloud_data = pc2.create_cloud(header, fields, points_list)

        self.integrated_pointcloud_pub.publish(cloud_data)

if __name__ == '__main__':
    try:
        slam_node = PointCloudSLAM()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
