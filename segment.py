import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import cv2
import mediapipe as mp
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2

class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node('image_segmentation_node', anonymous=True)

        # Get camera name from parameter server or use default
        self.camera_name = rospy.get_param('~camera_name', 'camera')

        self.bridge = CvBridge()

        # Initialize MediaPipe Selfie Segmentation
        self.base_options = mp.tasks.BaseOptions(model_asset_path='selfie_segmenter.tflite',
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU)
        # self.base_options = mp.tasks.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
        #                                          delegate=mp.tasks.BaseOptions.Delegate.CPU)

        self.segmenter_options = mp.tasks.vision.ImageSegmenterOptions(base_options=self.base_options,
                                                                       output_category_mask=True)
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(self.segmenter_options)

        # ROS Subscribers
        color_topic = f'{self.camera_name}/color/image_rect_color'
        depth_topic = f'{self.camera_name}/aligned_depth_to_color/image_raw'
        camera_info_topic = f'{self.camera_name}/color/camera_info'

        self.image_sub = rospy.Subscriber(color_topic, Image, self.image_callback)
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        # ROS Publishers
        self.pointcloud_pub = rospy.Publisher('segmented_pointcloud', PointCloud2, queue_size=10)
        self.segmented_image_pub = rospy.Publisher('segmented_image', Image, queue_size=10)

        self.color_image = None
        self.depth_image = None
        self.camera_info_received = False
        self.fx = self.fy = self.cx = self.cy = None

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.camera_info_received = True

    def image_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_images()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.process_images()

    def process_images(self):
        if self.color_image is None or self.depth_image is None or not self.camera_info_received:
            return

        # Apply segmentation
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.color_image)
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask.numpy_view()
        mask = category_mask == 0  # Assuming the mask of interest is labeled as 0

        # Debug: Check mask and images
        rospy.loginfo(f"Mask shape: {mask.shape}, Color image shape: {self.color_image.shape}, Depth image shape: {self.depth_image.shape}")

        # Create debug segmented image
        segmented_image = self.color_image.copy()
        segmented_image[~mask] = 0
        self.publish_segmented_image(segmented_image)

        # Create point cloud
        pointcloud = self.create_pointcloud(self.color_image, self.depth_image, mask)
        self.publish_pointcloud(pointcloud)
    def create_pointcloud(self, color_image, depth_image, mask):
        height, width = depth_image.shape

        # Create meshgrid for pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten arrays
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_image.flatten()
        mask_flat = mask.flatten()
        color_flat = color_image.reshape(-1, 3)

        # Apply mask
        u_flat = u_flat[mask_flat]
        v_flat = v_flat[mask_flat]
        depth_flat = depth_flat[mask_flat]
        color_flat = color_flat[mask_flat]

        # Convert depth to meters
        z = depth_flat / 1000.0

        # Compute 3D coordinates
        x = (u_flat - self.cx) * z / self.fx
        y = (v_flat - self.cy) * z / self.fy

        # Convert to correct types
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        r = color_flat[:, 0].astype(np.uint8)
        g = color_flat[:, 1].astype(np.uint8)
        b = color_flat[:, 2].astype(np.uint8)

        # Stack into points array
        points = np.vstack((x, y, z, r, g, b)).T

        print(points)
        rospy.loginfo(f"Number of points: {points.shape[0]}")  # Debug: Check number of points

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.UINT8, 1),
            PointField('g', 13, PointField.UINT8, 1),
            PointField('b', 14, PointField.UINT8, 1),
        ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'camera_link'

        pointcloud = pc2.create_cloud(header, fields, points)

        return pointcloud

    def publish_pointcloud(self, pointcloud):
        self.pointcloud_pub.publish(pointcloud)

    def publish_segmented_image(self, segmented_image):
        segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='bgr8')
        self.segmented_image_pub.publish(segmented_image_msg)

if __name__ == '__main__':
    try:
        node = ImageSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
