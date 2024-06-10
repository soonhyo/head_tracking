import rospy
import open3d as o3d
import open3d.core as o3c
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import mediapipe as mp
import time
from collections import deque

class ImageSegmentationNode:
    def __init__(self):
        rospy.init_node('image_segmentation_node', anonymous=True)

        # Get camera name from parameter server or use default
        self.camera_name = rospy.get_param('~camera_name', 'camera')

        self.bridge = CvBridge()

        # Initialize MediaPipe Selfie Segmentation
        # self.base_options = mp.tasks.BaseOptions(model_asset_path='selfie_segmenter.tflite',
        #                                          delegate=mp.tasks.BaseOptions.Delegate.CPU)
        self.base_options = mp.tasks.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite',
                                                 delegate=mp.tasks.BaseOptions.Delegate.CPU)

        self.segmenter_options = mp.tasks.vision.ImageSegmenterOptions(base_options=self.base_options,
                                                                       output_category_mask=True)
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(self.segmenter_options)

        # ROS Subscribers
        self.depth_sub = message_filters.Subscriber(f'{self.camera_name}/aligned_depth_to_color/image_raw', Image)
        self.color_sub = message_filters.Subscriber(f'{self.camera_name}/color/image_rect_color', Image)
        self.info_sub = message_filters.Subscriber(f'{self.camera_name}/aligned_depth_to_color/camera_info', CameraInfo)

        # ROS Publishers
        self.pointcloud_pub = rospy.Publisher('segmented_pointcloud', PointCloud2, queue_size=10)
        self.segmented_image_pub = rospy.Publisher('segmented_image', Image, queue_size=10)

        # Synchronize the messages
        self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.color_sub, self.info_sub], 10, 0.5)
        self.ts.registerCallback(self.callback)

        # VoxelBlockGrid Initialization
        self.device = o3c.Device("CUDA:0")  # 'CUDA:0' or 'CPU:0'
        self.voxel_size = 1.5 / 256
        self.block_resolution = 8
        self.block_count = 10000
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



    def callback(self, depth_msg, color_msg, info_msg):
        try:
            # Convert images
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")

            # 큐에 이미지 추가
            if not depth_image.any():
                print("no depth image")
                return

            self.depth_queue.append(depth_image)
            self.color_queue.append(color_image)

            # Camera intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(info_msg.width, info_msg.height, info_msg.K[0], info_msg.K[4], info_msg.K[2], info_msg.K[5])
            intrinsic = o3d.core.Tensor(intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

            # Apply segmentation
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
            segmentation_result = self.segmenter.segment(mp_image)
            category_mask = segmentation_result.category_mask.numpy_view()
            mask = category_mask == 1  # Assuming the mask of interest is labeled as 0

            # Create debug segmented image
            segmented_image = color_image.copy()
            segmented_image[~mask] = 0
            self.publish_segmented_image(segmented_image)

            # Integrate into voxel grid
            self.integrate(depth_image, color_image, intrinsic, self.depth_scale, self.depth_max, info_msg)

        except CvBridgeError as e:
            rospy.logerr(e)

    def publish_pointcloud(self, pcd, camera_info):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        r, g, b = (colors * 255).astype(np.uint8).T
        # Ensure the rgba values are properly packed
        rgba = (np.uint32(r) << 16) | (np.uint32(g) << 8) | np.uint32(b) | (0xFF << 24)
        # Combine point and color data
        points = np.concatenate((points, rgba.reshape(-1,1)), axis=1, dtype=object)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = camera_info.header.frame_id

        fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
                  pc2.PointField('rgba', 12, pc2.PointField.UINT32, 1)]

        # Convert points to list of tuples for create_cloud
        points_list = [tuple(point) for point in points]

        cloud_data = pc2.create_cloud(header, fields, points_list)

        self.pointcloud_pub.publish(cloud_data)

    def integrate(self, depth_image, color_image, intrinsic, depth_scale, depth_max, camera_info):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=self.voxel_size,
            block_resolution=self.block_resolution,
            block_count=self.block_count,
            device=self.device
        )

        for depth_image, color_image in zip(self.depth_queue, self.color_queue):
            start = time.time()

            # OpenCV 이미지를 Open3D 이미지로 변환
            depth_o3d = o3d.t.geometry.Image(depth_image).to(self.device)
            color_o3d = o3d.t.geometry.Image(color_image).to(self.device)

            # 카메라 외부 파라미터 (여기서는 단순화를 위해 단위 행렬 사용)
            # 실제 응용에서는 extrinsic_id를 사용하여 적절한 변환 행렬을 설정해야 합니다.
            extrinsic = np.eye(4)
            extrinsic = o3d.core.Tensor(extrinsic, o3d.core.Dtype.Float64)

            frustum_block_coords = self.vbg.compute_unique_block_coordinates(
                    depth_o3d, intrinsic, extrinsic, depth_scale, depth_max)

            self.vbg.integrate(frustum_block_coords, depth_o3d, color_o3d, intrinsic,
                               intrinsic, extrinsic, depth_scale, depth_max)

            dt = time.time() - start
            rospy.loginfo(f'Finished integrating frames in {dt} seconds')

        pcd = self.vbg.extract_point_cloud().to_legacy()
        if not pcd.is_empty():
            self.publish_pointcloud(pcd, camera_info)

    def publish_segmented_image(self, segmented_image):
        segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding='rgb8')
        self.segmented_image_pub.publish(segmented_image_msg)

if __name__ == '__main__':
    try:
        node = ImageSegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
