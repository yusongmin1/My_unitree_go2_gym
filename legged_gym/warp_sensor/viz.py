import viser
import trimesh
import numpy as np

class WarpViser():
    def __init__(self, warp_manager):
        self._viewer = viser.ViserServer()
        self._scene_frame = {}
        self.warp_manager = warp_manager
        self._current_env_id = 0
        self._setup_frame()
        self._setup_scene()

    def _setup_frame(self):
        self._env_slider = self._viewer.add_slider(
            "Environment ID",
            min=0,
            max=self.warp_manager.num_envs - 1,
            initial_value=0,
            step=1,
        )
        # FIXME: bug exists here
        @self._env_slider.on_update
        def _on_env_change(value):
            self._current_env_id = int(value)
            self._setup_env_frames()
        
        self._setup_env_frames()
        
    def _setup_env_frames(self):
        for name in list(self._scene_frame.keys()):
            if name.startswith(f"env_"):
                try:
                    self._viewer.scene.remove(name)
                except:
                    pass
                del self._scene_frame[name]
        
        env_name = f"env_{self._current_env_id}"
        self._env_frame = self._viewer.add_frame(
            name=env_name,
            show_axes=True,
        )
        self._scene_frame[env_name] = True
        
        self._sensor_frames = {}
        for sensor_name in self.warp_manager.sensors.keys():
            frame_name = f"{env_name}/{sensor_name}"
            sensor_frame = self._viewer.add_frame(
                name=frame_name,
                show_axes=True,
            )
            self._sensor_frames[sensor_name] = sensor_frame
            self._scene_frame[frame_name] = True
        
    def _setup_scene(self):
        if hasattr(self.warp_manager.gym_ptr, 'terrain'):
            terrain = self.warp_manager.gym_ptr.terrain
            self._viewer.scene.add_mesh_trimesh(
                name="terrain",
                mesh=trimesh.Trimesh(
                    vertices=terrain.vertices,
                    faces=terrain.triangles),
            )
    
    def update_scene(self):
        self._update_env_frame()
        
        for sensor_name, sensor_data in self.warp_manager.sensors.items():
            self._update_sensor_frame(sensor_name, sensor_data)
            self._update_sensor_data(sensor_name, sensor_data)
    
    def _update_env_frame(self):
        env_name = f"env_{self._current_env_id}"
        
        base_pos = self.warp_manager.base_pos[self._current_env_id].cpu().numpy()
        base_quat = self.warp_manager.base_ori[self._current_env_id].cpu().numpy()

        wxyz = np.array([base_quat[3], base_quat[0], base_quat[1], base_quat[2]])
        self._env_frame.position = base_pos
        self._env_frame.wxyz = wxyz
    
    def _update_sensor_frame(self, sensor_name, sensor_data):
        env_name = f"env_{self._current_env_id}"
        frame_name = f"{env_name}/{sensor_name}"
        
        if sensor_data.pos.shape[0] > self._current_env_id and sensor_data.pos.shape[1] > 0:
            sensor_offset_p = sensor_data.offset_p
            sensor_offset_q = sensor_data.offset_q
            
            if len(sensor_offset_p.shape) == 2:
                offset_pos = sensor_offset_p[self._current_env_id].cpu().numpy()
            else:
                offset_pos = sensor_offset_p.cpu().numpy()
                
            if len(sensor_offset_q.shape) == 2:
                offset_quat = sensor_offset_q[self._current_env_id].cpu().numpy()
            else:
                offset_quat = sensor_offset_q.cpu().numpy()

            wxyz = np.array([offset_quat[3], offset_quat[0], offset_quat[1], offset_quat[2]])
            self._sensor_frames[sensor_name].position = offset_pos
            self._sensor_frames[sensor_name].wxyz = wxyz
    
    def _update_sensor_data(self, sensor_name, sensor_data):
        env_name = f"env_{self._current_env_id}"
        frame_name = f"{env_name}/{sensor_name}"

        if hasattr(sensor_data, 'pcl_b') and sensor_data.pcl_b is not None:
            points = sensor_data.pcl_b[self._current_env_id].cpu().numpy()
            if points.shape[0] > 0:
                try:
                    self._viewer.scene.remove(f"{frame_name}/pointcloud")
                except:
                    pass
                self._viewer.scene.add_point_cloud(
                    name=f"{frame_name}/pointcloud",
                    points=points,
                    colors=np.tile(np.array([0.0, 0.0, 1.0]), (points.shape[0], 1)),
                    point_size = 0.03,
                )

        if hasattr(sensor_data, 'pixels') and sensor_data.pixels is not None and hasattr(sensor_data.cfg, 'pattern'):
            try:
                pattern = sensor_data.cfg.pattern
                height, width = pattern.height, pattern.width

                if hasattr(pattern, 'fx') and hasattr(pattern, 'fy'):
                    fx, fy = pattern.fx, pattern.fy
                else:
                    fx = fy = width / 2 
                
                fov = 2 * np.arctan2(height / 2, fy)
                aspect = width / height

                position = sensor_data.pos[self._current_env_id].flatten().cpu().numpy() if sensor_data.pos.shape[0] > self._current_env_id else np.zeros(3)
                wxyz = sensor_data.quat[self._current_env_id].flatten().cpu().numpy() if sensor_data.quat.shape[0] > self._current_env_id else np.array([1.0, 0.0, 0.0, 0.0])
                image = sensor_data.pixels[self._current_env_id, 0].cpu().numpy()
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=-1)

                downsample_factor = 4

                try:
                    self._viewer.scene.remove(f"{sensor_name}_frustum")
                except:
                    pass
                frame_name = f"{env_name}/{sensor_name}"
                self._viewer.scene.add_camera_frustum(
                    name=f"{frame_name}/frustum",
                    fov=fov,
                    aspect=aspect,
                    scale=0.15,
                    image=image[::downsample_factor, ::downsample_factor],
                )
                
            except Exception as e:
                print(f"Warning: Could not visualize camera {sensor_name}: {e}")
                pass
    
    def close(self):
        if hasattr(self._viewer, 'close'):
            self._viewer.close()
