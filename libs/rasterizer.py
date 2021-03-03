from typing import *
from collections import defaultdict


import numpy as np
from l5kit.data.map_api import MapAPI
from l5kit.geometry import rotation33_as_yaw, transform_point, transform_points
from l5kit.rasterization import SemanticRasterizer, RenderContext, Rasterizer
from l5kit.data import DataManager
from l5kit.rasterization.semantic_rasterizer import filter_tl_faces_by_status, \
    indices_in_bounds
from l5kit.rasterization.box_rasterizer import filter_agents_by_labels, get_ego_as_agent, filter_agents_by_track_id
from l5kit.rasterization import build_rasterizer as l5kit_build_rasterizer
from l5kit.rasterization.rasterizer_builder import _load_metadata
from l5kit.data.proto.road_network_pb2 import Lane, RoadNetworkSegment, MapElement, \
    _ROADNETWORKSEGMENT_ROADCLASS, _ROADNETWORKSEGMENT_TRAVELDIRECTION, _ACCESSRESTRICTION_TYPE, \
    _LANE_TURNTYPE


def build_rasterizer(cfg: dict, data_manager: DataManager, debug: bool = False) -> Rasterizer:
    raster_cfg = cfg["raster_params"]
    map_type = raster_cfg["map_type"]

    if map_type == "semantic_graph":
        dataset_meta_key = raster_cfg["dataset_meta_key"]
        filter_agents_threshold = raster_cfg["filter_agents_threshold"]
        history_num_frames = cfg["model_params"]["history_num_frames"]

        render_context = RenderContext(
            raster_size_px=np.array(raster_cfg["raster_size"]),
            pixel_size_m=np.array(raster_cfg["pixel_size"]),
            center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
            set_origin_to_bottom=False
        )

        semantic_map_filepath = data_manager.require(raster_cfg["semantic_map_key"])
        dataset_meta = _load_metadata(dataset_meta_key, data_manager)
        world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

        return SemGraphRasterizer(render_context, filter_agents_threshold, history_num_frames,
                                  semantic_map_filepath, world_to_ecef, debug)
    else:
        return l5kit_build_rasterizer(cfg, data_manager)


class SemGraphRasterizer(SemanticRasterizer):
    CROSSWALKS = "crosswalks"
    SPEED_BUMPS = "speed_bumps"
    SPEED_HUMPS = "speed_humps"
    STOP_SIGNS = "stop_signs"
    LANES = "lanes"
    AGENTS = "agents"
    EGO = "ego"

    keys = [CROSSWALKS, SPEED_BUMPS, SPEED_HUMPS, STOP_SIGNS, LANES,
            AGENTS, EGO]

    def __init__(
            self,
            render_context: RenderContext,
            filter_agents_threshold: float,
            history_num_frames: int,
            semantic_map_path: str,
            world_to_ecef: np.ndarray,
            debug: bool = False
    ):
        self.debug = debug
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames
        super(SemGraphRasterizer, self).__init__(render_context, semantic_map_path, world_to_ecef)

    def rasterize(self,
                  history_frames: np.ndarray,
                  history_agents: List[np.ndarray],
                  history_tl_faces: List[np.ndarray],
                  agent: Optional[np.ndarray] = None) -> Dict[str, List]:
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(history_frames[0]["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"],
                                          history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)
        world_from_raster = np.linalg.inv(raster_from_world)

        # get XY of center pixel in world coordinates
        center_in_raster_px = np.asarray(self.raster_size) * (0.5, 0.5)
        center_in_world_m = transform_point(center_in_raster_px, world_from_raster)

        nodes = self.get_semantic_nodes(center_in_world_m,
                                        raster_from_world,
                                        history_tl_faces[0])
        nodes.update(self.get_box_nodes(history_frames, history_agents, agent, raster_from_world))

        return nodes

    def get_semantic_nodes(self,
                           center_in_world: np.ndarray,
                           raster_from_world: np.ndarray,
                           tl_faces: np.ndarray) -> Dict[str, List]:
        """Renders the semantic map at given x,y coordinates.

        Args:
            center_in_world (np.ndarray): XY of the image center in world ref system
            raster_from_world (np.ndarray):
            tl_faces (np.ndarray):
        Returns:
            - lanes
            - crosswalk
            - traffic light
            - building
            - agent
        """

        # filter using half a radius from the center
        raster_radius = float(np.linalg.norm(self.raster_size * self.pixel_size)) / 2

        # get active traffic light faces
        active_tl_ids = set(filter_tl_faces_by_status(tl_faces, "ACTIVE")["face_id"].tolist())

        # iterate over each lanes
        lanes = list(self.get_lanes(center_in_world=center_in_world,
                                    raster_from_world=raster_from_world,
                                    raster_radius=raster_radius,
                                    active_tl_ids=active_tl_ids))
        crosswalks = list(self.get_crosswalks(center_in_world=center_in_world,
                                              raster_from_world=raster_from_world,
                                              raster_radius=raster_radius))
        speed_bumps = list(self.get_speed_bumps(center_in_world=center_in_world,
                                                raster_from_world=raster_from_world,
                                                raster_radius=raster_radius))
        speed_humps = list(self.get_speed_humps(center_in_world=center_in_world,
                                                raster_from_world=raster_from_world,
                                                raster_radius=raster_radius))
        stop_signs = list(self.get_stop_signs(center_in_world=center_in_world,
                                              raster_from_world=raster_from_world,
                                              raster_radius=raster_radius))

        output = {
            self.__class__.LANES: lanes,
            self.__class__.CROSSWALKS: crosswalks,
            self.__class__.SPEED_BUMPS: speed_bumps,
            self.__class__.SPEED_HUMPS: speed_humps,
            self.__class__.STOP_SIGNS: stop_signs
        }

        return output

    @staticmethod
    def remove_agents_by_track_id(agents: np.ndarray, track_id: int) -> np.ndarray:
        """Return all agent object (np.ndarray) of a given track_id.

        Arguments:
            agents (np.ndarray): agents array
            track_id (int): agent track id to select

        Returns:
            np.ndarray -- Selected agent.
        """
        return agents[np.nonzero(agents["track_id"] != track_id)[0]]

    def get_box_nodes(self,
                      history_frames: np.ndarray,
                      history_agents: List[np.ndarray],
                      agent: Optional[np.ndarray],
                      raster_from_world: np.ndarray) -> Dict[str, List]:
        """
        remove node not in the latest
        """

        all_agents_info = []
        ego_info = []

        # loop over historical frames
        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)
            agents = np.concatenate([agents, av_agent])

            agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
            agents = self.__class__.remove_agents_by_track_id(agents, track_id=agent["track_id"])

            agent_ego["centroid"] = transform_points(agent_ego["centroid"], raster_from_world)
            agents["centroid"] = transform_points(agents["centroid"], raster_from_world)

            all_agents_info.append(agents)
            ego_info.append(agent_ego)

        return {self.__class__.AGENTS: all_agents_info,
                self.__class__.EGO: ego_info}

    def _get_parent_road_network_segment(self, lane: Lane) -> RoadNetworkSegment:
        parent_road_network_seg_id: str = MapAPI.id_as_str(lane.parent_segment_or_junction)
        parent_road_network_seg: RoadNetworkSegment = self.mapAPI[parent_road_network_seg_id]
        return parent_road_network_seg

    def _get_road_class(self, road: RoadNetworkSegment) -> str:
        try:
            road_class_num: int = road.element.segment.road_class
            return _ROADNETWORKSEGMENT_ROADCLASS.values_by_number.get(road_class_num).name
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return _ROADNETWORKSEGMENT_ROADCLASS.values_by_number.get(0).name

    def _get_road_direction(self, road: RoadNetworkSegment) -> str:
        try:
            travel_direction_num: int = road.element.segment.travel_direction
            return _ROADNETWORKSEGMENT_TRAVELDIRECTION.values_by_number.get(travel_direction_num).name
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return _ROADNETWORKSEGMENT_TRAVELDIRECTION.values_by_number.get(0).name

    def _get_speed_limit(self, road: RoadNetworkSegment, default_speed: float = 13.41) -> float:
        try:
            return road.element.segment.speed_limit_meters_per_second
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return default_speed

    def _get_fwd_lane_count(self, road: RoadNetworkSegment, default_val: int = 2) -> int:
        try:
            return road.element.segment.forward_lane_set.num_driving_lanes
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return default_val

    def _get_bwd_lane_count(self, road: RoadNetworkSegment, default_val: int = 2) -> int:
        try:
            return road.element.segment.backward_lane_set.num_driving_lanes
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return default_val

    def _get_access_restriction(self, lane: Lane) -> str:
        try:
            return _ACCESSRESTRICTION_TYPE.values_by_number.get(lane.access_restriction.type).name
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return _ACCESSRESTRICTION_TYPE.values_by_number.get(0).name

    def _get_orientation(self, lane: Lane) -> str:
        try:
            travel_direction_num: int = lane.orientation_in_parent_segment
            return _ROADNETWORKSEGMENT_TRAVELDIRECTION.values_by_number.get(
                travel_direction_num).name
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return _ROADNETWORKSEGMENT_TRAVELDIRECTION.values_by_number.get(0).name

    def _get_turn_type(self, lane: Lane) -> str:
        try:
            return _LANE_TURNTYPE.values_by_number.get(lane.lane.turn_type_in_parent_junction).name
        except AttributeError as e:
            if self.debug:
                raise e
            else:
                return _LANE_TURNTYPE.values_by_number.get(0).name

    def _categorize_tl(self, traffic_control_ids: Iterable[str]) -> Dict[str, set]:
        output = defaultdict(set)

        colors = ["red", "yellow", "green"]
        signal_types = ["signal", "signal_left_arrow", "signal_right_arrow",
                        "signal_upper_left_arrow", "signal_upper_right_arrow"]

        for element_id in traffic_control_ids:
            element = self.mapAPI[element_id]
            if not element.element.HasField("traffic_control_element"):
                continue

            traffic_control_el = element.element.traffic_control_element

            for color in colors:
                for signal_type in signal_types:
                    key = f"{signal_type}_{color}_face"
                    if traffic_control_el.HasField(key):
                        output[key].add(element_id)
        return output

    @staticmethod
    def is_speed_bump(element: MapElement) -> bool:
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_element = element.element.traffic_control_element
        return bool(traffic_element.HasField("speed_bump") and traffic_element.points_x_deltas_cm)

    @staticmethod
    def is_speed_hump(element: MapElement) -> bool:
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_element = element.element.traffic_control_element
        return bool(traffic_element.HasField("stop_sign") and traffic_element.points_x_deltas_cm)

    @staticmethod
    def is_stop_sign(element: MapElement) -> bool:
        if not element.element.HasField("traffic_control_element"):
            return False
        traffic_element = element.element.traffic_control_element
        return bool(traffic_element.HasField("speed_hump") and traffic_element.geo_frame)

    def get_polyline_coords(self, element_id: str):
        element = self.mapAPI[element_id]
        # assert self.is_speed_bump(element) or self.is_speed_hump(element) or self.is_stop_sign(element)

        el = element.element.traffic_control_element

        xyz = self.mapAPI.unpack_deltas_cm(el.points_x_deltas_cm,
                                              el.points_y_deltas_cm,
                                              el.points_z_deltas_cm,
                                              el.geo_frame)
        return xyz

    def get_stop_sign_coords(self, element_id: str):
        element = self.mapAPI[element_id]
        assert self.is_stop_sign(element)

        stop_sign = element.element.traffic_control_element
        xyz = self.mapAPI.unpack_deltas_cm([0],
                                              [0],
                                              [0],
                                              stop_sign.geo_frame)
        return xyz

    def get_bounds(self) -> dict:
        """
        For each elements of interest returns bounds [[min_x, min_y],[max_x, max_y]] and proto ids
        Coords are computed by the MapAPI and, as such, are in the world ref system.

        Returns:
            dict: keys are classes of elements, values are dict with `bounds` and `ids` keys
        """
        lanes_ids = []
        crosswalks_ids = []
        speed_bumps_ids = []
        speed_humps_ids = []
        stop_signs_ids = []

        lanes_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        crosswalks_bounds = np.empty((0, 2, 2), dtype=np.float)  # [(X_MIN, Y_MIN), (X_MAX, Y_MAX)]
        speed_bump_bounds = np.empty((0, 2, 2), dtype=np.float)
        speed_hump_bounds = np.empty((0, 2, 2), dtype=np.float)
        stop_sign_bounds = np.empty((0, 2, 2), dtype=np.float)

        for element in self.mapAPI:
            element_id = MapAPI.id_as_str(element.id)

            if self.is_speed_bump(element):
                speed_bump_xyz = self.get_polyline_coords(element_id)
                x_min = np.min(speed_bump_xyz[:, 0])
                y_min = np.min(speed_bump_xyz[:, 1])
                x_max = np.max(speed_bump_xyz[:, 0])
                y_max = np.max(speed_bump_xyz[:, 1])

                speed_bump_bounds = np.append(speed_bump_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                speed_bumps_ids.append(element_id)

            if self.is_speed_hump(element):
                speed_hump_xyz = self.get_polyline_coords(element_id)
                x_min = np.min(speed_hump_xyz[:, 0])
                y_min = np.min(speed_hump_xyz[:, 1])
                x_max = np.max(speed_hump_xyz[:, 0])
                y_max = np.max(speed_hump_xyz[:, 1])

                speed_hump_bounds = np.append(speed_hump_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                speed_humps_ids.append(element_id)

            if self.is_stop_sign(element):
                stop_sign_xyz = self.get_stop_sign_coords(element_id)
                x_min = np.min(stop_sign_xyz[:, 0])
                y_min = np.min(stop_sign_xyz[:, 1])
                x_max = np.max(stop_sign_xyz[:, 0])
                y_max = np.max(stop_sign_xyz[:, 1])

                stop_sign_bounds = np.append(stop_sign_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                stop_signs_ids.append(element_id)

            if self.mapAPI.is_lane(element):
                lane = self.mapAPI.get_lane_coords(element_id)
                x_min = min(np.min(lane["xyz_left"][:, 0]), np.min(lane["xyz_right"][:, 0]))
                y_min = min(np.min(lane["xyz_left"][:, 1]), np.min(lane["xyz_right"][:, 1]))
                x_max = max(np.max(lane["xyz_left"][:, 0]), np.max(lane["xyz_right"][:, 0]))
                y_max = max(np.max(lane["xyz_left"][:, 1]), np.max(lane["xyz_right"][:, 1]))

                lanes_bounds = np.append(lanes_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0)
                lanes_ids.append(element_id)

            if self.mapAPI.is_crosswalk(element):
                crosswalk = self.mapAPI.get_crosswalk_coords(element_id)
                x_min = np.min(crosswalk["xyz"][:, 0])
                y_min = np.min(crosswalk["xyz"][:, 1])
                x_max = np.max(crosswalk["xyz"][:, 0])
                y_max = np.max(crosswalk["xyz"][:, 1])

                crosswalks_bounds = np.append(
                    crosswalks_bounds, np.asarray([[[x_min, y_min], [x_max, y_max]]]), axis=0,
                )
                crosswalks_ids.append(element_id)

        return {
            "lanes": {"bounds": lanes_bounds, "ids": lanes_ids},
            "crosswalks": {"bounds": crosswalks_bounds, "ids": crosswalks_ids},
            "speed_bumps": {"bounds": speed_bump_bounds, "ids": speed_bumps_ids},
            "speed_humps": {"bounds": speed_hump_bounds, "ids": speed_humps_ids},
            "stop_signs": {"bounds": stop_sign_bounds, "ids": stop_signs_ids},
        }

    def get_lanes(self,
                  center_in_world: np.ndarray,
                  raster_from_world: np.ndarray,
                  active_tl_ids: set,
                  raster_radius: float) -> Iterator[Dict[str, Any]]:

        lane_indexes = indices_in_bounds(center_in_world,
                                              self.mapAPI.bounds_info["lanes"]["bounds"],
                                              raster_radius)
        for idx in lane_indexes:
            lane_id = self.mapAPI.bounds_info["lanes"]["ids"][idx]
            lane = self.mapAPI[lane_id].element.lane
            parent_road = self._get_parent_road_network_segment(lane)
            lane_coords = self.mapAPI.get_lane_coords(self.mapAPI.bounds_info["lanes"]["ids"][idx])

            lane_tl_ids = set([MapAPI.id_as_str(la_tc) for la_tc in lane.traffic_controls])
            lane_tl_ids = lane_tl_ids.intersection(active_tl_ids)
            tl_categories = self._categorize_tl(lane_tl_ids)

            centroids = [lane_coords["xyz_left"], lane_coords["xyz_right"]]

            for centroid in centroids:
                lane_info = {
                    "lane_id": lane_id,
                    "access_restriction": self._get_access_restriction(lane),
                    "orientation_in_parent_segment": self._get_orientation(lane),
                    "turn_type_in_parent_junction": self._get_turn_type(lane),
                    "centroid": transform_points(centroid[:, :2], raster_from_world),
                    "has_adjacent_lane_change_left": len(MapAPI.id_as_str(lane.adjacent_lane_change_left)) > 0,
                    "has_adjacent_lane_change_right": len(MapAPI.id_as_str(lane.adjacent_lane_change_right)) > 0,
                    "tl_count": len(lane.traffic_controls),
                    "tl_signal_red_face_count": len(tl_categories["signal_red_face"]),
                    "tl_signal_yellow_face_count": len(tl_categories["signal_yellow_face"]),
                    "tl_signal_green_face_count": len(tl_categories["signal_green_face"]),
                    "tl_signal_left_arrow_red_face_count": len(tl_categories["signal_left_arrow_red_face"]),
                    "tl_signal_left_arrow_yellow_face_count": len(tl_categories["signal_left_arrow_yellow_face"]),
                    "tl_signal_left_arrow_green_face_count": len(tl_categories["signal_left_arrow_green_face"]),
                    "tl_signal_right_arrow_red_face_count": len(tl_categories["signal_right_arrow_red_face"]),
                    "tl_signal_right_arrow_yellow_face_count": len(tl_categories["signal_right_arrow_yellow_face"]),
                    "tl_signal_right_arrow_green_face_count": len(tl_categories["signal_right_arrow_green_face"]),
                    "can_have_parked_cars": lane.can_have_parked_cars,
                    "road_speed_limit_meters_per_second": self._get_speed_limit(parent_road),
                    "road_class": self._get_road_class(parent_road),
                    "road_direction": self._get_road_direction(parent_road),
                    "road_fwd_lane_count": self._get_fwd_lane_count(parent_road),
                    "road_bwd_lane_count": self._get_bwd_lane_count(parent_road)
                }

                yield lane_info

    def get_simple_element_info(self,
                                center_in_world: np.ndarray,
                                raster_from_world: np.ndarray,
                                raster_radius: float,
                                element_key: str):
        element_indexes = indices_in_bounds(center_in_world,
                                                 self.mapAPI.bounds_info[element_key]["bounds"],
                                                 raster_radius)
        for idx in element_indexes:
            xyz = self.get_polyline_coords(self.mapAPI.bounds_info[element_key]["ids"][idx])

            xy_pos = transform_points(xyz[:, :2], raster_from_world)

            yield {
                "centroid": xy_pos
            }

    def get_crosswalks(self,
                       center_in_world: np.ndarray,
                       raster_from_world: np.ndarray,
                       raster_radius: float):
        return self.get_simple_element_info(center_in_world, raster_from_world, raster_radius, "crosswalks")

    def get_speed_humps(self,
                        center_in_world: np.ndarray,
                        raster_from_world: np.ndarray,
                        raster_radius: float):
        return self.get_simple_element_info(center_in_world, raster_from_world, raster_radius, "speed_humps")

    def get_speed_bumps(self,
                        center_in_world: np.ndarray,
                        raster_from_world: np.ndarray,
                        raster_radius: float):
        return self.get_simple_element_info(center_in_world, raster_from_world, raster_radius, "speed_bumps")

    def get_stop_signs(self,
                       center_in_world: np.ndarray,
                       raster_from_world: np.ndarray,
                       raster_radius: float):
        return self.get_simple_element_info(center_in_world, raster_from_world, raster_radius, "stop_signs")
