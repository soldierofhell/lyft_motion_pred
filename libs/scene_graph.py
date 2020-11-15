import sys
from typing import *
from itertools import chain

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import torch.nn.functional as F

from libs.rasterizer import SemGraphRasterizer

ACTIVE_VAL = 1.
INACTIVE_VAL = 0.

node_types = [SemGraphRasterizer.EGO,
              SemGraphRasterizer.AGENTS,
              SemGraphRasterizer.LANES,
              SemGraphRasterizer.STOP_SIGNS,
              SemGraphRasterizer.SPEED_HUMPS,
              SemGraphRasterizer.SPEED_BUMPS,
              SemGraphRasterizer.CROSSWALKS]
node_type_col_ind = {node_type: ind for ind, node_type in enumerate(node_types)}


def standardize(vec: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (vec - mean) / std


def get_mask(x: torch.Tensor, element_type: str) -> torch.Tensor:
    node_type_col = node_type_col_ind[element_type]

    return torch.nonzero(x[:, node_type_col].eq(ACTIVE_VAL), as_tuple=True)[0]


class SceneGraphBuilder(object):

    def __init__(self, raster_size: Tuple[int] = (224, 224)):
        self.raster_size = raster_size
        self.one_hot_idx = dict(
            access_restriction=dict(UNKNOWN=0,
                                    NO_RESTRICTION=1,
                                    ONLY_HOV=2,
                                    ONLY_BUS=3,
                                    ONLY_BIKE=4,
                                    ONLY_TURN=5),
            orientation_in_parent_segment=dict(UNKNOWN_TRAVEL_DIRECTION=0,
                                               TWO_WAY=1,
                                               ONE_WAY_FORWARD=2,
                                               ONE_WAY_BACKWARD=3,
                                               ONE_WAY_REVERSIBLE=4),
            turn_type_in_parent_junction=dict(UNKNOWN=0,
                                              THROUGH=1,
                                              LEFT=2,
                                              SHARP_LEFT=3,
                                              RIGHT=4,
                                              SHARP_RIGHT=5,
                                              U_TURN=6),
            road_class=dict(UNKNOWN_ROAD_CLASS=0,
                            MOTORWAY=1,
                            TRUNK=2,
                            PRIMARY=3,
                            SECONDARY=4,
                            TERTIARY=5,
                            RESIDENTIAL=6,
                            UNCLASSIFIED=7,
                            SERVICE=8,
                            SERVICE_PARKING_AISLE=9,
                            SERVICE_DRIVEWAY=10,
                            SERVICE_ALLEY=11,
                            SERVICE_EMERGENCY_ACCESS=12,
                            SERVICE_DRIVE_THROUGH=13,
                            MOTORWAY_LINK=14,
                            TRUNK_LINK=15,
                            PRIMARY_LINK=16,
                            SECONDARY_LINK=17,
                            TERTIARY_LINK=18,
                            SERVICE_LIVING_STREET=19,
                            PEDESTRIAN=20,
                            PATH=21,
                            STEPS=22,
                            CYCLEWAY=23),
            road_direction=dict(UNKNOWN_TRAVEL_DIRECTION=0,
                                TWO_WAY=1,
                                ONE_WAY_FORWARD=2,
                                ONE_WAY_BACKWARD=3,
                                ONE_WAY_REVERSIBLE=4)
        )

        self.feature_mean = dict(
            road_speed_limit_meters_per_second=4.265,
            ego_extent=[4.226, 1.765, 1.372],
            ego_yaw=0.331,
            ego_velocity=[-0.116, 0.167],
            agent_extent=[3.067, 1.457, 1.207],
            agent_yaw=0.260,
            agent_velocity=[0.075, -0.003]
        )
        self.feature_std = dict(
            road_speed_limit_meters_per_second=6.468,
            ego_extent=[1.332, 0.339, 0.228],
            ego_yaw=1.786,
            ego_velocity=[4.254, 4.604],
            agent_extent=[1.833, 0.576, 0.463],
            agent_yaw=1.767,
            agent_velocity=[3.811, 3.928]
        )

    def get_one_hot(self, n: int,
                    lane: Dict[str, Any],
                    feature_key: str,
                    active_val: float = ACTIVE_VAL,
                    inactive_val: float = INACTIVE_VAL) -> torch.Tensor:
        feature_val = lane[feature_key]
        feature_map = self.one_hot_idx[feature_key]
        feature_one_col = feature_map[feature_val]
        feature_vec = inactive_val * torch.ones(size=(n, len(feature_map)),
                                                dtype=torch.float32)
        feature_vec[:, feature_one_col] = active_val

        return feature_vec

    @staticmethod
    def get_boolean_feature(n: int,
                            feature_val: bool,
                            active_val: float = ACTIVE_VAL,
                            inactive_val: float = INACTIVE_VAL) -> torch.Tensor:
        feature_val_int = active_val if feature_val else inactive_val
        return torch.ones(size=(n, 1), dtype=torch.float32) * feature_val_int

    def get_singular_feature(self, n: int, featuren_val: float, feature_key: str) -> torch.Tensor:
        feature_mean = self.feature_mean.get(feature_key, 0.)
        feature_sd = self.feature_std.get(feature_key, 1.)
        normed_val = (featuren_val - feature_mean) / feature_sd
        return torch.ones(size=(n, 1), dtype=torch.float32) * normed_val

    def extract_lane_feature(self, lanes: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            features
                - pos (2 float)
                - access_restriction (6 one-hot)
            feature shape: (N, )
        """

        if len(lanes) == 0:
            return torch.empty(0, 63), torch.tensor([])

        lane_features = []
        lane_batches = []

        line_id = 0
        for lane in lanes:
            pos = lane["centroid"]
            mask = self.get_valid_pos_mask(pos)
            pos = pos[mask]
            n = pos.shape[0]

            if n == 0:
                continue

            access_restriction = self.get_one_hot(n, lane, "access_restriction")
            orient_in_parent = self.get_one_hot(n, lane, "orientation_in_parent_segment")
            turn_type_in_parent = self.get_one_hot(n, lane, "turn_type_in_parent_junction")
            has_change_left = self.__class__.get_boolean_feature(n, lane["has_adjacent_lane_change_left"])
            has_change_right = self.__class__.get_boolean_feature(n, lane["has_adjacent_lane_change_right"])
            has_tl = self.__class__.get_boolean_feature(n, lane["tl_count"] > 0)

            red_face_tl = self.__class__.get_boolean_feature(n, lane["tl_signal_red_face_count"] > 0)
            red_left_face_tl = self.__class__.get_boolean_feature(n, lane["tl_signal_left_arrow_red_face_count"] > 0)
            red_right_face_tl = self.__class__.get_boolean_feature(n, lane["tl_signal_right_arrow_red_face_count"] > 0)

            yellow_face_tl = self.__class__.get_boolean_feature(n,
                                                                lane["tl_signal_yellow_face_count"] > 0)
            yellow_left_face_tl = self.__class__.get_boolean_feature(n,
                                                                     lane["tl_signal_left_arrow_yellow_face_count"] > 0)
            yellow_right_face_tl = self.__class__.get_boolean_feature(n,
                                                                      lane[
                                                                          "tl_signal_right_arrow_yellow_face_count"] > 0)

            green_face_tl = self.__class__.get_boolean_feature(n, lane["tl_signal_green_face_count"] > 0)
            green_left_face_tl = self.__class__.get_boolean_feature(n,
                                                                    lane["tl_signal_left_arrow_green_face_count"] > 0)
            green_right_face_tl = self.__class__.get_boolean_feature(n,
                                                                     lane["tl_signal_right_arrow_green_face_count"] > 0)

            can_have_parked_cars = self.__class__.get_boolean_feature(n, lane["can_have_parked_cars"])
            road_speed_limit = self.get_singular_feature(n,
                                                         lane["road_speed_limit_meters_per_second"],
                                                         "road_speed_limit_meters_per_second")
            road_class = self.get_one_hot(n, lane, "road_class")
            road_direction = self.get_one_hot(n, lane, "road_direction")
            features = torch.cat([torch.from_numpy(pos), access_restriction, orient_in_parent, turn_type_in_parent,
                                  has_change_left, has_change_right, has_tl,
                                  red_face_tl, red_left_face_tl, red_right_face_tl,
                                  yellow_face_tl, yellow_left_face_tl, yellow_right_face_tl,
                                  green_face_tl, green_left_face_tl, green_right_face_tl,
                                  can_have_parked_cars, road_speed_limit, road_class, road_direction], dim=1)
            lane_features.append(features)
            lane_batches.append([line_id] * n)

            line_id += 1

        lane_features = torch.cat(lane_features, dim=0)
        lane_batches = torch.tensor(list(chain.from_iterable(lane_batches)))

        return lane_features, lane_batches

    def extract_ego_feature(self, ego: np.ndarray) -> torch.Tensor:
        ego_frames = np.concatenate(ego, axis=0)
        mask = self.get_valid_pos_mask(ego_frames["centroid"])
        ego_frames = ego_frames[mask]

        pos = torch.from_numpy(ego_frames["centroid"].copy())

        extent_mean = torch.tensor(self.feature_mean["ego_extent"])
        extent_std = torch.tensor(self.feature_std["ego_extent"])
        extent = standardize(torch.tensor(ego_frames["extent"]), extent_mean, extent_std)

        yaw_mean = torch.tensor(self.feature_mean["ego_yaw"])
        yaw_std = torch.tensor(self.feature_std["ego_yaw"])
        yaw = standardize(torch.tensor(ego_frames["yaw"]), yaw_mean, yaw_std)
        yaw = torch.unsqueeze(yaw, dim=1)

        velocity_mean = torch.tensor(self.feature_mean["ego_velocity"])
        velocity_std = torch.tensor(self.feature_std["ego_velocity"])
        velocity = standardize(torch.tensor(ego_frames["velocity"]), velocity_mean, velocity_std)

        label_probabilities = torch.from_numpy(ego_frames["label_probabilities"])

        return torch.cat([pos, extent, yaw, velocity, label_probabilities], dim=1)

    def get_valid_pos_mask(self, pos: np.ndarray) -> np.ndarray:
        mask = (pos[:, 0] >= 0) & (pos[:, 0] <= self.raster_size[0]) & \
               (pos[:, 1] >= 0) & (pos[:, 1] <= self.raster_size[1])
        return mask

    def extract_agent_feature(self, agents: np.ndarray) -> torch.Tensor:
        # only present frame
        agent_frames = agents[0]

        valid_agent_mask = self.get_valid_pos_mask(agent_frames["centroid"])
        agent_frames = agent_frames[valid_agent_mask]

        pos = torch.from_numpy(agent_frames["centroid"].copy())

        extent_mean = torch.tensor(self.feature_mean["agent_extent"])
        extent_std = torch.tensor(self.feature_std["agent_extent"])
        extent = standardize(torch.tensor(agent_frames["extent"]), extent_mean, extent_std)

        yaw_mean = torch.tensor(self.feature_mean["agent_yaw"])
        yaw_std = torch.tensor(self.feature_std["agent_yaw"])
        yaw = standardize(torch.tensor(agent_frames["yaw"]), yaw_mean, yaw_std)
        yaw = torch.unsqueeze(yaw, dim=1)

        velocity_mean = torch.tensor(self.feature_mean["agent_velocity"])
        velocity_std = torch.tensor(self.feature_std["agent_velocity"])
        velocity = standardize(torch.tensor(agent_frames["velocity"]), velocity_mean, velocity_std)

        label_probabilities = torch.from_numpy(agent_frames["label_probabilities"])

        return torch.cat([pos, extent, yaw, velocity, label_probabilities], dim=1)

    def extract_stop_sign_features(self, stop_sign: List[Dict[str, Any]]) -> torch.Tensor:
        if len(stop_sign) == 0:
            return torch.empty(0, 2)
        stop_sign_pos = np.concatenate([s["centroid"] for s in stop_sign], axis=0)
        mask = self.get_valid_pos_mask(stop_sign_pos)
        stop_sign_pos = stop_sign_pos[mask]

        return torch.from_numpy(stop_sign_pos)

    def extract_speed_hump_features(self,
                                    speed_humps: List[Dict[str, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor]:
        speed_hump_id = 0

        features = []
        batches = []

        for speed_hump in speed_humps:
            pos = speed_hump["centroid"]
            mask = self.get_valid_pos_mask(pos)
            pos = pos[mask]
            n = pos.shape[0]

            if n == 0:
                continue

            features.append(pos)
            batches.append([speed_hump_id] * n)

            speed_hump_id += 1

        if len(features) == 0:
            return torch.empty(0, 2), torch.tensor([])

        features = torch.from_numpy(np.concatenate(features, axis=0))
        batches = torch.tensor(list(chain.from_iterable(batches)))

        return features, batches

    def extract_crosswalk_features(self, crosswalks):
        crosswalk_batch_id = 0

        features = []
        batches = []

        for crosswalk in crosswalks:
            pos = crosswalk["centroid"]
            mask = self.get_valid_pos_mask(pos)
            pos = pos[mask]
            n = pos.shape[0]

            if n == 0:
                continue

            features.append(pos)
            batches.append([crosswalk_batch_id] * n)

            crosswalk_batch_id += 1

        if len(features) == 0:
            return torch.empty(0, 2), torch.tensor([])

        features = torch.from_numpy(np.concatenate(features, axis=0))
        batches = torch.tensor(list(chain.from_iterable(batches)))

        return features, batches

    def extract_feature(self, scene_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        # features
        #   - one-hot node type (7): [ego, agent, lane, stop_sign, speed_hump, speed_bump, crosswalk]
        #   - x_pos
        #   - y_pos
        #   - node type specific features

        graph = scene_info["graph"]
        ego_features = self.extract_ego_feature(graph[SemGraphRasterizer.EGO])
        agent_features = self.extract_agent_feature(graph[SemGraphRasterizer.AGENTS])
        lane_features, lane_batches = self.extract_lane_feature(graph[SemGraphRasterizer.LANES])
        stop_sign_features = self.extract_stop_sign_features(graph[SemGraphRasterizer.STOP_SIGNS])
        speed_hump_features, speed_hump_batches = self.extract_speed_hump_features(
            graph[SemGraphRasterizer.SPEED_HUMPS])
        speed_bump_features, speed_bump_batches = self.extract_speed_hump_features(
            graph[SemGraphRasterizer.SPEED_BUMPS])
        crosswalk_features, crosswalk_batches = self.extract_crosswalk_features(graph[SemGraphRasterizer.CROSSWALKS])

        features = [ego_features, agent_features, lane_features,
                    stop_sign_features, speed_hump_features,
                    speed_bump_features, crosswalk_features]

        node_type_mat = []
        max_feature_dim = -sys.maxsize
        for col, feature in enumerate(features):
            feature_n, feature_dim = feature.shape
            max_feature_dim = max(max_feature_dim, feature_dim)

            _one_hot_mat = torch.ones((feature_n, len(features)), dtype=torch.float) * INACTIVE_VAL
            _one_hot_mat[:, col] = ACTIVE_VAL
            node_type_mat.append(_one_hot_mat)

        node_type_mat = torch.cat(node_type_mat, dim=0)

        padded_features = []
        for feature in features:
            pad_size = max_feature_dim - feature.shape[1]
            padded_feature = F.pad(feature, (0, pad_size), "constant", 0)
            padded_features.append(padded_feature)

        padded_features = torch.cat(padded_features, dim=0)

        x = torch.cat([node_type_mat, padded_features], dim=1)

        get_batch_fn = self.__class__.get_batch_key
        feature = {
            "x": x,
            get_batch_fn(SemGraphRasterizer.LANES): lane_batches,
            get_batch_fn(SemGraphRasterizer.SPEED_HUMPS): speed_hump_batches,
            get_batch_fn(SemGraphRasterizer.SPEED_BUMPS): speed_bump_batches,
            get_batch_fn(SemGraphRasterizer.CROSSWALKS): crosswalk_batches,
        }
        return feature

    def get_target_positions(self, graph_info: Dict[str, Any]) -> torch.Tensor:
        return torch.tensor(graph_info["target_positions"])

    def get_target_availabilities(self, graph_info: Dict[str, Any]) -> torch.Tensor:
        return torch.tensor(graph_info["target_availabilities"], dtype=torch.uint8)

    @classmethod
    def get_pos(cls, x: torch.Tensor) -> torch.Tensor:
        pos_starting_col = len(node_types)
        return x[:, [pos_starting_col, pos_starting_col + 1]]

    @classmethod
    def get_batch_key(cls, ele_type: str):
        assert ele_type in node_type_col_ind
        return f"{ele_type}_batch"

    def knn_graph(self, pos: torch.Tensor, batch: Optional[torch.Tensor], k: int, flow: str) -> torch.Tensor:
        if pos.shape[0] == 0:
            return torch.empty(2, 0, dtype=torch.int64)
        else:
            return knn_graph(pos, batch=batch, k=k, flow=flow)

    def to_data(self, scene_info: Dict[str, Any], k: int, flow: str = "source_to_target") -> Data:
        feature = self.extract_feature(scene_info)

        x = feature["x"]

        ego_mask = get_mask(x, SemGraphRasterizer.EGO)

        crosswalk_mask = get_mask(x, SemGraphRasterizer.CROSSWALKS)
        crosswalk_batch = feature[self.__class__.get_batch_key(SemGraphRasterizer.CROSSWALKS)]

        speed_bump_mask = get_mask(x, SemGraphRasterizer.SPEED_BUMPS)
        speed_bump_batch = feature[self.__class__.get_batch_key(SemGraphRasterizer.SPEED_BUMPS)]

        speed_hump_mask = get_mask(x, SemGraphRasterizer.SPEED_HUMPS)
        speed_hump_batch = feature[self.__class__.get_batch_key(SemGraphRasterizer.SPEED_HUMPS)]

        lane_mask = get_mask(x, SemGraphRasterizer.LANES)
        lane_batch = feature[self.__class__.get_batch_key(SemGraphRasterizer.LANES)]

        y = self.get_target_positions(scene_info)
        availabilities = self.get_target_availabilities(scene_info)
        pos = self.__class__.get_pos(x)

        ego_edge_index = self.knn_graph(pos[ego_mask], batch=None, k=k, flow=flow)
        crosswalk_edge_index = self.knn_graph(pos[crosswalk_mask], batch=crosswalk_batch, k=k, flow=flow)
        speed_bump_edge_index = self.knn_graph(pos[speed_bump_mask], batch=speed_bump_batch, k=k, flow=flow)
        speed_hump_edge_index = self.knn_graph(pos[speed_hump_mask], batch=speed_hump_batch, k=k, flow=flow)
        lane_edge_index = self.knn_graph(pos[lane_mask], batch=lane_batch, k=k, flow=flow)
        scene_edge_index = self.knn_graph(pos, batch=None, k=k, flow=flow)

        return SceneGraphData(x=x.float().to_sparse(),
                              y=y.float().to_sparse(),
                              raster_from_world=torch.tensor(scene_info["raster_from_world"]).float(),
                              agent_from_world=torch.tensor(scene_info["agent_from_world"]).float(),
                              availabilities=availabilities.to_sparse(),
                              ego_edge_index=ego_edge_index,
                              crosswalk_edge_index=crosswalk_edge_index,
                              speed_bump_edge_index=speed_bump_edge_index,
                              speed_hump_edge_index=speed_hump_edge_index,
                              lane_edge_index=lane_edge_index,
                              scene_edge_index=scene_edge_index)


class SceneGraphData(Data):

    @property
    def ego_mask(self):
        return get_mask(self.x, SemGraphRasterizer.EGO)

    @property
    def ego_node_count(self) -> int:
        return self.ego_mask.sum().item()

    @property
    def crosswalk_mask(self):
        return get_mask(self.x, SemGraphRasterizer.CROSSWALKS)

    @property
    def crosswalk_node_count(self) -> int:
        return self.crosswalk_mask.sum().item()

    @property
    def speed_bump_mask(self):
        return get_mask(self.x, SemGraphRasterizer.SPEED_BUMPS)

    @property
    def speed_bump_node_count(self) -> int:
        return self.speed_bump_mask.sum().item()

    @property
    def speed_hump_mask(self):
        return get_mask(self.x, SemGraphRasterizer.SPEED_HUMPS)

    @property
    def speed_hump_node_count(self) -> int:
        return self.speed_hump_mask.sum().item()

    @property
    def lane_mask(self):
        return get_mask(self.x, SemGraphRasterizer.LANES)

    @property
    def lane_node_count(self) -> int:
        return self.lane_mask.sum().item()

    def __inc__(self, key: str, value: int) -> int:
        if key == "ego_edge_index":
            return self.ego_node_count
        elif key == "crosswalk_edge_index":
            return self.crosswalk_node_count
        elif key == "speed_bump_edge_index":
            return self.speed_bump_node_count
        elif key == "speed_hump_edge_index":
            return self.speed_hump_node_count
        elif key == "lane_edge_index":
            return self.lane_node_count
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0
