from typing import *
import warnings
import glob
import os.path as osp
from multiprocessing import Pool, cpu_count
from functools import partial

from libs.rasterizer import build_rasterizer
from libs.scene_graph import SceneGraphBuilder

import torch
import numpy as np
from torch_geometric.data import Dataset
from l5kit.data import ChunkedDataset, LocalDataManager, get_frames_slice_from_scenes
from l5kit.dataset import AgentDataset
from l5kit.configs import load_config_data
from tqdm.auto import tqdm


class AgentGraphDataset(AgentDataset):

    def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            track_id (Optional[int]): the agent to rasterize or None for the AV
        Returns:
            dict: the rasterised image, the target trajectory (position and yaw) along with their availability,
            the 2D matrix to center that agent, the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        try:
            if self.cfg["raster_params"]["disable_traffic_light_faces"]:
                tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        except KeyError:
            warnings.warn(
                "disable_traffic_light_faces not found in config, this will raise an error in the future",
                RuntimeWarning,
                stacklevel=2,
            )
        data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        graph = data["image"]

        target_positions = np.array(data["target_positions"], dtype=np.float32)
        target_yaws = np.array(data["target_yaws"], dtype=np.float32)

        history_positions = np.array(data["history_positions"], dtype=np.float32)
        history_yaws = np.array(data["history_yaws"], dtype=np.float32)

        timestamp = frames[state_index]["timestamp"]
        track_id = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch

        return {
            "graph": graph,
            "target_positions": target_positions,
            "target_yaws": target_yaws,
            "target_availabilities": data["target_availabilities"],
            "history_positions": history_positions,
            "history_yaws": history_yaws,
            "history_availabilities": data["history_availabilities"],
            "raster_from_world": data["raster_from_world"],
            "raster_from_agent": data["raster_from_agent"],
            "agent_from_world": data["agent_from_world"],
            "world_from_agent": data["world_from_agent"],
            "track_id": track_id,
            "timestamp": timestamp,
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "extent": data["extent"],
        }


def process_one_instance(workload, graph_builder, processed_dir, split="val"):
    idx, data_point = workload
    output_file = to_filename(idx, split)
    data = graph_builder.to_data(scene_info=data_point, k=3)
    output_path = osp.join(processed_dir, output_file)
    torch.save(data, output_path)


def process_data_points(dataset, graph_builder, processed_dir, split, worker_count: Optional[int] = None):
    worker_count = worker_count or (cpu_count() - 1)
    print(f"Start processing with {worker_count} workers")
    _process_one_instance = partial(process_one_instance,
                                    graph_builder=graph_builder,
                                    processed_dir=processed_dir,
                                    split=split)
    with Pool(worker_count) as p:
        result = tqdm(p.imap(_process_one_instance, enumerate(dataset)),
                      total=len(dataset))
        _ = list(result)


def to_filename(idx: int, split: str) -> str:
    return f"{split}_graph_{idx:09d}.pt"


class LyftGraphDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 config_file: str,
                 split: str = "val",
                 transform=None,
                 pre_transform=None):
        self.split = split
        self.cfg = load_config_data(config_file)

        dm = LocalDataManager()
        data_loader_conf = self.cfg.get(f"{self.split}_data_loader")
        dataset_path = dm.require(data_loader_conf.get("key"))

        print("dataset path", dataset_path)
        zarr_dataset = ChunkedDataset(dataset_path)
        zarr_dataset.open()

        rasterizer = build_rasterizer(cfg=self.cfg, data_manager=dm)
        self.dataset = AgentGraphDataset(cfg=self.cfg,
                                         zarr_dataset=zarr_dataset,
                                         rasterizer=rasterizer)
        self.graph_builder = SceneGraphBuilder(raster_size=self.cfg["raster_params"]["raster_size"])

        super(LyftGraphDataset, self).__init__(data_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["meta.json"]

    @property
    def processed_file_names(self):
        return [to_filename(i, self.split) for i in range(len(self.dataset))]

    def download(self):
        print("NO action! Please manually download the raw dataset.")

    def process(self):
        process_data_points(self.dataset,
                            graph_builder=SceneGraphBuilder(raster_size=self.cfg["raster_params"]["raster_size"]),
                            processed_dir=str(self.processed_dir),
                            split=str(self.split))
        print("Done!")

    def len(self):
        return len(glob.glob(osp.join(self.processed_dir, "*.pt")))

    def get(self, idx: int):
        data = torch.load(osp.join(self.processed_dir, to_filename(idx=idx, split=self.split)))
        return data
