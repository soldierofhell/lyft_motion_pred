import os
from typing import *

from libs.rasterizer import build_rasterizer, SemGraphRasterizer
from libs.dataset import AgentGraphDataset

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.configs import load_config_data

os.environ["L5KIT_DATA_FOLDER"] = "/Users/phiradet/workspace/motion_prediction/lyft-motion-prediction-autonomous-vehicles"
config_file = "/Users/phiradet/workspace/motion_prediction/l5kit/examples/visualisation/visualisation_config.yaml"


def test_graph_rasterizer_no_error():
    # contain traffic light
    index = 150

    cfg = load_config_data(config_file)
    cfg["raster_params"]["map_type"] = "semantic_graph"

    data_loader_conf = cfg.get(f"val_data_loader")
    dm = LocalDataManager()
    dataset_path = dm.require(data_loader_conf.get("key"))

    zarr_dataset = ChunkedDataset(dataset_path)
    zarr_dataset.open()

    rasterizer = build_rasterizer(cfg=cfg, data_manager=dm)
    dataset = AgentGraphDataset(cfg=cfg,
                                zarr_dataset=zarr_dataset,
                                rasterizer=rasterizer)
    data_point = dataset[index]

    assert "graph" in data_point
    assert "lanes" in data_point["graph"]
    assert isinstance(data_point["graph"]["lanes"], List)

    print()
    print(data_point.keys())
    element_types = SemGraphRasterizer.keys
    for e in element_types:
        print(f"---- {e} ----")
        if len(data_point["graph"][e]) > 0:
            print(data_point["graph"][e][0])
