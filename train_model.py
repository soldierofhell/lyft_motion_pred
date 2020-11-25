from torch_geometric.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from libs.dataset import LyftGraphDataset
from libs.model.gnn import GNN


##### CONFIG ######
L5KIT_DATA_FOLDER = "./dataset"
L5KIT_CONFIG_FILE = "./config/training_dataset_config.yaml"
batch_size = 16
max_epochs = 20
save_top_k = 5
##################


def get_model():
    ego_rf = 25 - 2
    agent_rf = 25 - 2
    lane_rf = 63 - 2
    stop_sign_rf = 2 - 2
    speed_hump_rf = 2 - 2
    speed_bump_rf = 2 - 2
    crosswalk_rf = 2 - 2

    node_dim = 128
    node_type = 7
    positional_emb = 32

    model_config = {
        "embedding_dim": positional_emb,
        "rasterize_size": [244, 244],
        "crosswalk_in_dim": 2 * positional_emb + crosswalk_rf,
        "crosswalk_out_dim": node_dim,
        "speed_bump_in_dim": 2 * positional_emb + speed_bump_rf,
        "speed_bump_out_dim": node_dim,
        "speed_hump_in_dim": 2 * positional_emb + speed_hump_rf,
        "speed_hump_out_dim": node_dim,
        "lane_in_dim": 2 * positional_emb + lane_rf,
        "lane_out_dim": node_dim,
        "ego_in_dim": 2 * positional_emb + ego_rf,
        "ego_out_dim": node_dim,
        "stop_sign_in_dim": 2 * positional_emb + stop_sign_rf,
        "stop_sign_out_dim": node_dim,
        "agent_in_dim": 2 * positional_emb + agent_rf,
        "agent_out_dim": node_dim,
        "scene_in_dim": node_type + node_dim,
        "scene_out_dim": [100, (node_type + node_dim) // 2],
        "mode_count": 3,
        "future_len": 50,
        "history_len": 10
    }

    model = GNN(**model_config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total param: {total_params:,}")

    return model


def main():
    lyft_graph_dataset = LyftGraphDataset(data_dir=L5KIT_DATA_FOLDER,
                                          config_file=L5KIT_CONFIG_FILE,
                                          split="train",
                                          transform=None,
                                          pre_transform=None)

    train_loader = DataLoader(lyft_graph_dataset, batch_size=batch_size, shuffle=True)
    model = get_model()

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='./logs',
        save_top_k=save_top_k,
        mode='min')

    logger = TensorBoardLogger(save_dir="./tensorboard_logs", name='lyft_motion_pred')

    trainer = pl.Trainer(max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         logger=logger)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
