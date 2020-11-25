from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pytorch_lightning as pl

from libs.rasterizer import SemGraphRasterizer
from libs.scene_graph import SceneGraphData, get_mask
from libs.model.loss import neg_multi_log_likelihood_batch


class PositionalEmbedding(nn.Module):

    def __init__(self, raster_size: Tuple[int, int], embedding_dim: int, scale: int = 100):
        super(PositionalEmbedding, self).__init__()

        self.raster_size = raster_size
        self.scale = scale

        self.x_embedding = nn.Embedding(num_embeddings=raster_size[0] * scale,
                                        embedding_dim=embedding_dim)
        self.y_embedding = nn.Embedding(num_embeddings=raster_size[1] * scale,
                                        embedding_dim=embedding_dim)

    def forward(self, xy_pos: torch.FloatTensor):
        # xy_pos: (batch_size, 2)
        x_scaled_pos = torch.round(xy_pos[:, 0] * self.scale).long()
        y_scaled_pos = torch.round(xy_pos[:, 1] * self.scale).long()

        # (batch_size, embedding_dim)
        x_embedding = self.x_embedding(x_scaled_pos)
        y_embedding = self.y_embedding(y_scaled_pos)

        # (batch_size, 2*embedding_dim)
        return torch.cat([x_embedding, y_embedding], dim=1)


class MultiLayerGCN(nn.Module):

    def __init__(self, in_channels: int, out_channels: Union[int, List[int]]):
        super(MultiLayerGCN, self).__init__()

        if isinstance(out_channels, int):
            out_channels = [out_channels]

        self.conv_layers = []
        for o_dim in out_channels:
            _conv = GCNConv(in_channels=in_channels,
                            out_channels=o_dim)
            self.conv_layers.append(_conv)
            in_channels = o_dim
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

        return x


class GNN(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()

        self.kwargs = kwargs

        # positional embedding
        embedding_dim = kwargs.get("embedding_dim")
        rasterer_size = kwargs.get("rasterize_size")
        self.positional_embedding = PositionalEmbedding(raster_size=rasterer_size,
                                                        embedding_dim=embedding_dim,
                                                        scale=100)

        # polyline GCNConv
        self.crosswalk_conv = MultiLayerGCN(in_channels=kwargs.get("crosswalk_in_dim"),
                                            out_channels=kwargs.get("crosswalk_out_dim"))
        self.speed_bump_conv = MultiLayerGCN(in_channels=kwargs.get("speed_bump_in_dim"),
                                             out_channels=kwargs.get("speed_bump_out_dim"))
        self.speed_hump_conv = MultiLayerGCN(in_channels=kwargs.get("speed_hump_in_dim"),
                                             out_channels=kwargs.get("speed_hump_out_dim"))
        self.lane_conv = MultiLayerGCN(in_channels=kwargs.get("lane_in_dim"),
                                       out_channels=kwargs.get("lane_out_dim"))
        self.ego_conv = MultiLayerGCN(in_channels=kwargs.get("ego_in_dim"),
                                      out_channels=kwargs.get("ego_out_dim"))

        # Single point linear
        self.stop_sign_linear = nn.Sequential(
            nn.Linear(in_features=kwargs.get("stop_sign_in_dim"), out_features=kwargs.get("stop_sign_out_dim")),
            nn.ReLU(),
            nn.Dropout()
        )
        self.agent_linear = nn.Sequential(
            nn.Linear(in_features=kwargs.get("agent_in_dim"), out_features=kwargs.get("agent_out_dim")),
            nn.ReLU(),
            nn.Dropout()
        )

        # Scene GCN
        scene_out_dim = kwargs.get("scene_out_dim")
        if isinstance(scene_out_dim, int):
            scene_out_dim = [scene_out_dim]
        self.scene_conv = MultiLayerGCN(in_channels=kwargs.get("scene_in_dim"),
                                        out_channels=scene_out_dim)

        # Output linear
        mode_count = kwargs.get("mode_count")
        future_len = kwargs.get("future_len")

        future_pos = 2 * future_len  # 2D coordinate
        out_features = (future_pos * mode_count) + mode_count  # future_pos * mode + conf_for_each_mode
        self.logit = nn.Linear(scene_out_dim[-1], out_features=out_features)

        self.criterion = neg_multi_log_likelihood_batch

    def forward(self, batch: SceneGraphData):
        x = batch.x
        batch_size, _ = x.shape

        node_type_count = 7
        node_type_mat = x[:, :node_type_count]
        xy_pos = x[:, node_type_count:node_type_count + 2]
        node_raw_feature = x[:, node_type_count + 2:]

        crosswalk_mask = get_mask(node_type_mat, SemGraphRasterizer.CROSSWALKS)
        speed_bump_mask = get_mask(node_type_mat, SemGraphRasterizer.SPEED_BUMPS)
        speed_hump_mask = get_mask(node_type_mat, SemGraphRasterizer.SPEED_HUMPS)
        stop_sign_mask = get_mask(node_type_mat, SemGraphRasterizer.STOP_SIGNS)
        lane_mask = get_mask(node_type_mat, SemGraphRasterizer.LANES)
        agent_mask = get_mask(node_type_mat, SemGraphRasterizer.AGENTS)
        ego_mask = get_mask(node_type_mat, SemGraphRasterizer.EGO)

        # (batch_size, 2*emb_dim)
        positional_emb = self.positional_embedding(xy_pos)

        # x: (batch_size, 2*emb_dim + node_raw_feature_dim)
        x = torch.cat([positional_emb, node_raw_feature], dim=1)

        # polyline features
        # ego (25), agent (25), lane (63), stop_sign (2), speed_hump (2), speed_bump (2), crosswalk (2)
        pos_emb_dim = 2 * self.kwargs.get("embedding_dim")

        crosswalk_feature = self.crosswalk_conv(x[crosswalk_mask, :pos_emb_dim],
                                                batch.crosswalk_edge_index)
        speed_bump_feature = self.speed_bump_conv(x[speed_bump_mask, :pos_emb_dim],
                                                  batch.speed_bump_edge_index)
        speed_hump_feature = self.speed_hump_conv(x[speed_hump_mask, :pos_emb_dim],
                                                  batch.speed_hump_edge_index)
        lane_feature = self.lane_conv(x[lane_mask, :pos_emb_dim+63-2], batch.lane_edge_index)
        ego_feature = self.ego_conv(x[ego_mask, :pos_emb_dim+25-2], batch.ego_edge_index)

        # point features
        stop_sign_feature = self.stop_sign_linear(x[stop_sign_mask, :pos_emb_dim])
        agent_feature = self.agent_linear(x[agent_mask, :pos_emb_dim+25-2])

        x = torch.zeros(batch_size, self.kwargs.get("scene_in_dim")-7, requires_grad=True)

        # Is this ok?
        with torch.no_grad():
            x[crosswalk_mask] = crosswalk_feature
            x[speed_bump_mask] = speed_bump_feature
            x[speed_hump_mask] = speed_hump_feature
            x[stop_sign_mask] = stop_sign_feature
            x[lane_mask] = lane_feature
            x[agent_mask] = agent_feature
            x[ego_mask] = ego_feature
        x = torch.cat([node_type_mat, x], dim=1)

        scene_feature = self.scene_conv(x, batch.edge_index)
        ego_feature = scene_feature[ego_mask]

        history_len = self.kwargs.get("history_len")
        last_ind = [i * history_len for i in range(batch.num_graphs)]
        last_frame_ego_feature = ego_feature[last_ind, :]

        mode_count = self.kwargs.get("mode_count")

        # (num_scene, mode*future_pos + mode)
        pred = self.logit(last_frame_ego_feature)

        # (num_scene, mode)
        conf = pred[:, -mode_count:]
        conf = torch.softmax(conf, dim=1)

        # (num_scene, mode, time, 2)
        future_len = self.kwargs.get("future_len")
        pred_pos = pred[:, :-mode_count].view(batch.num_graphs, mode_count, future_len, 2)

        return pred_pos, conf

    def training_step(self, batch, batch_idx):
        actual_pos = batch.y.view(batch.num_graphs, -1, 2)
        pred_pos, conf = self.forward(batch)
        availabilities = batch.availabilities.view(batch.num_graphs, -1)
        loss = self.criterion(actual=actual_pos,
                              pred=pred_pos,
                              confidences=conf,
                              avails=availabilities)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
