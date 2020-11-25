import matplotlib.pyplot as plt
from l5kit.geometry import transform_points
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR


def plot_image(data_point, rasterizer):
    im = data_point["image"].transpose(1, 2, 0)
    im = rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data_point["target_positions"],
                                               data_point["raster_from_agent"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data_point["target_yaws"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(im[::-1])
