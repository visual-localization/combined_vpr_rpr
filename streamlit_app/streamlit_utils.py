from io import BytesIO
from base64 import b64decode
from data_models import ImageDTOS, QueryResponseDTOS
from scipy.spatial.transform import Rotation
from bokeh.plotting import figure
from bokeh.models import Arrow, VeeHead, ColumnDataSource, HoverTool
from bokeh.tile_providers import get_provider
import xyzservices.providers as xyz
import utm
import math
import numpy as np
import re


def parse_image_bytes(image_dtos: ImageDTOS):
    base64_image = image_dtos.data
    image = b64decode(base64_image)
    return BytesIO(image)


def quat_to_euler(w: float, x: float, y: float, z: float):
    quat = np.array([x, y, z, w])

    if np.linalg.norm(quat) != 1:
        quat /= np.linalg.norm(quat)

    r = Rotation.from_quat(quat)
    return r.as_euler("zyx")


def quat_to_heading(w: float, x: float, y: float, z: float) -> float:
    yaw = math.atan2(2.0 * (z * w + x * y), -1.0 + 2.0 * (w * w + x * x))
    return yaw


def pittsburgh_name_to_yaw(name: str) -> float:
    m = re.search("_yaw(\d+)", name)
    yaw_type = int(m.group(1))
    yaw = math.radians((yaw_type % 12) * 30)
    return yaw


def pittsburgh_utm_to_latlon(easting: float, northing: float) -> tuple[float, float]:
    return utm.to_latlon(easting, northing, 17, "T")


def latlon_to_mercator(lat: float, lon: float) -> tuple[float, float]:
    r_major = 6378137.000
    y = r_major * math.radians(lon)
    scale = y / lon
    x = (
        180.0
        / math.pi
        * math.log(math.tan(math.pi / 4.0 + lat * (math.pi / 180.0) / 2.0))
        * scale
    )

    return y, x


def pittsburgh_utm_to_mercator(easting: float, northing: float) -> tuple[float, float]:
    lat, lon = pittsburgh_utm_to_latlon(easting, northing)
    return latlon_to_mercator(lat, lon)


def scatter_coords(
    fig: figure,
    coords: list[tuple[float, float]],
    headings: list[float] | None,
    descs: list[str],
    color: str,
    arrow_length: float,
):
    source = ColumnDataSource(
        data=dict(
            x=[coord[0] for coord in coords],
            y=[coord[1] for coord in coords],
            desc=descs,
        )
    )

    arrow_head = VeeHead(
        size=25, fill_color=color, line_cap="round", line_color="black", line_width=2
    )

    fig.scatter(
        "x",
        "y",
        source=source,
        size=20,
        fill_color=color,
        line_color="black",
        line_width=2,
        marker="circle",
    )

    if headings is not None:
        for coord, heading in zip(coords, headings):
            start = coord
            end = (
                start[0] + arrow_length * math.cos(heading),
                start[1] + arrow_length * math.sin(heading),
            )

            arrow = Arrow(
                end=arrow_head,
                x_start=start[0],
                y_start=start[1],
                x_end=end[0],
                y_end=end[1],
                line_color="black",
                line_width=2,
                line_dash=[10, 10],
            )

            fig.add_layout(arrow)

    return fig


def generate_bokeh_figure(query_response: QueryResponseDTOS, add_arrows=False):
    ground_truth_coords = pittsburgh_utm_to_mercator(
        query_response.query.translation[0], query_response.query.translation[2]
    )

    inferred_coords = pittsburgh_utm_to_mercator(
        query_response.pose.t[0], query_response.pose.t[2]
    )

    retrieved_coordss = [
        pittsburgh_utm_to_mercator(scene.translation[0], scene.translation[2])
        for scene in query_response.retrieved_scenes
    ]

    if add_arrows:
        ground_truth_heading = quat_to_heading(
            query_response.query.rotation[0],
            query_response.query.rotation[1],
            query_response.query.rotation[2],
            query_response.query.rotation[3],
        )

        inferred_heading = quat_to_heading(
            query_response.pose.r[0],
            query_response.pose.r[1],
            query_response.pose.r[2],
            query_response.pose.r[3],
        )

        retrieved_headings = [
            quat_to_heading(
                scene.rotation[0],
                scene.rotation[1],
                scene.rotation[2],
                scene.rotation[3],
            )
            for scene in query_response.retrieved_scenes
        ]

    hover = HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(x,y)", "(@x, @y)"),
            ("desc", "@desc"),
        ]
    )

    x_range = (
        min([coord[0] for coord in retrieved_coordss + [ground_truth_coords]]) - 10000,
        max([coord[0] for coord in retrieved_coordss + [ground_truth_coords]]) + 10000,
    )

    y_range = (
        min([coord[1] for coord in retrieved_coordss + [ground_truth_coords]]) - 10000,
        max([coord[1] for coord in retrieved_coordss + [ground_truth_coords]]) + 10000,
    )

    fig = figure(
        title="Query and retrieved images",
        x_range=x_range,
        y_range=y_range,
        x_axis_type="mercator",
        y_axis_type="mercator",
    )

    fig.add_tools(hover)

    tile_provider = get_provider(xyz.CartoDB.Positron)
    fig.add_tile(tile_provider)

    scatter_coords(
        fig,
        retrieved_coordss,
        retrieved_headings if add_arrows and retrieved_headings else None,
        [f"retrieved_{scene.name}" for scene in query_response.retrieved_scenes],
        "white",
        10,
    )

    scatter_coords(
        fig,
        [ground_truth_coords],
        [ground_truth_heading] if add_arrows and ground_truth_heading else None,
        [f"ground_truth_{query_response.query.name}"],
        "green",
        10,
    )

    scatter_coords(
        fig,
        [inferred_coords],
        [inferred_heading] if add_arrows and inferred_heading else None,
        ["inferred"],
        "red",
        10,
    )

    return fig
