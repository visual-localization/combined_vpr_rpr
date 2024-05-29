from typing import Callable, Literal
import streamlit as st
from pathlib import Path
import modal
from streamlit_image_select import image_select

from data_models import (
    ImageDTOS,
    QueryResponseDTOS,
)

from streamlit_utils import generate_bokeh_figure, parse_image_bytes

SAMPLE_DIR = Path(__file__).parent / "samples"
with open((SAMPLE_DIR / "scene_names.txt"), "r") as f:
    scene_names = f.read().splitlines()
sample_image_paths = [
    str(SAMPLE_DIR / f"{index}.jpg") for index in range(len(scene_names))
]

ASSET_DIR = Path(__file__).parent / "assets"

sample_query_dataset_modal_fn = modal.Function.lookup(
    "EssentialMixer", "MixVprEssMatRprModel.sample_query_dataset"
)

query_modal_fn = modal.Function.lookup("EssentialMixer", "MixVprEssMatRprModel.query")

get_images_modal_fn = modal.Function.lookup(
    "EssentialMixer", "MixVprEssMatRprModel.get_images"
)

sample_query_dataset: Callable[[int, int], list[dict]] = (
    lambda a, b: sample_query_dataset_modal_fn.remote(a, b)
)

query: Callable[[str, int, int, Literal["max", "weighted"]], dict] = (
    lambda a, b, c, d: query_modal_fn.remote(a, b, c, d)
)

get_images: Callable[[list[tuple[Literal["database", "query"], str]]], list[dict]] = (
    lambda a: get_images_modal_fn.remote(a)
)

st.set_page_config(
    page_title="EssentialMixer",
    page_icon="📌",
    layout="centered",
)

st.title("EssentialMixer")

st.image(str(ASSET_DIR / "model.svg"), use_column_width=True)

st.header("Input")

with st.form("pipeline_input", border=False):
    with st.container(height=520):
        selected_sample_index = image_select(
            "Query samples",
            images=sample_image_paths,
            captions=[Path(scene_name).name for scene_name in scene_names],
            index=0,
            return_value="index",
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        num_retrievals = st.number_input(
            "Retrieve top K images", value=10, min_value=2, max_value=100
        )

    with col2:
        num_rerank = st.number_input(
            "Rerank top k images", value=5, min_value=2, max_value=100
        )

    with col3:
        query_type = st.radio("Query type", ["max", "weighted"], index=0)

    if num_rerank > num_retrievals:
        st.warning(
            "Number of reranked images should be less than or equal to number of retrieved images."
        )

    st.form_submit_button("Query")

st.header("Output")

with st.spinner("Querying. This might take up to a minute on a cold-start..."):
    query_response = query(
        scene_names[selected_sample_index], num_retrievals, num_rerank, query_type
    )

    query_response_dto = QueryResponseDTOS.from_dict(query_response)
    fig = generate_bokeh_figure(query_response_dto)

    st.bokeh_chart(fig, use_container_width=True)

with st.expander("Step-by-step breakdown"):
    with st.spinner("Loading images..."):
        images_response = get_images(
            [
                ("query", query_response_dto.query.name),
                *[
                    ("database", scene.name)
                    for scene in query_response_dto.retrieved_scenes
                ],
            ]
        )

        images = [
            parse_image_bytes(ImageDTOS.from_dict(image)) for image in images_response
        ]

        query_image = images[0]
        retrieved_images = images[1:]
        reranked_images = [
            retrieved_images[i] for i in query_response_dto.reranking_indices
        ]

        retrieved_captions = [
            scene.name for scene in query_response_dto.retrieved_scenes
        ]

        reranked_captions = [
            retrieved_captions[i] for i in query_response_dto.reranking_indices
        ]

        st.subheader("Query")
        st.image(
            query_image,
            caption=query_response_dto.query.name,
            use_column_width=True,
        )

        st.subheader("Retrieved images")
        with st.container(height=520):
            st.image(
                retrieved_images,
                caption=retrieved_captions,
                use_column_width=True,
            )

        st.subheader("Reranked images")
        with st.container(height=520):
            st.image(
                reranked_images,
                caption=reranked_captions,
                use_column_width=True,
            )
