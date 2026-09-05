
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Banana Ripeness Assessment",
    page_icon="🍌",
    layout="wide",
)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = Path(__file__).with_name(
    "eugene_watershed_grabcut_svm.joblib"
)

DEFAULT_LABELS = [
    "unripe",
    "ripe",
    "overripe",
    "rotten",
]

DEFAULT_SVM_IMAGE_SIZE = (64, 64)
DEFAULT_PROCESSING_SIZE = (256, 256)
DEFAULT_GAUSSIAN_KERNEL = (5, 5)
DEFAULT_GAUSSIAN_SIGMA = 0

DEFAULT_WATERSHED_BORDER_RATIO = 0.06
DEFAULT_WATERSHED_CENTER_RADIUS_X = 0.43
DEFAULT_WATERSHED_CENTER_RADIUS_Y = 0.43
DEFAULT_WATERSHED_LOW_GRADIENT_PERCENTILE = 35
DEFAULT_WATERSHED_MIN_SEED_AREA_RATIO = 0.001
DEFAULT_WATERSHED_MAX_FOREGROUND_SEEDS = 8

DEFAULT_GRABCUT_ITERATIONS = 5
DEFAULT_GRABCUT_SURE_FOREGROUND_RATIO = 0.50
DEFAULT_GRABCUT_BORDER_RATIO = 0.04
DEFAULT_GRABCUT_BBOX_MARGIN_RATIO = 0.08
DEFAULT_GRABCUT_MIN_COMPONENT_AREA_RATIO = 0.001

DEFAULT_BANANA_MIN_SATURATION = 25
DEFAULT_BANANA_MIN_CHROMA = 10.0
DEFAULT_BANANA_DARK_VALUE_MAX = 135
DEFAULT_BANANA_DARK_MIN_SATURATION = 15
DEFAULT_BANANA_DARK_MIN_CHROMA = 7.0
DEFAULT_BANANA_HUE_MAX = 95
DEFAULT_BANANA_MIN_COMPONENT_AREA_RATIO = 0.001
DEFAULT_BANANA_MAX_HOLE_AREA_RATIO = 0.004

REGION_FEATURE_NAMES = [
    "foreground_area_ratio",
    "component_count",
    "largest_component_ratio",
    "perimeter_normalised",
    "bbox_aspect_ratio",
    "extent",
    "solidity",
    "circularity",
]


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None

    loaded = joblib.load(MODEL_PATH)

    if isinstance(loaded, dict) and "model" in loaded:
        return loaded

    return {
        "model": loaded,
        "classes": DEFAULT_LABELS,
        "svm_image_size": DEFAULT_SVM_IMAGE_SIZE,
        "processing_size": DEFAULT_PROCESSING_SIZE,
        "gaussian_kernel": DEFAULT_GAUSSIAN_KERNEL,
        "gaussian_sigma": DEFAULT_GAUSSIAN_SIGMA,
    }


# ============================================================
# IMAGE INPUT
# ============================================================

def decode_uploaded_image(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8,
    )

    image_bgr = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            "The uploaded file could not be decoded as an image."
        )

    return image_bgr


def preprocess_rgb_from_bgr(
    image_bgr,
    processing_size,
    gaussian_kernel,
    gaussian_sigma,
):
    original_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    resized_rgb = cv2.resize(
        original_rgb,
        tuple(processing_size),
        interpolation=cv2.INTER_AREA,
    )

    gaussian_rgb = cv2.GaussianBlur(
        resized_rgb,
        tuple(gaussian_kernel),
        gaussian_sigma,
    )

    return (
        original_rgb,
        resized_rgb,
        gaussian_rgb,
    )


# ============================================================
# MARKER-CONTROLLED WATERSHED
# ============================================================

def sobel_gradient(gray):
    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(gx, gy)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return np.clip(
        magnitude * 255.0,
        0,
        255,
    ).astype(np.uint8)


def create_watershed_markers(
    gray,
    gradient,
    border_ratio,
    center_radius_x,
    center_radius_y,
    low_gradient_percentile,
    min_seed_area_ratio,
    max_foreground_seeds,
):
    h, w = gray.shape

    border_y = max(
        1,
        int(round(h * border_ratio)),
    )

    border_x = max(
        1,
        int(round(w * border_ratio)),
    )

    background_seed = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    background_seed[:border_y, :] = 255
    background_seed[-border_y:, :] = 255
    background_seed[:, :border_x] = 255
    background_seed[:, -border_x:] = 255

    yy, xx = np.mgrid[0:h, 0:w]

    centre_x = w / 2.0
    centre_y = h / 2.0

    centre_ellipse = (
        (
            (xx - centre_x)
            / max(center_radius_x * w, 1)
        ) ** 2
        +
        (
            (yy - centre_y)
            / max(center_radius_y * h, 1)
        ) ** 2
        <= 1.0
    )

    threshold = np.percentile(
        gradient[centre_ellipse],
        low_gradient_percentile,
    )

    candidate_seed = (
        (gradient <= threshold)
        & centre_ellipse
    ).astype(np.uint8)

    count, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            candidate_seed,
            connectivity=8,
        )
    )

    min_area = max(
        10,
        int(
            round(
                h
                * w
                * min_seed_area_ratio
            )
        ),
    )

    ranked_components = []

    for component_id in range(1, count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < min_area:
            continue

        cx, cy = centroids[component_id]

        distance = np.hypot(
            (cx - centre_x) / w,
            (cy - centre_y) / h,
        )

        score = area / (
            1.0 + 2.0 * distance
        )

        ranked_components.append(
            (score, component_id)
        )

    ranked_components = sorted(
        ranked_components,
        reverse=True,
    )[:int(max_foreground_seeds)]

    foreground_seed = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    for _, component_id in ranked_components:
        foreground_seed[
            component_labels == component_id
        ] = 255

    if np.count_nonzero(foreground_seed) == 0:
        cv2.ellipse(
            foreground_seed,
            (
                int(round(centre_x)),
                int(round(centre_y)),
            ),
            (
                max(3, int(round(w * 0.08))),
                max(3, int(round(h * 0.08))),
            ),
            0,
            0,
            360,
            255,
            -1,
        )

    markers = np.zeros(
        (h, w),
        dtype=np.int32,
    )

    markers[
        background_seed > 0
    ] = 1

    foreground_count, foreground_labels = (
        cv2.connectedComponents(
            (foreground_seed > 0).astype(np.uint8),
            connectivity=8,
        )
    )

    for foreground_id in range(
        1,
        foreground_count,
    ):
        markers[
            foreground_labels == foreground_id
        ] = foreground_id + 1

    return (
        markers,
        background_seed,
        foreground_seed,
    )


def run_marker_controlled_watershed(
    preprocessed_rgb,
    bundle,
):
    gray = cv2.cvtColor(
        preprocessed_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    gradient = sobel_gradient(gray)

    (
        markers,
        background_seed,
        foreground_seed,
    ) = create_watershed_markers(
        gray=gray,
        gradient=gradient,
        border_ratio=float(
            bundle.get(
                "watershed_border_ratio",
                DEFAULT_WATERSHED_BORDER_RATIO,
            )
        ),
        center_radius_x=float(
            bundle.get(
                "watershed_center_radius_x",
                DEFAULT_WATERSHED_CENTER_RADIUS_X,
            )
        ),
        center_radius_y=float(
            bundle.get(
                "watershed_center_radius_y",
                DEFAULT_WATERSHED_CENTER_RADIUS_Y,
            )
        ),
        low_gradient_percentile=float(
            bundle.get(
                "watershed_low_gradient_percentile",
                DEFAULT_WATERSHED_LOW_GRADIENT_PERCENTILE,
            )
        ),
        min_seed_area_ratio=float(
            bundle.get(
                "watershed_min_seed_area_ratio",
                DEFAULT_WATERSHED_MIN_SEED_AREA_RATIO,
            )
        ),
        max_foreground_seeds=int(
            bundle.get(
                "watershed_max_foreground_seeds",
                DEFAULT_WATERSHED_MAX_FOREGROUND_SEEDS,
            )
        ),
    )

    watershed_input = cv2.cvtColor(
        gradient,
        cv2.COLOR_GRAY2BGR,
    )

    watershed_labels = markers.copy()

    cv2.watershed(
        watershed_input,
        watershed_labels,
    )

    final_mask = np.where(
        watershed_labels > 1,
        255,
        0,
    ).astype(np.uint8)

    colour_cutout = cv2.bitwise_and(
        preprocessed_rgb,
        preprocessed_rgb,
        mask=final_mask,
    )

    return {
        "gray": gray,
        "gradient": gradient,
        "markers": markers,
        "background_seed": background_seed,
        "foreground_seed": foreground_seed,
        "watershed_labels": watershed_labels,
        "final_mask": final_mask,
        "colour_cutout": colour_cutout,
    }


# ============================================================
# GRABCUT + BANANA-ONLY REFINEMENT
# ============================================================

def remove_tiny_components(
    mask,
    min_area_ratio,
):
    binary = (
        mask > 0
    ).astype(np.uint8)

    h, w = binary.shape

    min_area = max(
        5,
        int(
            round(
                h
                * w
                * min_area_ratio
            )
        ),
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    cleaned = np.zeros_like(binary)

    for component_id in range(1, count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= min_area:
            cleaned[
                labels == component_id
            ] = 1

    return (
        cleaned * 255
    ).astype(np.uint8)


def fill_small_holes(
    mask,
    max_hole_area_ratio,
):
    binary = (
        mask > 0
    ).astype(np.uint8)

    h, w = binary.shape

    max_hole_area = int(
        round(
            h
            * w
            * max_hole_area_ratio
        )
    )

    inverted = (
        1 - binary
    ).astype(np.uint8)

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            inverted,
            connectivity=8,
        )
    )

    output = binary.copy()

    for component_id in range(1, count):
        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        component = (
            labels == component_id
        )

        touches_border = (
            component[0, :].any()
            or component[-1, :].any()
            or component[:, 0].any()
            or component[:, -1].any()
        )

        if (
            not touches_border
            and area <= max_hole_area
        ):
            output[
                component
            ] = 1

    return (
        output * 255
    ).astype(np.uint8)


def banana_only_colour_mask(
    preprocessed_rgb,
    grabcut_mask,
    bundle,
):
    current_foreground = (
        grabcut_mask > 0
    )

    hsv = cv2.cvtColor(
        preprocessed_rgb,
        cv2.COLOR_RGB2HSV,
    )

    H, S, V = cv2.split(hsv)

    lab = cv2.cvtColor(
        preprocessed_rgb,
        cv2.COLOR_RGB2LAB,
    )

    a = (
        lab[..., 1].astype(np.float32)
        - 128.0
    )

    b = (
        lab[..., 2].astype(np.float32)
        - 128.0
    )

    chroma = np.sqrt(
        a ** 2 + b ** 2
    )

    min_saturation = float(
        bundle.get(
            "banana_min_saturation",
            DEFAULT_BANANA_MIN_SATURATION,
        )
    )

    min_chroma = float(
        bundle.get(
            "banana_min_chroma",
            DEFAULT_BANANA_MIN_CHROMA,
        )
    )

    dark_value_max = float(
        bundle.get(
            "banana_dark_value_max",
            DEFAULT_BANANA_DARK_VALUE_MAX,
        )
    )

    dark_min_saturation = float(
        bundle.get(
            "banana_dark_min_saturation",
            DEFAULT_BANANA_DARK_MIN_SATURATION,
        )
    )

    dark_min_chroma = float(
        bundle.get(
            "banana_dark_min_chroma",
            DEFAULT_BANANA_DARK_MIN_CHROMA,
        )
    )

    hue_max = float(
        bundle.get(
            "banana_hue_max",
            DEFAULT_BANANA_HUE_MAX,
        )
    )

    min_component_area_ratio = float(
        bundle.get(
            "banana_min_component_area_ratio",
            DEFAULT_BANANA_MIN_COMPONENT_AREA_RATIO,
        )
    )

    max_hole_area_ratio = float(
        bundle.get(
            "banana_max_hole_area_ratio",
            DEFAULT_BANANA_MAX_HOLE_AREA_RATIO,
        )
    )

    coloured_banana = (
        (H <= hue_max)
        &
        (
            (S >= min_saturation)
            |
            (chroma >= min_chroma)
        )
        &
        (V >= 25)
    )

    dark_banana = (
        (V <= dark_value_max)
        &
        (
            (S >= dark_min_saturation)
            |
            (chroma >= dark_min_chroma)
        )
    )

    banana_candidate = (
        coloured_banana
        |
        dark_banana
    )

    banana_mask = (
        current_foreground
        &
        banana_candidate
    ).astype(np.uint8) * 255

    banana_mask = remove_tiny_components(
        banana_mask,
        min_area_ratio=min_component_area_ratio,
    )

    banana_mask = fill_small_holes(
        banana_mask,
        max_hole_area_ratio=max_hole_area_ratio,
    )

    return banana_mask


def watershed_guided_grabcut(
    preprocessed_rgb,
    watershed_result,
    bundle,
):
    initial_mask = (
        watershed_result[
            "final_mask"
        ]
        .copy()
        .astype(np.uint8)
    )

    initial_binary = (
        initial_mask > 0
    ).astype(np.uint8)

    h, w = initial_binary.shape

    if np.count_nonzero(
        initial_binary
    ) == 0:
        return {
            "grabcut_initialisation": np.zeros_like(
                initial_mask
            ),
            "raw_grabcut_mask": initial_mask,
            "refined_mask": initial_mask,
            "refined_colour_cutout": watershed_result[
                "colour_cutout"
            ].copy(),
        }

    iterations = int(
        bundle.get(
            "grabcut_iterations",
            DEFAULT_GRABCUT_ITERATIONS,
        )
    )

    sure_foreground_ratio = float(
        bundle.get(
            "grabcut_sure_foreground_ratio",
            DEFAULT_GRABCUT_SURE_FOREGROUND_RATIO,
        )
    )

    border_ratio = float(
        bundle.get(
            "grabcut_border_ratio",
            DEFAULT_GRABCUT_BORDER_RATIO,
        )
    )

    bbox_margin_ratio = float(
        bundle.get(
            "grabcut_bbox_margin_ratio",
            DEFAULT_GRABCUT_BBOX_MARGIN_RATIO,
        )
    )

    min_component_area_ratio = float(
        bundle.get(
            "grabcut_min_component_area_ratio",
            DEFAULT_GRABCUT_MIN_COMPONENT_AREA_RATIO,
        )
    )

    gc_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )

    # Entire Watershed foreground stays probable foreground.
    gc_mask[
        initial_binary > 0
    ] = cv2.GC_PR_FGD

    watershed_seed = (
        watershed_result[
            "foreground_seed"
        ] > 0
    ).astype(np.uint8)

    if np.count_nonzero(
        watershed_seed
    ) > 0:
        seed_distance = cv2.distanceTransform(
            watershed_seed,
            cv2.DIST_L2,
            5,
        )

        if float(
            seed_distance.max()
        ) > 0:
            sure_foreground = (
                seed_distance
                >=
                sure_foreground_ratio
                *
                float(
                    seed_distance.max()
                )
            )

            gc_mask[
                sure_foreground
            ] = cv2.GC_FGD

    background_seed = (
        watershed_result[
            "background_seed"
        ] > 0
    )

    gc_mask[
        background_seed
    ] = cv2.GC_BGD

    border_y = max(
        1,
        int(
            round(
                h * border_ratio
            )
        ),
    )

    border_x = max(
        1,
        int(
            round(
                w * border_ratio
            )
        ),
    )

    gc_mask[:border_y, :] = cv2.GC_BGD
    gc_mask[-border_y:, :] = cv2.GC_BGD
    gc_mask[:, :border_x] = cv2.GC_BGD
    gc_mask[:, -border_x:] = cv2.GC_BGD

    ys, xs = np.where(
        initial_binary > 0
    )

    if (
        len(xs) > 0
        and len(ys) > 0
    ):
        margin_x = int(
            round(
                w
                * bbox_margin_ratio
            )
        )

        margin_y = int(
            round(
                h
                * bbox_margin_ratio
            )
        )

        x0 = max(
            0,
            int(xs.min()) - margin_x,
        )

        x1 = min(
            w,
            int(xs.max()) + 1 + margin_x,
        )

        y0 = max(
            0,
            int(ys.min()) - margin_y,
        )

        y1 = min(
            h,
            int(ys.max()) + 1 + margin_y,
        )

        allowed_zone = np.zeros(
            (h, w),
            dtype=bool,
        )

        allowed_zone[
            y0:y1,
            x0:x1
        ] = True

        gc_mask[
            ~allowed_zone
        ] = cv2.GC_BGD

    if not np.any(
        gc_mask == cv2.GC_FGD
    ):
        seed_y, seed_x = np.where(
            watershed_seed > 0
        )

        if len(seed_x) > 0:
            middle = len(
                seed_x
            ) // 2

            gc_mask[
                seed_y[middle],
                seed_x[middle],
            ] = cv2.GC_FGD

        else:
            object_y, object_x = np.where(
                initial_binary > 0
            )

            middle = len(
                object_x
            ) // 2

            gc_mask[
                object_y[middle],
                object_x[middle],
            ] = cv2.GC_FGD

    image_bgr = cv2.cvtColor(
        preprocessed_rgb,
        cv2.COLOR_RGB2BGR,
    )

    background_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    foreground_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    refined_gc_mask = (
        gc_mask.copy()
    )

    try:
        cv2.grabCut(
            image_bgr,
            refined_gc_mask,
            None,
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )

        raw_grabcut_mask = np.where(
            np.isin(
                refined_gc_mask,
                [
                    cv2.GC_FGD,
                    cv2.GC_PR_FGD,
                ],
            ),
            255,
            0,
        ).astype(np.uint8)

    except cv2.error:
        raw_grabcut_mask = (
            initial_mask.copy()
        )

    raw_grabcut_mask = remove_tiny_components(
        raw_grabcut_mask,
        min_area_ratio=min_component_area_ratio,
    )

    initial_area = max(
        np.count_nonzero(
            initial_mask
        ),
        1,
    )

    raw_area = np.count_nonzero(
        raw_grabcut_mask
    )

    if (
        raw_area
        / initial_area
        < 0.20
    ):
        raw_grabcut_mask = (
            initial_mask.copy()
        )

    refined_mask = banana_only_colour_mask(
        preprocessed_rgb,
        raw_grabcut_mask,
        bundle,
    )

    if np.count_nonzero(
        refined_mask
    ) == 0:
        refined_mask = (
            raw_grabcut_mask.copy()
        )

    refined_colour_cutout = cv2.bitwise_and(
        preprocessed_rgb,
        preprocessed_rgb,
        mask=refined_mask,
    )

    return {
        "grabcut_initialisation": gc_mask,
        "raw_grabcut_mask": raw_grabcut_mask,
        "refined_mask": refined_mask,
        "refined_colour_cutout": refined_colour_cutout,
    }


# ============================================================
# FEATURES / PREDICTION
# ============================================================

def rgb_pixel_vector(
    rgb,
    svm_image_size,
):
    resized = cv2.resize(
        rgb,
        tuple(
            svm_image_size
        ),
        interpolation=cv2.INTER_AREA,
    )

    return (
        resized.astype(np.float32)
        / 255.0
    ).reshape(-1)


def extract_region_features(mask):
    binary = (
        mask > 0
    ).astype(np.uint8)

    h, w = binary.shape
    image_area = h * w

    foreground_pixels = int(
        np.count_nonzero(binary)
    )

    foreground_area_ratio = (
        foreground_pixels
        / image_area
    )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    component_count = max(
        count - 1,
        0,
    )

    component_areas = [
        int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )
        for component_id
        in range(
            1,
            count,
        )
    ]

    largest_component_ratio = (
        max(component_areas)
        / image_area
        if component_areas
        else 0.0
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return np.zeros(
            len(REGION_FEATURE_NAMES),
            dtype=np.float32,
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(contour)
    )

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    x, y, bw, bh = cv2.boundingRect(
        contour
    )

    hull_area = float(
        cv2.contourArea(
            cv2.convexHull(
                contour
            )
        )
    )

    perimeter_normalised = (
        perimeter
        /
        max(
            2.0 * (h + w),
            1.0,
        )
    )

    bbox_aspect_ratio = (
        bw / max(bh, 1)
    )

    extent = (
        area
        / max(
            bw * bh,
            1,
        )
    )

    solidity = (
        area
        / max(
            hull_area,
            1.0,
        )
    )

    circularity = (
        4.0
        * np.pi
        * area
        /
        max(
            perimeter ** 2,
            1.0,
        )
    )

    return np.array(
        [
            foreground_area_ratio,
            component_count,
            largest_component_ratio,
            perimeter_normalised,
            bbox_aspect_ratio,
            extent,
            solidity,
            circularity,
        ],
        dtype=np.float32,
    )


# ============================================================
# UI
# ============================================================

bundle = load_model_bundle()

st.title(
    "🍌 Banana Ripeness Assessment"
)

st.caption(
    "Marker-Controlled Watershed → GrabCut Refinement "
    "→ Banana-Only Colour Cleanup → RBF SVM"
)


if bundle is None:
    st.error(
        "Model file not found: "
        "`eugene_watershed_grabcut_svm.joblib`"
    )

    st.markdown(
        "Place `eugene_watershed_grabcut_svm.joblib` "
        "in the same GitHub/Streamlit folder as `app.py`."
    )

    st.stop()


model = bundle["model"]

svm_image_size = tuple(
    bundle.get(
        "svm_image_size",
        DEFAULT_SVM_IMAGE_SIZE,
    )
)

processing_size = tuple(
    bundle.get(
        "processing_size",
        DEFAULT_PROCESSING_SIZE,
    )
)

gaussian_kernel = tuple(
    bundle.get(
        "gaussian_kernel",
        DEFAULT_GAUSSIAN_KERNEL,
    )
)

gaussian_sigma = float(
    bundle.get(
        "gaussian_sigma",
        DEFAULT_GAUSSIAN_SIGMA,
    )
)


with st.sidebar:
    st.header("Model")

    st.write("Classes")
    for class_name in bundle.get(
        "classes",
        DEFAULT_LABELS,
    ):
        st.write(
            f"• {str(class_name).title()}"
        )

    st.divider()

    st.write(
        "**Main technique**"
    )

    st.code(
        "Marker-Controlled Watershed",
        language=None,
    )

    st.write(
        "**Supporting refinement**"
    )

    st.code(
        "GrabCut + Banana-Only Colour Cleanup",
        language=None,
    )

    st.write(
        "**Classifier**"
    )

    st.code(
        "RBF Support Vector Machine",
        language=None,
    )

    if "watershed_test_accuracy" in bundle:
        st.metric(
            "Watershed test accuracy",
            f"{float(bundle['watershed_test_accuracy']) * 100:.2f}%",
        )

    if "enhanced_test_accuracy" in bundle:
        st.metric(
            "GrabCut refined test accuracy",
            f"{float(bundle['enhanced_test_accuracy']) * 100:.2f}%",
        )


uploaded_file = st.file_uploader(
    "Upload a banana image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
)


if uploaded_file is None:
    st.info(
        "Upload a banana image to start."
    )
    st.stop()


try:
    image_bgr = decode_uploaded_image(
        uploaded_file
    )

    (
        original_rgb,
        resized_rgb,
        gaussian_rgb,
    ) = preprocess_rgb_from_bgr(
        image_bgr=image_bgr,
        processing_size=processing_size,
        gaussian_kernel=gaussian_kernel,
        gaussian_sigma=gaussian_sigma,
    )

    watershed = (
        run_marker_controlled_watershed(
            gaussian_rgb,
            bundle,
        )
    )

    grabcut = watershed_guided_grabcut(
        gaussian_rgb,
        watershed,
        bundle,
    )

    # IMPORTANT:
    # This joblib was trained on the final
    # Watershed + GrabCut refined colour cut-out.
    features = rgb_pixel_vector(
        grabcut[
            "refined_colour_cutout"
        ],
        svm_image_size,
    )

    expected_feature_length = bundle.get(
        "feature_length"
    )

    if (
        expected_feature_length
        is not None
        and len(features)
        != int(
            expected_feature_length
        )
    ):
        raise ValueError(
            "Feature length mismatch. "
            f"Expected {expected_feature_length}, "
            f"received {len(features)}."
        )

    feature_row = features.reshape(
        1,
        -1,
    )

    predicted_class = (
        model.predict(
            feature_row
        )[0]
    )

    probabilities = None
    confidence = None
    class_names = None

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(
                feature_row
            )[0]
        )

        if (
            hasattr(
                model,
                "named_steps",
            )
            and "svm"
            in model.named_steps
        ):
            class_names = list(
                model.named_steps[
                    "svm"
                ].classes_
            )

        elif hasattr(
            model,
            "classes_",
        ):
            class_names = list(
                model.classes_
            )

        if (
            class_names is not None
            and predicted_class in class_names
        ):
            predicted_index = (
                class_names.index(
                    predicted_class
                )
            )

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

    watershed_region_features = (
        extract_region_features(
            watershed[
                "final_mask"
            ]
        )
    )

    refined_region_features = (
        extract_region_features(
            grabcut[
                "refined_mask"
            ]
        )
    )

except Exception as exc:
    st.exception(exc)
    st.stop()


# ============================================================
# PREDICTION
# ============================================================

st.subheader(
    "Prediction"
)

left, right = st.columns(
    [1, 1]
)

with left:
    st.image(
        original_rgb,
        caption="Uploaded image",
        use_container_width=True,
    )

with right:
    st.metric(
        "Predicted ripeness",
        str(
            predicted_class
        ).upper(),
    )

    if confidence is not None:
        st.metric(
            "Prediction confidence",
            f"{confidence * 100:.2f}%",
        )

    if "enhanced_test_accuracy" in bundle:
        st.metric(
            "Model held-out test accuracy",
            f"{float(bundle['enhanced_test_accuracy']) * 100:.2f}%",
        )

    st.caption(
        "The prediction uses the final GrabCut-refined "
        "colour banana cut-out."
    )


# ============================================================
# WATERSHED PIPELINE
# ============================================================

st.divider()

st.header(
    "Marker-Controlled Watershed Pipeline"
)

row1 = st.columns(4)

with row1[0]:
    st.image(
        resized_rgb,
        caption=(
            f"1. Resize "
            f"({processing_size[0]}×{processing_size[1]})"
        ),
        use_container_width=True,
    )

with row1[1]:
    st.image(
        gaussian_rgb,
        caption=(
            f"2. Gaussian Filter "
            f"{gaussian_kernel}"
        ),
        use_container_width=True,
    )

with row1[2]:
    st.image(
        watershed["gray"],
        caption="3. Grayscale",
        clamp=True,
        use_container_width=True,
    )

with row1[3]:
    st.image(
        watershed["gradient"],
        caption="4. Sobel Gradient",
        clamp=True,
        use_container_width=True,
    )


marker_display = cv2.normalize(
    watershed[
        "markers"
    ].astype(np.float32),
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)


boundary_view = (
    gaussian_rgb.copy()
)

boundary_view[
    watershed[
        "watershed_labels"
    ] == -1
] = [
    255,
    255,
    255,
]


row2 = st.columns(4)

with row2[0]:
    st.image(
        marker_display,
        caption="5. Foreground / Background Markers",
        clamp=True,
        use_container_width=True,
    )

with row2[1]:
    st.image(
        boundary_view,
        caption="6. Watershed Boundaries",
        use_container_width=True,
    )

with row2[2]:
    st.image(
        watershed[
            "final_mask"
        ],
        caption="7. Watershed Mask",
        clamp=True,
        use_container_width=True,
    )

with row2[3]:
    st.image(
        watershed[
            "colour_cutout"
        ],
        caption="8. Watershed Colour Cut-Out",
        use_container_width=True,
    )


if "watershed_test_accuracy" in bundle:
    st.caption(
        "Recorded held-out accuracy for Watershed alone: "
        f"{float(bundle['watershed_test_accuracy']) * 100:.2f}%"
    )


# ============================================================
# GRABCUT REFINEMENT PIPELINE
# ============================================================

st.divider()

st.header(
    "After GrabCut Refinement"
)

st.write(
    "The Watershed mask is used as the initial object estimate. "
    "GrabCut then refines foreground/background separation, "
    "followed by the banana-only colour cleanup used in the "
    "final trained model."
)


grabcut_initialisation_display = cv2.normalize(
    grabcut[
        "grabcut_initialisation"
    ].astype(np.float32),
    None,
    0,
    255,
    cv2.NORM_MINMAX,
).astype(np.uint8)


grab_cols = st.columns(4)

with grab_cols[0]:
    st.image(
        grabcut_initialisation_display,
        caption="1. GrabCut Initialisation",
        clamp=True,
        use_container_width=True,
    )

with grab_cols[1]:
    st.image(
        grabcut[
            "raw_grabcut_mask"
        ],
        caption="2. Raw GrabCut Mask",
        clamp=True,
        use_container_width=True,
    )

with grab_cols[2]:
    st.image(
        grabcut[
            "refined_mask"
        ],
        caption="3. Final Refined Banana Mask",
        clamp=True,
        use_container_width=True,
    )

with grab_cols[3]:
    st.image(
        grabcut[
            "refined_colour_cutout"
        ],
        caption="4. Final Refined Colour Cut-Out",
        use_container_width=True,
    )


if "enhanced_test_accuracy" in bundle:
    st.caption(
        "Recorded held-out accuracy for Watershed + GrabCut refinement: "
        f"{float(bundle['enhanced_test_accuracy']) * 100:.2f}%"
    )


# ============================================================
# REGION FEATURES
# ============================================================

st.divider()

st.header(
    "Region Analysis"
)

feature_compare_df = pd.DataFrame(
    {
        "Feature": REGION_FEATURE_NAMES,
        "Watershed": watershed_region_features,
        "After GrabCut": refined_region_features,
    }
)

st.dataframe(
    feature_compare_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PROBABILITIES
# ============================================================

if (
    probabilities is not None
    and class_names is not None
):
    st.divider()

    st.header(
        "SVM Class Probabilities"
    )

    probability_df = pd.DataFrame(
        {
            "Class": class_names,
            "Probability": probabilities,
        }
    ).sort_values(
        "Probability",
        ascending=False,
    )

    probability_df[
        "Probability (%)"
    ] = (
        probability_df[
            "Probability"
        ]
        * 100
    ).round(2)

    st.dataframe(
        probability_df[
            [
                "Class",
                "Probability (%)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        probability_df.set_index(
            "Class"
        )[
            "Probability"
        ]
    )


st.divider()

st.caption(
    "Final prediction pipeline: "
    "Upload → Resize → Gaussian → Grayscale → Sobel Gradient → "
    "Marker-Controlled Watershed → GrabCut Refinement → "
    "Banana-Only Colour Cleanup → Final Colour Cut-Out → "
    f"{svm_image_size[0]}×{svm_image_size[1]} RGB Features → "
    "StandardScaler → RBF SVM → Ripeness Prediction"
)
