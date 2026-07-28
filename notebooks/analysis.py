import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    from collections import defaultdict
    from io import BytesIO
    from math import ceil
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    import yaml
    from PIL import Image
    from sklearn.metrics import confusion_matrix
    from torchvision.transforms import ToTensor
    from torchvision.datasets import ImageFolder

    from cabrnet.archs.generic.model import CaBRNet
    from cabrnet.core.evaluation.local_perturbation_analysis import (
        analyze as perturbation_analyze,
    )
    from cabrnet.core.evaluation.relevance_analysis import (
        analyze as pointing_game_analyze,
    )
    from cabrnet.core.utils.data import DatasetManager
    from cabrnet.core.utils.parser import load_config
    from cabrnet.core.utils.save import load_projection_info
    from cabrnet.core.visualization.depictor import check_attribution_type, check_view_type
    from cabrnet.core.visualization.explainer import PrototypeAnalysisGraph
    from cabrnet.core.visualization.radar_plot import radar_plot
    from cabrnet.core.visualization.view import SUPPORTED_VIEWING_FUNCTIONS
    from cabrnet.core.visualization.visualizer import (
        SimilarityVisualizer,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cabrnet's model analyser

    This script enables you to select a trained model and visualize explanations in a wide range of situations.
    """)
    return


@app.cell(hide_code=True)
def _():
    checkpoint_browser = mo.ui.file_browser(
        selection_mode="directory",
        multiple=False,
    )
    checkpoint_browser
    return (checkpoint_browser,)


@app.cell(hide_code=True)
def _(checkpoint_browser):
    mo.stop(
        len(checkpoint_browser.value) == 0,
        mo.callout("Select a path containing a trained model to continue", "warn"),
    )
    checkpoint_path = checkpoint_browser.value[0].path
    state_dict_path = checkpoint_path / CaBRNet.DEFAULT_MODEL_STATE
    mo.stop(
        not state_dict_path.exists(),
        mo.callout("Selected path does not contain checkpoint", "danger"),
    )
    dataloaders = DatasetManager.get_dataloaders(checkpoint_path / DatasetManager.DEFAULT_DATASET_CONFIG)
    datasets, _ = DatasetManager.get_datasets_and_indices(checkpoint_path / DatasetManager.DEFAULT_DATASET_CONFIG)
    test_dataset = datasets["test_set"]["dataset"]
    test_dataset_raw = datasets["test_set"]["raw_dataset"]
    classes = test_dataset.classes
    return (
        checkpoint_path,
        classes,
        dataloaders,
        datasets,
        state_dict_path,
        test_dataset,
        test_dataset_raw,
    )


@app.cell(hide_code=True)
def _():
    device_selector = mo.ui.radio(
        ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []),
        value="cuda:0" if torch.cuda.is_available() else "cpu",
        inline=True,
    )
    device_selector
    return (device_selector,)


@app.cell
def _(checkpoint_path, device_selector, state_dict_path):
    model = CaBRNet.build_from_config(
        checkpoint_path / CaBRNet.DEFAULT_MODEL_CONFIG,
        state_dict_path=state_dict_path,
    ).to(device_selector.value)
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Evaluation
    """)
    return


@app.cell
def _():
    run_button = mo.ui.run_button(label="Evaluate model")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def _(dataloaders, device_selector, model, run_button):
    mo.stop(
        not run_button.value,
        mo.callout("Click on previous button to evaluate model", "info"),
    )
    evaluation = model._evaluate_batches(
        dataloader=dataloaders["test_set"],
        device=device_selector.value,
        verbose=True,
        collect_predictions=True,
    )
    stats_eval = {f"test_set/{key}": value for key, value in evaluation.stats.items()}
    outputs, labels = evaluation.logits, evaluation.labels
    return labels, outputs, stats_eval


@app.cell(hide_code=True)
def _(labels, outputs, stats_eval):
    preds = outputs.argmax(1)
    confusion = confusion_matrix(labels.cpu(), preds.cpu())
    mo.ui.tabs(
        {
            "Evaluation statistics": {k: round(v, 3) for k, v in stats_eval.items()},
            "Confusion matrix": mo.mpl.interactive(plt.imshow(confusion, cmap="Reds").axes),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Analysis
    """)
    return


@app.cell(hide_code=True)
def _(checkpoint_path):
    output_directory_picker = mo.ui.file_browser(
        initial_path=checkpoint_path,
        multiple=False,
        selection_mode="directory",
    )
    output_directory_picker
    return (output_directory_picker,)


@app.cell(hide_code=True)
def _(checkpoint_path, output_directory_picker):
    # Determine prototype path and load existing visualization config if available
    if output_directory_picker.value:
        prototype_path = (output_directory_picker.value[0]) / "prototypes"
    else:
        prototype_path = checkpoint_path / "prototypes"

    viz_config_file = prototype_path / SimilarityVisualizer.DEFAULT_VISUALIZATION_CONFIG
    if viz_config_file.exists():
        with open(viz_config_file) as f:
            _existing_viz_config = yaml.safe_load(f)
        default_attribution = check_attribution_type(
            _existing_viz_config.get("attribution", {}).get("type", "saliency"),
            SimilarityVisualizer.SUPPORTED_ATTRIBUTION_METHODS,
            viz_config_file,
        )
        check_view_type(_existing_viz_config.get("view", {}).get("type", "heatmap"), viz_config_file)
        default_attribution_params = _existing_viz_config.get("attribution", {}).get("params", {})
        default_view = _existing_viz_config.get("view", {}).get("type", "heatmap")
        default_view_params = _existing_viz_config.get("view", {}).get("params", {})
    else:
        default_attribution = "saliency"
        default_attribution_params = {}
        default_view = "heatmap"
        default_view_params = {}
    return (
        default_attribution,
        default_attribution_params,
        default_view,
        default_view_params,
        prototype_path,
        viz_config_file,
    )


@app.cell
def _(default_attribution):
    attribution_selector = mo.ui.radio(
        SimilarityVisualizer.SUPPORTED_ATTRIBUTION_METHODS,
        value=default_attribution,
        inline=True,
    )
    attribution_selector
    return (attribution_selector,)


@app.cell(hide_code=True)
def _(attribution_selector, default_attribution_params):
    _attribution = attribution_selector.value
    _is_grad_method = _attribution in ["prp", "smoothgrad", "saliency", "randgrad"]

    _attribution_elements = {
        "num_samples": mo.ui.slider(
            start=1,
            stop=50,
            step=1,
            value=default_attribution_params.get("num_samples", 10),
            show_value=True,
        ),
        "noise_ratio": mo.ui.number(
            value=default_attribution_params.get("noise_ratio", 0.2),
        ),
        "stability_factor": mo.ui.number(
            value=default_attribution_params.get("stability_factor", 1e-6),
        ),
        "polarity": mo.ui.radio(
            ["positive", "negative", "absolute", "none"],
            value=default_attribution_params.get("polarity", "positive"),
        ),
        "gaussian_ksize": mo.ui.slider(
            start=1,
            stop=20,
            step=1,
            value=default_attribution_params.get("gaussian_ksize", 5),
            show_value=True,
        ),
        "similarity_threshold": mo.ui.number(
            value=default_attribution_params.get("similarity_threshold", 0.1),
        ),
        "grads_x_input": mo.ui.switch(
            value=default_attribution_params.get("grads_x_input", False),
        ),
    }

    # Only include relevant parameters based on attribution method
    _param_map = {
        "smoothgrad": [
            "num_samples",
            "noise_ratio",
            "polarity",
            "gaussian_ksize",
            "similarity_threshold",
            "grads_x_input",
        ],
        "prp": ["stability_factor", "normalize"],
        "saliency": [
            "polarity",
            "gaussian_ksize",
            "similarity_threshold",
            "grads_x_input",
        ],
        "randgrad": [
            "polarity",
            "gaussian_ksize",
            "similarity_threshold",
            "grads_x_input",
        ],
    }

    if _attribution in _param_map:
        explanation_parameters = mo.ui.dictionary(
            {k: v for k, v in _attribution_elements.items() if k in _param_map[_attribution]},
            label="Attribution settings",
        )
    else:
        explanation_parameters = mo.ui.dictionary({})
    explanation_parameters if _attribution in _param_map else mo.callout(
        "No parameter for this attribution function", "info"
    )
    return (explanation_parameters,)


@app.cell
def _(default_view):
    view_selector = mo.ui.radio(list(SUPPORTED_VIEWING_FUNCTIONS.keys()), value=default_view, inline=True)
    view_selector
    return (view_selector,)


@app.cell(hide_code=True)
def _(default_view_params, view_selector):
    view_method = view_selector.value

    view_parameters = mo.ui.dictionary(
        {
            "percentile": mo.ui.slider(
                start=0,
                stop=1,
                step=0.05,
                value=default_view_params.get("percentile", 0.7),
                show_value=True,
            ),
            "thickness": mo.ui.slider(
                start=1,
                stop=5,
                step=1,
                value=default_view_params.get("thickness", 2),
                show_value=True,
                disabled=(view_method == "crop_to_percentile"),
            ),
            "overlay": mo.ui.switch(
                value=default_view_params.get("overlay", True),
                disabled=(view_method != "heatmap"),
            ),
        },
        label="Visualization settings",
    )
    view_parameters
    return (view_parameters,)


@app.cell(hide_code=True)
def _(
    attribution_selector,
    checkpoint_path,
    explanation_parameters,
    model,
    view_parameters,
    view_selector,
    viz_config_file,
):
    viz_config = {
        "attribution": {
            "type": attribution_selector.value,
            "params": explanation_parameters.value,
        },
        "view": {"type": view_selector.value, "params": view_parameters.value},
    }

    existing_viz_config = None
    if viz_config_file.exists():
        with open(viz_config_file) as _f:
            existing_viz_config = yaml.safe_load(_f)

    visualizer = SimilarityVisualizer.build_from_config(
        viz_config, model=model, dataset_config=load_config(checkpoint_path / DatasetManager.DEFAULT_DATASET_CONFIG)
    )
    return existing_viz_config, visualizer, viz_config


@app.cell
def _(existing_viz_config, viz_config):
    run_extract_proto = mo.ui.run_button(
        label="Extract prototypes" if viz_config != existing_viz_config else "Force re-extract prototype"
    )
    run_extract_proto
    return (run_extract_proto,)


@app.cell(hide_code=True)
def _(
    checkpoint_path,
    datasets,
    device_selector,
    existing_viz_config,
    model,
    prototype_path,
    run_extract_proto,
    visualizer,
    viz_config,
    viz_config_file,
):
    mo.stop(
        not run_extract_proto.value and existing_viz_config != viz_config,
        mo.callout(
            "Prototypes must be re-generated, click on 'Extract prototypes'",
            "warn",
        ),
    )
    prototype_path.mkdir(exist_ok=True)

    if run_extract_proto.value:
        model.extract_prototypes(
            raw_dataset=datasets["projection_set"]["raw_dataset"],
            projection_info=load_projection_info(filename=checkpoint_path / CaBRNet.DEFAULT_PROJECTION_INFO),
            depictor=visualizer,
            dir_path=prototype_path,
            device=device_selector.value,
            verbose=True,
        )

    # Save visualization config to prototypes folder

    with open(viz_config_file, "w") as _f:
        yaml.dump(viz_config, _f)

    # create an artificial dependency so that marimo can reload
    model_with_prototypes = model
    return (model_with_prototypes,)


@app.cell(hide_code=True)
def _(existing_viz_config, run_extract_proto, viz_config):
    mo.stop(
        not run_extract_proto.value and existing_viz_config != viz_config,
        mo.callout(
            "Prototypes must be re-generated, click on 'Extract prototypes'",
            "warn",
        ),
    )
    run_explain_global = mo.ui.run_button(label="Generate global explanation")
    run_explain_global
    return (run_explain_global,)


@app.cell(hide_code=True)
def _(
    checkpoint_path,
    model,
    output_directory_picker,
    prototype_path,
    run_explain_global,
):
    mo.stop(not run_explain_global.value)

    if output_directory_picker.value:
        global_explanation_path = (output_directory_picker.value[0]) / "global"
    else:
        global_explanation_path = checkpoint_path / "global"

    model.explain_global(prototype_dir=prototype_path, output_dir=global_explanation_path, output_format="pdf")
    mo.pdf(global_explanation_path / "global_explanation.pdf")
    return


@app.function(hide_code=True)
def sample_examples_from_dataset(
    dataset: torch.utils.data.Dataset, limit_per_class=5, n_attempts=1000
) -> dict[int, list[int]]:
    """
    Returns random samples from a dataset, with a maximum of samples per class.
    This is a very naive function (samples uniformly in the dataset, don't apppend if the class has enough samples)
    The upside is that it works with an arbitrary dataset

    Args:
        dataset: dataset to sample
        limit_per_class: maximum number of samples per class
        n_attempts: total number of samples, before filtering to respect limit_per_class

    Returns:
        a dictionnary class_number -> samples
    """
    result = defaultdict(list)
    for i in torch.randperm(len(dataset))[:n_attempts].tolist():
        target = dataset[i][1]
        if len(result[target]) < limit_per_class:
            result[target].append(i)
    return dict(sorted(result.items(), key=lambda item: item[0]))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Select sample image**
    """)
    return


@app.cell(hide_code=True)
def _(classes, test_dataset_raw):
    samples_for_selection = sample_examples_from_dataset(test_dataset_raw, 5, 2000)

    def _create_row(class_number, indices):
        result = {"name": classes[class_number]}
        for i, index in enumerate(indices):
            # marimo magic: we create a UI element, but load it only when visible.
            result[str(i)] = mo.lazy(test_dataset_raw[index][0])
        return result

    img_picker = mo.ui.file()
    table = mo.ui.table(
        [_create_row(class_number, indices) for (class_number, indices) in samples_for_selection.items()],
        selection="single-cell",
        freeze_columns_left=["name"],
    )
    example_picker = mo.ui.tabs({"File": img_picker, "Dataset": table})
    example_picker
    return example_picker, img_picker, samples_for_selection, table


@app.cell(hide_code=True)
def _(
    example_picker,
    img_picker,
    samples_for_selection,
    table,
    test_dataset_raw,
):
    if example_picker.value == "File":
        mo.stop(
            len(img_picker.value) == 0,
            mo.callout("Select an image to compute explanation", "warn"),
        )
        img = Image.open(BytesIO(img_picker.value[0].contents))
    else:
        mo.stop(
            len(table.value) == 0,
            mo.callout("Select an image to compute explanation", "warn"),
        )
        img_id = list(samples_for_selection.values())[int(table.value[0].row)][int(table.value[0].column)]
        img = test_dataset_raw[img_id][0]

    run_button_explanation = mo.ui.run_button(label="Compute explanation")

    class_specific_switch = mo.ui.switch(value=True, label="Show only prototypes for chosen class")
    mo.vstack(
        [
            mo.image(img, height="100%").style({"height": "400px"}),
            class_specific_switch,
            run_button_explanation,
        ]
    )
    return class_specific_switch, img, img_id, run_button_explanation


@app.cell
def _(
    checkpoint_path,
    class_specific_switch,
    device_selector,
    img,
    model_with_prototypes,
    prototype_path,
    run_button_explanation,
    test_dataset,
    visualizer,
):
    mo.stop(
        not run_button_explanation.value,
        mo.callout("Click on previous button to run explanation", "info"),
    )

    model_with_prototypes.to("cuda:0").explain(
        img=img,
        preprocess=test_dataset.transform,
        depictor=visualizer,
        device=device_selector.value,
        prototype_dir=prototype_path,
        output_dir=checkpoint_path / "explanations",
        exist_ok=True,
        output_format="pdf",
        class_specific=class_specific_switch.value,
    )
    mo.pdf(checkpoint_path / "explanations" / "explanation.pdf")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Explanation metrics

    This part provides different ways to measure the correctness / robustness of the provided explanations
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Local perturbation analysis

    Tests prototype robustness to image perturbations (brightness, contrast, saturation, hue, blur, distortion).
    """)
    return


@app.cell
def _():
    pert_options = mo.ui.dictionary(
        {
            "num_prototypes": mo.ui.slider(
                start=1,
                stop=10,
                step=1,
                value=2,
                show_value=True,
            ),
            "dual_mode": mo.ui.switch(value=True),
        },
        label="Analysis settings",
    )
    pert_options
    return (pert_options,)


@app.cell(hide_code=True)
def _():
    perturbation_type = mo.ui.radio(
        options=[
            "brightness",
            "contrast",
            "saturation",
            "hue",
            "blur",
            "distortion",
            "all",
        ],
        inline=True,
        value="brightness",
    )
    perturbation_type
    return (perturbation_type,)


@app.cell(hide_code=True)
def _(perturbation_type):
    _perturbation_elements = {
        "brightness": mo.ui.slider(
            start=0,
            stop=1,
            step=0.1,
            value=0.3,
            show_value=True,
        ),
        "contrast": mo.ui.slider(
            start=0,
            stop=1,
            step=0.1,
            value=0.2,
            show_value=True,
        ),
        "saturation": mo.ui.slider(
            start=0,
            stop=1,
            step=0.1,
            value=0.3,
            show_value=True,
        ),
        "hue": mo.ui.slider(
            start=-0.5,
            stop=0.5,
            step=0.1,
            value=0.3,
            show_value=True,
        ),
        "gaussian_ksize": mo.ui.slider(
            start=1,
            stop=29,
            step=2,
            value=21,
            show_value=True,
        ),
        "gaussian_sigma": mo.ui.slider(
            start=0,
            stop=5,
            step=0.5,
            value=2.0,
            show_value=True,
        ),
        "distortion_periods": mo.ui.slider(
            start=0,
            stop=10,
            step=1,
            value=5,
            show_value=True,
        ),
        "distortion_amplitude": mo.ui.slider(
            start=0,
            stop=10,
            step=0.5,
            value=7.0,
            show_value=True,
        ),
        "distortion_direction": mo.ui.dropdown(
            options=["horizontal", "vertical", "both"],
            value="both",
        ),
    }

    # Only include relevant parameters based on selection
    _perturbation_type = perturbation_type.value
    if _perturbation_type == "all":
        perturbation_parameters = mo.ui.dictionary(_perturbation_elements, label="Perturbation strength")
    else:
        # Map perturbation type to its required parameter keys
        param_map = {
            "brightness": ["brightness"],
            "contrast": ["contrast"],
            "saturation": ["saturation"],
            "hue": ["hue"],
            "blur": ["gaussian_ksize", "gaussian_sigma"],
            "distortion": [
                "distortion_periods",
                "distortion_amplitude",
                "distortion_direction",
            ],
        }
        perturbation_parameters = mo.ui.dictionary(
            {k: v for k, v in _perturbation_elements.items() if k in param_map.get(_perturbation_type, [])},
            label="Perturbation strength",
        )
    perturbation_parameters
    return (perturbation_parameters,)


@app.cell(hide_code=True)
def _():
    PERTURBATION_CONFIGS = {
        "brightness": lambda p: {
            "type": "brightness",
            "params": {"brightness_factor": p["brightness"]},
        },
        "contrast": lambda p: {
            "type": "contrast",
            "params": {"contrast_factor": p["contrast"]},
        },
        "saturation": lambda p: {
            "type": "saturation",
            "params": {"saturation_factor": p["saturation"]},
        },
        "hue": lambda p: {"type": "hue", "params": {"hue_factor": p["hue"]}},
        "blur": lambda p: {
            "type": "blur",
            "params": {
                "gaussian_blur_ksize": int(p["gaussian_ksize"]),
                "gaussian_blur_sigma": p["gaussian_sigma"],
            },
        },
        "distortion": lambda p: {
            "type": "distortion",
            "params": {
                "distortion_periods": int(p["distortion_periods"]),
                "distortion_amplitude": p["distortion_amplitude"],
                "distortion_direction": p["distortion_direction"],
            },
        },
    }

    PERTURBATION_NAMES = list(PERTURBATION_CONFIGS.keys())

    def run_all_perturbations(
        model,
        img,
        preprocess,
        visualizer,
        device,
        params: dict,
        num_prototypes: int,
        dual_mode: bool,
        output_dir,
        prototype_dir,
    ) -> tuple[dict, dict]:
        """Run all perturbations and return delta scores per prototype."""
        r = {}
        original_scores = {}

        for perturbation_name in PERTURBATION_NAMES:
            cfg = PERTURBATION_CONFIGS[perturbation_name](params)
            perturbations = {perturbation_name: cfg}

            stats = perturbation_analyze(
                model=model,
                img=img,
                img_id="selected",
                preprocess=preprocess,
                visualizer=visualizer,
                device=device,
                perturbations=perturbations,
                num_prototypes=num_prototypes,
                enable_dual_mode=dual_mode,
                debug_dir=output_dir,
                debug_format="png",
                prototype_dir=prototype_dir,
            )

            for stat in stats:
                delta_score = abs(stat["original_score"] - stat["score_after_perturbation"])
                proto_idx = stat["proto_idx"]
                if proto_idx not in r:
                    original_scores[proto_idx] = stat["original_score"]
                    r[proto_idx] = [delta_score]
                else:
                    r[proto_idx].append(delta_score)

        return r, original_scores

    return PERTURBATION_CONFIGS, PERTURBATION_NAMES, run_all_perturbations


@app.function(hide_code=True)
def build_analysis_graph(
    names: list,
    delta_scores: dict,
    original_scores: dict,
    output_dir: Path,
    prototype_dir: Path,
) -> Path:
    """Build radar plots and aggregate graph. Return path to final image."""
    max_value = ceil(max([max(scores) for scores in delta_scores.values()]))

    graph = PrototypeAnalysisGraph()
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)

    for proto_idx, scores in delta_scores.items():
        radar_path = output_dir / f"proto_{proto_idx}.png"
        radar_plot(
            labels=names,
            data=scores,
            output_file=radar_path,
            title=f"Prototype {proto_idx}",
            max_value=max_value,
        )

        patch_path = images_dir / f"img_p{proto_idx}_patch.png"
        prototype_img_path = prototype_dir / f"prototype_{proto_idx}.png"

        graph.add_block(
            prototype_label=f"Prototype {proto_idx}",
            prototype_img_path=prototype_img_path,
            test_patch_img_path=patch_path,
            radar_plot_path=radar_path,
            original_sim_score=original_scores[proto_idx],
        )

    output_path = output_dir / "img_sensitivity_radar.pdf"
    graph.render(output_path.with_suffix(""), output_format="pdf")
    return output_path


@app.cell
def _():
    run_perturbation_btn = mo.ui.run_button(label="Run perturbation analysis")
    run_perturbation_btn
    return (run_perturbation_btn,)


@app.cell(hide_code=True)
def _(
    PERTURBATION_CONFIGS,
    PERTURBATION_NAMES,
    checkpoint_path,
    device_selector,
    img,
    model_with_prototypes,
    pert_options,
    perturbation_parameters,
    perturbation_type,
    prototype_path,
    run_all_perturbations,
    run_perturbation_btn,
    test_dataset,
    visualizer,
):
    mo.stop(
        not run_perturbation_btn.value,
        mo.callout("Click on previous button to run perturbation analysis", "info"),
    )

    # Prepare output directory
    perturbation_dir = checkpoint_path / "perturbation_analysis"
    perturbation_dir.mkdir(exist_ok=True, parents=True)

    p = perturbation_parameters.value
    selected = perturbation_type.value
    delta_scores = None

    if selected != "all":
        # Single perturbation analysis
        cfg = PERTURBATION_CONFIGS[selected](p)
        perturbations = {selected: cfg}
        stats_perturbation = perturbation_analyze(
            model=model_with_prototypes,
            img_id="",
            img=img,
            preprocess=test_dataset.transform,
            visualizer=visualizer,
            device=device_selector.value,
            perturbations=perturbations,
            num_prototypes=pert_options.value["num_prototypes"],
            enable_dual_mode=pert_options.value["dual_mode"],
            debug_dir=perturbation_dir,
            debug_format="pdf",
            prototype_dir=prototype_path,
        )
        result_path = perturbation_dir / "img_sensitivity.pdf"
        mo.ui.table(stats_perturbation, label="Perturbation analysis results")
    else:
        # Analyze all perturbations
        delta_scores, original_scores = run_all_perturbations(
            model=model_with_prototypes,
            img=img,
            preprocess=test_dataset.transform,
            visualizer=visualizer,
            device=device_selector.value,
            params=p,
            num_prototypes=pert_options.value["num_prototypes"],
            dual_mode=pert_options.value["dual_mode"],
            output_dir=perturbation_dir,
            prototype_dir=prototype_path,
        )

        # Build visualization
        result_path = build_analysis_graph(
            names=PERTURBATION_NAMES,
            delta_scores=delta_scores,
            original_scores=original_scores,
            output_dir=perturbation_dir,
            prototype_dir=prototype_path,
        )

    mo.pdf(result_path)
    return


@app.cell
def _():
    num_prototypes_pointing = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=2,
        show_value=True,
    )
    area_percentage = mo.ui.slider(
        start=0.01,
        stop=1,
        step=0.01,
        value=0.1,
        show_value=True,
    )
    mo.ui.dictionary(
        {
            "num_prototypes": num_prototypes_pointing,
            "area_percentage": area_percentage,
        },
        label="Analysis settings",
    )
    return area_percentage, num_prototypes_pointing


@app.cell
def _():
    run_pointing_btn = mo.ui.run_button(label="Run pointing game analysis")
    run_pointing_btn
    return (run_pointing_btn,)


@app.cell
def _(
    area_percentage,
    checkpoint_path,
    device_selector,
    example_picker,
    img,
    img_id,
    model_with_prototypes,
    num_prototypes_pointing,
    prototype_path,
    run_pointing_btn,
    test_dataset,
    visualizer,
):
    mo.stop(
        example_picker.value != "Dataset",
        mo.callout(
            mo.md(
                f"Pointing game analysis is only supported with dataset images: "
                "select the image from the dataset instead of from file (see image selector; tab 'Dataset')",
            ),
            "danger",
        ),
    )
    mo.stop(
        not run_pointing_btn.value,
        mo.callout("Click on previous button to run pointing game analysis", "info"),
    )

    # Look for segmentations directory
    segmentation_dir = Path(test_dataset.root).parent.parent / "segmentations"

    mo.stop(
        not segmentation_dir.exists(),
        mo.callout(
            mo.md(
                f"Segmentation directory not found at {segmentation_dir}. \n"
                f"Please add segmentation masks to {segmentation_dir} to use pointing game analysis. You may use this command:\n\n"
                f"`.venv/bin/python3 tools/download_datasets.py --use-segmentation --target <dataset_name>`\n",
            ),
            "warn",
        ),
    )

    # Find corresponding segmentation file (search for common image extensions)
    segmentation_file = (segmentation_dir / "/".join(test_dataset.imgs[img_id][0].split("/")[-2:])).with_suffix(".png")

    # Load segmentation with PIL
    seg = Image.open(segmentation_file)

    pointing_dir = checkpoint_path / "relevance"
    pointing_dir.mkdir(exist_ok=True, parents=True)

    stats_pointing_game = pointing_game_analyze(
        model=model_with_prototypes,
        img=img,
        img_id="selected",
        seg=seg,
        preprocess=test_dataset.transform,
        visualizer=visualizer,
        device=device_selector.value,
        area_percentage=area_percentage.value,
        num_prototypes=num_prototypes_pointing.value,
        debug_dir=pointing_dir,
        debug_format="pdf",
        prototype_dir=prototype_path,
    )

    mo.pdf(pointing_dir / "imgselected_relevance_analysis.pdf")
    return


if __name__ == "__main__":
    app.run()
