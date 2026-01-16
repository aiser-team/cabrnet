from pathlib import Path
import graphviz
import numpy as np
from loguru import logger
from PIL import Image

from cabrnet.core.utils.image import square_resize
from cabrnet.core.visualization.view import heatmap


class GenericGraph:
    r"""Generic object based on Graphviz used to render image graphs.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, output_dir: Path | None) -> None:
        r"""Initializes graph.

        Args:
            output_dir (Path): Path to output directory.
        """
        self._dot = graphviz.Digraph()
        self._dot.attr(rankdir="LR")
        self._dot.attr(margin="0")
        self._dot.attr("node", shape="plaintext", label="", fixedsize="True", width="2", height="2")
        self._dot.attr("edge", penwidth="0.5")
        self.output_dir = output_dir or Path.cwd()

    def render(self, path: Path | None = None, output_format: str = "pdf") -> None:
        r"""Renders graph into a file.

        Args:
            path (Path, optional): If specified, path to the render file. Otherwise, it is set to "explanation".
                Default: None.
            output_format (str, optional): Output file format. Default: pdf.
        """
        logger.debug(self._dot.source)
        if path is None:
            path = self.output_dir / "explanation"
        self._dot.render(filename=path, format=output_format)


class ExplanationGraph(GenericGraph):
    r"""Object based on Graphviz used to generate explanation graphs."""

    def __init__(self, output_dir: Path) -> None:
        r"""Initializes explanation.

        Args:
            output_dir (Path): Path to output directory.
        """
        super().__init__(output_dir)
        self._num_nodes = 0

    def set_test_image(
        self,
        img_path: Path,
        label: str = "",
        font_color: str = "black",
        draw_arrows: bool = True,
    ) -> None:
        r"""Sets the test image.

        Args:
            img_path (Path): Path to image.
            label (str, optional): Image label. Default: "".
            font_color (str, optional): Font color. Default: black.
            draw_arrows (bool, optional): If True, draw arrows connecting all images. Default: True.
        """
        if label == "":
            self._dot.node(name=f"node_{self._num_nodes}_test", label="", image=str(img_path), imagescale="True")
        else:
            self._dot.node(
                name=f"node_{self._num_nodes}_test",
                height=str(2.3 + 0.2 * label.count("\n")),
                imagepos="tc",
                label=label,
                fontcolor=font_color,
                labelloc="b",
                image=str(img_path),
                imagescale="True",
            )
        if self._num_nodes > 0:
            self._dot.edge(
                tail_name=f"node_{self._num_nodes - 1}_test",
                head_name=f"node_{self._num_nodes}_test",
                label="",
                style="invis" if not draw_arrows else "",
            )
        self._num_nodes += 1

    def add_similarity(
        self,
        prototype_img_path: Path,
        test_patch_img_path: Path,
        label: str,
        font_color: str = "black",
        draw_arrows: bool = True,
    ):
        r"""Adds a similarity comparison to the explanation graph.

        Args:
            prototype_img_path (Path): Path to prototype patch visualization.
            test_patch_img_path (Path): Path to test image patch visualization.
            label (str): Description of the similarity (e.g. similarity score).
            font_color (str, optional): Font color. Default: black.
            draw_arrows (bool, optional): If True, draw arrows connecting all images. Default: True.
        """
        # Create subgraph
        subgraph = graphviz.Digraph()
        subgraph.attr(rank="same")
        subgraph.node(name=f"node_{self._num_nodes}_test", image=str(test_patch_img_path), imagescale="True")
        subgraph.node(
            name=f"node_{self._num_nodes}_label", label=label, fontcolor=font_color, fontsize="10", height="0.5"
        )
        subgraph.node(
            name=f"node_{self._num_nodes}_proto", image=str(prototype_img_path), imagescale="True", imagepos="tc"
        )
        subgraph.edge(
            tail_name=f"node_{self._num_nodes}_test",
            head_name=f"node_{self._num_nodes}_label",
            style="invis" if not draw_arrows else "",
        )
        subgraph.edge(
            tail_name=f"node_{self._num_nodes}_label",
            head_name=f"node_{self._num_nodes}_proto",
            style="invis" if not draw_arrows else "",
        )
        self._dot.subgraph(subgraph)
        self._dot.edge(
            tail_name=f"node_{self._num_nodes-1}_test",
            head_name=f"node_{self._num_nodes}_test",
            label="",
            style="invis" if not draw_arrows else "",
        )
        self._num_nodes += 1

    def add_prediction(self, class_id: str | int) -> None:
        r"""Adds prediction to the explanation graph.

        Args:
            class_id (str or int): Prediction value.
        """
        # Create subgraph
        subgraph = graphviz.Digraph()
        subgraph.attr(rank="same")
        subgraph.node(name="node_prediction", label=f"Class: {class_id}")
        self._dot.subgraph(subgraph)
        self._dot.edge(
            tail_name=f"node_{self._num_nodes-1}_test",
            head_name="node_prediction",
            label="",
        )


class PerturbationGraph(GenericGraph):
    r"""Object based on Graphviz used to visualize perturbations.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, font_color: str = "black", **kwargs) -> None:
        r"""Initializes graph.

        Args:
            font_color (str, optional): Font color. Default: black.
        """
        super().__init__(**kwargs)
        self._dot.attr(layout="neato")
        # Default node attributes
        self._dot.attr("node", height="2.3", imagepos="tc", labelloc="b", imagescale="True", fontcolor=font_color)
        self._num_blocks = 0

    def add_block(
        self,
        perturbation: str,
        prototype_label: str,
        prototype_img_path: Path,
        test_patch_img_path: Path,
        test_patch_heatmap_path: Path,
        focus_test_patch_img_path: Path,
        dual_test_patch_img_path: Path | None,
        original_sim_score: float,
        focus_sim_score: float,
        dual_sim_score: float = 0.0,
    ):
        r"""Adds a perturbation block to the graph.

        Args:
            perturbation (str): Perturbation name.
            prototype_label (str): Prototype name.
            prototype_img_path (Path): Path to prototype patch visualization.
            test_patch_img_path (Path): Path to test image patch visualization.
            test_patch_heatmap_path (Path): Path to test image patch visualization in heatmap form.
            focus_test_patch_img_path (Path): Path to test image patch visualization, after focused perturbation.
            dual_test_patch_img_path (Path): Path to test image patch visualization, after dual perturbation.
            original_sim_score (float): Original similarity score, before perturbation.
            focus_sim_score (float): Similarity score, after focused perturbation.
            dual_sim_score (float, optional): Similarity score, after dual perturbation. Default: 0.0.
        """
        # Reference coordinate
        y_ref = -self._num_blocks * 8

        # Create subgraph
        subgraph = graphviz.Digraph()

        # Add all image nodes
        subgraph.node(
            name=f"node_{self._num_blocks}_test_ref",
            label="Test patch",
            image=str(test_patch_img_path),
            pos=f"4,{y_ref + 5}!",
        )
        subgraph.node(
            name=f"node_{self._num_blocks}_proto",
            label=f"Prototype {prototype_label}",
            image=str(prototype_img_path),
            pos=f"8,{y_ref+2.5}!",
        )
        subgraph.node(
            name=f"node_{self._num_blocks}_test_hm",
            label="Patch heatmap",
            image=str(test_patch_heatmap_path),
            pos=f"0,{y_ref + 2.5}!",
        )
        subgraph.node(
            name=f"node_{self._num_blocks}_test_focus",
            height=str(2.3 + 0.2 * perturbation.count("\n")),
            label=f"{perturbation} (focus)",
            image=str(focus_test_patch_img_path),
            pos=f"4,{y_ref+2.5}!",
        )
        if dual_test_patch_img_path is not None:
            subgraph.node(
                name=f"node_{self._num_blocks}_test_dual",
                height=str(2.3 + 0.2 * perturbation.count("\n")),
                label=f"{perturbation} (dual)",
                image=str(dual_test_patch_img_path),
                pos=f"4,{y_ref}!",
            )

        # Connect everything
        subgraph.edge(
            tail_name=f"node_{self._num_blocks}_test_ref",
            head_name=f"node_{self._num_blocks}_proto",
            dir="both",
            label=f"{original_sim_score:.2f}",
        )
        self._dot.edge(
            tail_name=f"node_{self._num_blocks}_test_ref",
            head_name=f"node_{self._num_blocks}_test_hm",
        )
        subgraph.edge(
            tail_name=f"node_{self._num_blocks}_test_hm",
            head_name=f"node_{self._num_blocks}_test_focus",
        )
        subgraph.edge(
            tail_name=f"node_{self._num_blocks}_test_focus",
            head_name=f"node_{self._num_blocks}_proto",
            dir="both",
            label=f"{focus_sim_score:.2f}",
            fontcolor="blue" if (original_sim_score - focus_sim_score) / original_sim_score > 0.1 else "red",
        )
        if dual_test_patch_img_path is not None:
            subgraph.edge(
                tail_name=f"node_{self._num_blocks}_test_hm",
                head_name=f"node_{self._num_blocks}_test_dual",
            )
            subgraph.edge(
                tail_name=f"node_{self._num_blocks}_test_dual",
                head_name=f"node_{self._num_blocks}_proto",
                dir="both",
                label=f"{dual_sim_score:.2f}",
                fontcolor="blue" if (original_sim_score - dual_sim_score) / original_sim_score > 0.1 else "red",
            )
        self._dot.subgraph(subgraph)
        self._num_blocks += 1


class PointingGameGraph(GenericGraph):
    r"""Object based on Graphviz used to visualize the pointing game.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, font_color: str = "black", **kwargs) -> None:
        r"""Initializes graph.

        Args:
            font_color (str, optional): Font color. Default: black.
        """
        super().__init__(**kwargs)
        self._dot.attr(layout="neato")
        # Default node attributes
        self._dot.attr("node", height="2.3", imagepos="tc", labelloc="b", imagescale="True", fontcolor=font_color)
        self._num_blocks = 0

    def add_block(
        self,
        prototype_img_path: Path,
        original_img: Image.Image,
        test_patch_img: Image.Image,
        segmentation: Image.Image,
        attribution: np.ndarray,
        area_percentage: float,
        img_id: str | int,
        proto_idx: int,
        sim_score: float,
        energy_score: float,
        mask_score: float,
        prototype_mode: bool,
    ):
        r"""Adds a perturbation block to the graph.

        Args:
            prototype_img_path (Path): Path to prototype patch visualization.
            original_img (Image): Original test image.
            test_patch_img (Image): Image patch visualization.
            segmentation (Image): Object segmentation.
            attribution (numpy array): Attribution array.
            area_percentage (float): Percentage of the most relevant pixels to intersect with segmentation mask.
            img_id (str or int): Test image identifier.
            proto_idx (int): Prototype index.
            sim_score (float): Similarity score between test patch and prototype.
            energy_score (float): Energy-based pointing game relevance of the attribution w.r.t the segmentation.
            mask_score (float): Mask-based pointing game relevance of the attribution w.r.t the segmentation.
            prototype_mode (bool): In prototype mode, ignore the similarity with a test patch.
        """
        # Reference coordinate
        y_ref = -self._num_blocks * 7

        # Save the test patch, the segmentation, the heatmap and their intersection
        output_dir = self.output_dir / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        object_seg = np.sum(np.asarray(segmentation), axis=-1)
        test_img_seg_path = output_dir / f"img_{img_id}.png"
        test_patch_img_path = output_dir / f"img_{img_id}_v_proto{proto_idx}.png"
        test_patch_heatmap_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_heatmap.png"
        test_patch_heatmap_in_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_heatmap_in.png"
        test_patch_heatmap_out_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_heatmap_out.png"
        test_patch_mask_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_mask.png"
        test_patch_mask_in_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_mask_in.png"
        test_patch_mask_out_path = output_dir / f"img_{img_id}_v_proto{proto_idx}_mask_out.png"

        square_resize(segmentation).save(test_img_seg_path)
        square_resize(test_patch_img).save(test_patch_img_path)
        square_resize(heatmap(img=original_img, sim_map=attribution, overlay=True)).save(test_patch_heatmap_path)
        square_resize(heatmap(img=original_img, sim_map=attribution * (object_seg > 0), overlay=True)).save(
            test_patch_heatmap_in_path
        )
        square_resize(heatmap(img=original_img, sim_map=attribution * (object_seg == 0), overlay=True)).save(
            test_patch_heatmap_out_path
        )

        sorted_attribution = np.sort(np.reshape(attribution, (-1)))
        threshold = sorted_attribution[int(len(sorted_attribution) * (1 - area_percentage))]
        masked_attribution = attribution > threshold
        square_resize(Image.fromarray(masked_attribution)).save(test_patch_mask_path)
        square_resize(Image.fromarray(masked_attribution * (object_seg > 0))).save(test_patch_mask_in_path)
        square_resize(Image.fromarray(masked_attribution * (object_seg == 0))).save(test_patch_mask_out_path)

        # Add all image nodes
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch",
            label="Test patch" if not prototype_mode else f"Prototype {proto_idx}",
            image=str(test_patch_img_path),
            pos=f"0,{y_ref + 3}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_seg",
            label="Segmentation",
            image=str(test_img_seg_path),
            pos=f"0,{y_ref}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_heatmap",
            label="Patch heatmap",
            image=str(test_patch_heatmap_path),
            pos=f"3,{y_ref + 3}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_mask",
            label=f"Patch mask ({int(area_percentage*100)}% area)",
            image=str(test_patch_mask_path),
            pos=f"3, {y_ref}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_heatmap_in",
            label=f"Inside ({int(energy_score*100)}%)",
            image=str(test_patch_heatmap_in_path),
            pos=f"7,{y_ref+3}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_mask_in",
            label=f"Inside ({int(mask_score*100)}%)",
            image=str(test_patch_mask_in_path),
            pos=f"7,{y_ref}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_heatmap_out",
            label=f"Outside ({100-int(energy_score*100)}%)",
            image=str(test_patch_heatmap_out_path),
            pos=f"10,{y_ref+3}!",
        )
        self._dot.node(
            name=f"node_{self._num_blocks}_test_patch_mask_out",
            label=f"Outside ({100-int(mask_score*100)}%)",
            image=str(test_patch_mask_out_path),
            pos=f"10,{y_ref}!",
        )

        # Connect everything
        self._dot.edge(
            tail_name=f"node_{self._num_blocks}_test_patch", head_name=f"node_{self._num_blocks}_test_patch_heatmap"
        )
        self._dot.edge(
            tail_name=f"node_{self._num_blocks}_test_patch_heatmap",
            head_name=f"node_{self._num_blocks}_test_patch_mask",
        )

        if not prototype_mode:
            prototype_resize_img_path = output_dir / f"prototype_{proto_idx}.png"
            square_resize(Image.open(prototype_img_path)).save(prototype_resize_img_path)  # Resize and copy prototype
            self._dot.node(
                name=f"node_{self._num_blocks}_proto",
                label=f"Prototype {proto_idx}",
                image=str(prototype_resize_img_path),
                pos=f"-4,{y_ref+3}!",
            )
            self._dot.edge(
                tail_name=f"node_{self._num_blocks}_test_patch",
                head_name=f"node_{self._num_blocks}_proto",
                dir="both",
                label=f"Similarity: {sim_score:.2f}   ",
                labelloc="tc",
            )

        self._num_blocks += 1


class PrototypeAnalysisGraph(GenericGraph):
    r"""Object based on Graphviz used to visualize global prototype analysis.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, font_color: str = "black", **kwargs) -> None:
        r"""Initializes graph.

        Args:
            output_dir (Path): Path to output directory.
            font_color (str, optional): Font color. Default: black.
        """
        super().__init__(output_dir=None)
        self._dot.attr(layout="neato")
        # Default node attributes
        self._dot.attr("node", height="2.3", imagepos="tc", labelloc="b", imagescale="True", fontcolor=font_color)
        self._num_blocks = 0

    def add_block(
        self,
        prototype_label: str,
        prototype_img_path: Path,
        test_patch_img_path: Path,
        radar_plot_path: Path,
        original_sim_score: float,
    ):
        r"""Adds an analysis block to the graph.

        Args:
            prototype_label (str): Prototype name.
            prototype_img_path (Path): Path to prototype patch visualization.
            test_patch_img_path (Path): Path to test image patch visualization.
            radar_plot_path (Path): Path to the radar plot.
            original_sim_score (float): Original similarity score, before perturbation.
        """
        # Reference coordinate
        y_ref = -self._num_blocks * 4

        # Create subgraph
        subgraph = graphviz.Digraph()

        # Add all image nodes
        subgraph.node(
            name=f"node_{self._num_blocks}_test_ref",
            label="Test patch",
            image=str(test_patch_img_path),
            pos=f"4,{y_ref + 2.5}!",
        )
        subgraph.node(
            name=f"node_{self._num_blocks}_proto",
            label=prototype_label,
            image=str(prototype_img_path),
            pos=f"8,{y_ref+2.5}!",
        )
        subgraph.node(
            height="5",
            width="5",
            imagescale="True",
            name=f"node_{self._num_blocks}_radar",
            image=str(radar_plot_path),
            pos=f"0,{y_ref + 2.5}!",
        )
        # Connect everything
        subgraph.edge(
            tail_name=f"node_{self._num_blocks}_test_ref",
            head_name=f"node_{self._num_blocks}_proto",
            dir="both",
            label=f"{original_sim_score:.2f}",
        )
        self._dot.subgraph(subgraph)
        self._num_blocks += 1
