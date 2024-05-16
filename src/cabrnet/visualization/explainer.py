import os
import graphviz
from loguru import logger


class ExplanationGraph:
    r"""Object based on Graphviz used to generate explanation graphs.

    Attributes:
        output_dir: Path to output directory.
    """

    def __init__(self, output_dir: str) -> None:
        r"""Initializes explanation.

        Args:
            output_dir (str): Path to output directory.
        """
        self._dot = graphviz.Digraph()
        self._dot.attr(rankdir="LR")
        self._dot.attr(margin="0")
        self._dot.attr("node", shape="plaintext", label="", fixedsize="True", width="2", height="2")
        self._dot.attr("edge", penwidth="0.5")
        self.output_dir = output_dir
        self._num_nodes = 0

    def set_test_image(self, img_path: str, label: str = "") -> None:
        r"""Sets the test image.

        Args:
            img_path (str): Path to image.
            label (str, optional): Image label. Default: "".
        """
        # Relative path between the image and the final directory where rendering will occur
        rel_path = os.path.relpath(img_path, self.output_dir)
        if label == "":
            self._dot.node(name=f"node_{self._num_nodes}_test", label="", image=rel_path, imagescale="True")
        else:
            self._dot.node(
                name=f"node_{self._num_nodes}_test",
                height="2.3",
                imagepos="tc",
                label=label,
                labelloc="b",
                image=rel_path,
                imagescale="True",
            )
        if self._num_nodes > 0:
            self._dot.edge(
                tail_name=f"node_{self._num_nodes - 1}_test",
                head_name=f"node_{self._num_nodes}_test",
                label="",
            )
        self._num_nodes += 1

    def add_similarity(self, prototype_img_path: str, test_patch_img_path: str, label: str, font_color: str = "black"):
        r"""Adds a similarity comparison to the explanation graph.

        Args:
            prototype_img_path (str): Path to prototype patch visualization.
            test_patch_img_path (str): Path to test image patch visualization.
            label (str): Description of the similarity (e.g. similarity score).
            font_color (str, optional): Font color. Default: black.
        """
        rel_test_patch_img_path = os.path.relpath(test_patch_img_path, self.output_dir)
        rel_prototype_img_path = os.path.relpath(prototype_img_path, self.output_dir)
        # Create subgraph
        subgraph = graphviz.Digraph()
        subgraph.attr(rank="same")
        subgraph.node(name=f"node_{self._num_nodes}_test", image=rel_test_patch_img_path, imagescale="True")
        subgraph.node(
            name=f"node_{self._num_nodes}_label", label=label, fontcolor=font_color, fontsize="10", height="0.5"
        )
        subgraph.node(
            name=f"node_{self._num_nodes}_proto", image=rel_prototype_img_path, imagescale="True", imagepos="tc"
        )
        subgraph.edge(
            tail_name=f"node_{self._num_nodes}_test",
            head_name=f"node_{self._num_nodes}_label",
        )
        subgraph.edge(
            tail_name=f"node_{self._num_nodes}_label",
            head_name=f"node_{self._num_nodes}_proto",
        )
        self._dot.subgraph(subgraph)
        self._dot.edge(
            tail_name=f"node_{self._num_nodes-1}_test",
            head_name=f"node_{self._num_nodes}_test",
            label="",
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
        subgraph.node(name=f"node_prediction", label=f"Class: {class_id}")
        self._dot.subgraph(subgraph)
        self._dot.edge(
            tail_name=f"node_{self._num_nodes-1}_test",
            head_name=f"node_prediction",
            label="",
        )

    def render(self, path: str | None = None) -> None:
        r"""Generates explanation as a PDF file.

        Args:
            path (str, optional): If specified, path to the render file. Otherwise, it is set to "explanation".
                Default: None.
        """
        logger.debug(self._dot.source)
        if path is None:
            path = os.path.join(self.output_dir, "explanation")
        self._dot.render(filename=path)
