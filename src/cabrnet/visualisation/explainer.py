import os
import graphviz
from loguru import logger


class ExplanationGraph:
    def __init__(self, output_dir: str) -> None:
        """Init explanation

        Args:
            output_dir: path to output directory containing explanation
        """
        self._dot = graphviz.Digraph()
        self._dot.attr(rankdir="LR")
        self._dot.attr(margin="0")
        self._dot.attr("node", shape="plaintext", label="", fixedsize="True", width="2", height="2")
        self._dot.attr("edge", penwidth="0.5")
        self.output_dir = output_dir
        self._num_nodes = 0

    def set_test_image(self, img_path: str) -> None:
        """Set test image

        Args:
            img_path: path to image
        """
        # Relative path between the image and the final directory where rendering will occur
        rel_path = os.path.relpath(img_path, self.output_dir)
        self._dot.node(name="node_0_test", label="", image=rel_path, imagescale="True")
        self._num_nodes += 1

    def add_similarity(self, prototype_img_path: str, test_patch_img_path: str, label: str):
        rel_test_patch_img_path = os.path.relpath(test_patch_img_path, self.output_dir)
        rel_prototype_img_path = os.path.relpath(prototype_img_path, self.output_dir)
        # Create subgraph
        subgraph = graphviz.Digraph()
        subgraph.attr(rank="same")
        subgraph.node(name=f"node_{self._num_nodes}_test", image=rel_test_patch_img_path, imagescale="True")
        subgraph.node(name=f"node_{self._num_nodes}_label", label=label, fontsize="10", height="0.5")
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

    def render(self) -> None:
        """Generate explanation file"""
        logger.debug(self._dot.source)
        self._dot.render(filename=os.path.join(self.output_dir, "explanation"))
