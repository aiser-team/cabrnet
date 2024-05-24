from cabrnet.visualization.explainer import ExplanationGraph
import os
import graphviz


class DebugGraph(ExplanationGraph):
    r"""Object based on Graphviz used to generate explanation graphs.

    Attributes:
        output_dir: Path to output directory.
    """

    def add_pairs(
        self,
        top_img_path: str,
        top_img_label: str,
        bot_img_path: str,
        bot_img_label: str,
        top_img_font_color: str = "black",
        bot_img_font_color: str = "black",
    ):
        r"""Adds a pair of images to the explanation graph.

        Args:
            top_img_path (str): Path to top image visualization.
            top_img_label (str): Description of the top image.
            bot_img_path (str): Path to bottom image visualization.
            bot_img_label (str): Description of the bottom image.
            top_img_font_color (str, optional): Font color for top image. Default: black.
            bot_img_font_color (str, optional): Font color for bottom image. Default: black.

        """
        rel_top_img_path = os.path.relpath(top_img_path, self.output_dir)
        rel_bot_img_path = os.path.relpath(bot_img_path, self.output_dir)

        # Create subgraph
        subgraph = graphviz.Digraph()
        subgraph.attr(rank="same")
        # Keep node naming consistent with add_similarity to maintain references if necessary
        subgraph.node(
            name=f"node_{self._num_nodes}_proto",
            height=str(2.3 + 0.2 * bot_img_label.count("\n")),
            imagepos="tc",
            label=bot_img_label,
            fontcolor=bot_img_font_color,
            labelloc="b",
            image=rel_bot_img_path,
            imagescale="True",
        )
        subgraph.node(
            name=f"node_{self._num_nodes}_test",
            height=str(2.3 + 0.2 * top_img_label.count("\n")),
            imagepos="tc",
            label=top_img_label,
            fontcolor=top_img_font_color,
            labelloc="b",
            image=rel_top_img_path,
            imagescale="True",
        )

        self._dot.subgraph(subgraph)
        self._dot.edge(
            tail_name=f"node_{self._num_nodes - 1}_test",
            head_name=f"node_{self._num_nodes}_test",
            label="",
            style="invis",
        )
        self._num_nodes += 1
