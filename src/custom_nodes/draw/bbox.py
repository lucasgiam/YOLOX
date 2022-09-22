"""
Draws bounding boxes over detected objects.
"""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.bbox import draw_bboxes


class Node(AbstractNode):
    """Draws bounding boxes on image.
    The :mod:`draw.bbox` node uses :term:`bboxes` and, optionally,
    :term:`bbox_labels` from the model predictions to draw the bbox predictions
    onto the image.
    Inputs:
        |img_data|
        |bboxes_data|
        |bbox_labels_data|
    Outputs:
        |none_output_data|
    Configs:
        show_labels (:obj:`bool`): **default = False**. |br|
            If ``True``, shows class label, e.g., "person", above the bounding
            box.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        print(inputs["bbox_labels"])

        # if len(inputs["bbox_labels"]) <= 1:
        #     if inputs["bbox_labels"] == ['all ppe']:
        #         color_ppe_status = [0, 255, 0]
        #     else:
        #         color_ppe_status = [0, 0, 255]
        # else:
        #     color_ppe_status = []
        #     for status in inputs["bbox_labels"]:
        #         if status == 'all ppe':
        #             color = [0, 255, 0]
        #         else:
        #             color = [0, 0, 255]
        #         color_ppe_status.append(color)

        if inputs["bbox_labels"] == ['all ppe']:
            color_ppe_status = [0, 255, 0]
        else:
            color_ppe_status = [0, 0, 255]

        # TODO: in utils/bbox.py, need to change line 63 to: color = color_choice[i]

        print(color_ppe_status)
        draw_bboxes(
            inputs["img"], inputs["bboxes"], inputs["bbox_labels"], self.show_labels, color_ppe_status
        )
        return {}

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"show_labels": bool}
