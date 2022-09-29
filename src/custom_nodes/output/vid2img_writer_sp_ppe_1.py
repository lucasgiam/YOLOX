"""
Writes the output video to images.
Writes the output image(s) to image(s).
"""

import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):
    """Outputs the processed image or video to a file. A timestamp is appended to the
    end of the file name.
    Inputs:
        |img_data|
        |filename_data|
        |saved_video_fps_data|
        |pipeline_end_data|
    Outputs:
        |none_output_data|
    Configs:
        output_dir (:obj:`str`): **default = "PeekingDuck/data/output"**. |br|
            Output directory for files to be written locally.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.output_dir = Path(self.output_dir)  # type: ignore
        self._file_name: Optional[str] = None
        self._file_path_with_timestamp: Optional[str] = None
        self._image_type: Optional[str] = None
        self.writer = None
        self._prepare_directory(self.output_dir)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.logger.info(f"Output directory used is: {self.output_dir}")

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes media information to filepath."""
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            if self.writer:  # images automatically releases writer
                self.writer.release()
            return {}
        self._prepare_writer(
                inputs["filename"], inputs["img"], inputs["saved_video_fps"]
            )
        self._write(inputs["img"])
        filename = str(Path(self._file_path_with_timestamp).stem + ".jpg")
        output = {"filename": filename}
        return output

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"output_dir": str}

    def _write(self, img: np.ndarray) -> None:
        if self._image_type == "image":
            cv2.imwrite(self._file_path_with_timestamp, img)
        else:
            self._file_path_with_timestamp = Path(self._file_path_with_timestamp).with_suffix(".jpg")
            cv2.imwrite(str(self._file_path_with_timestamp), img)

    def _prepare_writer(
        self, filename: str, img: np.ndarray, saved_video_fps: int
    ) -> None:
        self._file_path_with_timestamp = self._append_datetime_filename(filename)

        if filename.split(".")[-1] in ["jpg", "jpeg", "png"]:
            self._image_type = "image"
        else:
            self._image_type = "video"


    @staticmethod
    def _prepare_directory(output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _append_datetime_filename(self, filename: str) -> str:
        self._file_name = filename
        current_time = datetime.datetime.now()
        # output as 'YYYYMMDD_hhmmss'
        time_str = current_time.strftime("%y%m%d_%H%M%S_%f")

        # append timestamp to filename before extension Format: filename_timestamp.extension
        filename_with_timestamp = f"_{time_str}.".join(filename.split(".")[-2:])
        file_path_with_timestamp = self.output_dir / filename_with_timestamp

        return str(file_path_with_timestamp)