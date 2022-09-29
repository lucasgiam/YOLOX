"""Records the nodes' outputs to a CSV file."""

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.abstract_node import AbstractNode
from peekingduck.pipeline.nodes.output.utils.csvlogger import CSVLogger


class Node(AbstractNode):
    """Tracks user-specified parameters and outputs the results in a CSV file.
    Inputs:
        ``all`` (:obj:`List`): A placeholder that represents a flexible input.
        Actual inputs to be written into the CSV file can be configured in
        ``stats_to_track``.
    Outputs:
        |none_output_data|
    Configs:
        stats_to_track (:obj:`List[str]`):
            **default = ["keypoints", "bboxes", "bbox_labels"]**. |br|
            Parameters to log into the CSV file. The chosen parameters must be
            present in the data pool.
        file_path (:obj:`str`):
            **default = "PeekingDuck/data/stats.csv"**. |br|
            Path of the CSV file to be saved. The resulting file name would have an appended
            timestamp.
        logging_interval (:obj:`int`): **default = 1**. |br|
            Interval between each log, in terms of seconds.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(self.output_dir)  # type: ignore
        self._stats_checked = False
        self.stats_to_track: List[str]
        self.csv_logger = None
        self.frame_counter = 0
        self.track_ids = []

    def smart_notif(self, inputs):
        obj_attrs_ids = inputs["obj_attrs"]["ids"]
        self.track_ids.append(obj_attrs_ids)
        begin = self.frame_counter == 10
        while begin:
            print(self.frame_counter - 10, self.frame_counter)
            last_10 = self.track_ids[self.frame_counter - 10: self.frame_counter]
            print(last_10)




    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes the current state of the tracked statistics into
        the csv file as a row entry
        Args:
            inputs (dict): The data pool of the pipeline.
        Returns:
            outputs: [None]
        """
        # reset and terminate when there are no more data

        if inputs["pipeline_end"]:
            self._reset()
            return {}

        self.csv_logger = CSVLogger(Path(inputs["filename"]), self.stats_to_track)

        if not self._stats_checked:
            self._check_tracked_stats(inputs)
            # self._stats_to_track might change after the check
            self.csv_logger = CSVLogger(Path(inputs["filename"]), self.stats_to_track)

        self.csv_logger.write(inputs, self.stats_to_track)
        self.frame_counter += 1
        print(self.smart_notif(inputs))
        return {"frame_counts": self.frame_counter}

    def _check_tracked_stats(self, inputs: Dict[str, Any]) -> None:
        """Checks whether user input statistics is present in the data pool
        of the pipeline. Statistics not present in data pool will be
        ignored and dropped.
        """
        valid = []
        invalid = []

        for stat in self.stats_to_track:
            if stat in inputs:
                valid.append(stat)
            else:
                invalid.append(stat)

        if invalid:
            msg = textwrap.dedent(
                f"""\
                {invalid} are not valid outputs.
                Data pool only has this outputs: {list(inputs.keys())}
                Only {valid} will be logged in the csv file.
                """
            )
            self.logger.warning(msg)

        # update stats_to_track with valid stats found in data pool
        self.stats_to_track = valid
        self._stats_checked = True

    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"output_dir": str, "stats_to_track": List[str]}

    def _reset(self) -> None:
        del self.csv_logger

        # initialize for use in run
        self._stats_checked = False