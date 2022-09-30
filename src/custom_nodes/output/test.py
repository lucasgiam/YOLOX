"""Records the nodes' outputs to a CSV file."""

import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from collections import deque, Counter
import numpy as np
import time
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
        self.track_ids = deque([]) # empty queue that is 2d
        # [[0, 1], [1, 0]] -> 2 frames
        self.begin = False 
        self.class_labels = deque([])
        # assume tracking id and class id match for at least 10 frames -- rationale is to determine exactly which of the 7 PPE violations it is
        # if the PPE violation keeps switching back and forth every frame, this is not deterministic as to exactly which PPE violation it is
        # assume tracking id for each unique person does not change with time
        self.global_violation_ids = {}

        
    def smart_notif(self, inputs):
        obj_attrs_ids = inputs["obj_attrs"]["ids"]

        self.track_ids.append(obj_attrs_ids)
        self.class_labels.append(inputs["bbox_labels"])
       
        begin = self.frame_counter == 10
        if begin:
            self.begin = True
        if self.begin:
            # assume that we are only looking at the past 10 frames
            # print(self.track_ids)
            # print(self.class_labels)
            last_10_track_ids = list(self.track_ids)
            last_10_class_labels = list(self.class_labels)
            last_10_track_ids = [item for sublist in last_10_track_ids for item in sublist]
            last_10_class_labels = [item for sublist in last_10_class_labels for item in sublist]
            # print(last_10_track_ids)
            # print(last_10_class_labels)

            violation_classes = ["person", "cup"]
            violation_ids = {}
            for track_id, class_label in zip(last_10_track_ids, last_10_class_labels):
                if class_label in violation_classes:
                    if track_id not in violation_ids:
                        violation_ids[track_id] = [1, class_label]   # [track_id_counter, class_label]
                    else:
                        violation_ids[track_id][0] += 1
                        
            # print(violation_ids)

            for track_id, v in violation_ids.items():
                track_id_counter = v[0]
                class_label = v[1]
                if track_id_counter == 10:
                    start_time = time.time()
                    # print("self.global_violation_ids: ", self.global_violation_ids)
                    if track_id not in self.global_violation_ids:
                        self.global_violation_ids[track_id] = start_time
                        self.send_to_payload(violation_type=class_label, track_id=track_id)
                    elif time.time() - self.global_violation_ids[track_id] >= 10:   # compute difference in time to trigger payload, 10 seconds is currently the threshold used
                        self.global_violation_ids[track_id] = start_time
                        self.send_to_payload(violation_type=class_label, track_id=track_id)
                    # print("current time: ", self.global_violation_ids[track_id])
                        
            self.track_ids.popleft()
            self.class_labels.popleft()
            
        return 1


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

        self.frame_counter += 1
        _ = self.smart_notif(inputs)
        return {"frame_counts": self.frame_counter}


    def _get_config_types(self) -> Dict[str, Any]:
        """Returns dictionary mapping the node's config keys to respective types."""
        return {"output_dir": str, "stats_to_track": List[str]}


    def send_to_payload(self, violation_type, track_id):

        # import requests
        # import datetime 
        # import time 

        print("triggered send_to_payload:", "violation type =", violation_type, "track_id =", track_id)

        # url = 'http://52.221.211.53/SendNotification'

        # dt = datetime.datetime.now()
        # dt = int(time.mktime(dt.timetuple()))
        # instance = 1
        # # violationType = 0
        # vidURL = 'https://drive.google.com/uc?export=download&id=1CQwjM0m3vgoelz3Xr7VwOJs0UY8cStqm'

        # payload = {
        #     "alarms" : [
        #         {
        #             "camId": '1',
        #             "time": dt,
        #             "startTime": dt,
        #             "endTime": dt+10,
        #             "instance": instance,
        #             "violationType": violation_type,
        #             "videoUrl": vidURL
        #         }
        #     ]
        # }
        # x = requests.post(url, json = payload)
        # print(x.text)
        # print(payload)
