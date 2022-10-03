"""Records the nodes' outputs to a CSV file."""

from typing import Any, Dict
from collections import deque
import requests
import datetime 
import time 
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.frame_counter = 0
        self.track_ids = deque([]) # empty queue that is 2d, e.g. [[0, 1], [1, 0]] -> 2 frames
        self.begin = False 
        self.class_labels = deque([])
        # assume tracking id and class id match for at least 10 frames -- rationale is to determine exactly which of the 7 PPE violations it is
        # if the PPE violation keeps switching back and forth every frame, this is not deterministic as to exactly which PPE violation it is
        # assume tracking id for each unique person does not change with time
        self.global_violation_ids = {}
        self.frames_threshold = 20
        self.time_betw_trigs = 10
        self.violation_classes = {"no ppe": 0, "helmet": 2, "mask": 3, "vest": 4, "mask + vest": 5, "helmet + mask": 6, "helmet + vest": 7}

        
    def smart_notif(self, inputs):
        obj_attrs_ids = inputs["obj_attrs"]["ids"]

        self.track_ids.append(obj_attrs_ids)
        self.class_labels.append(inputs["bbox_labels"])
       
        begin = self.frame_counter == self.frames_threshold
        if begin:
            self.begin = True
        if self.begin:
            last_N_track_ids = list(self.track_ids)
            last_N_class_labels = list(self.class_labels)
            last_N_track_ids = [item for sublist in last_N_track_ids for item in sublist]
            last_N_class_labels = [item for sublist in last_N_class_labels for item in sublist]
            # print(last_10_track_ids)
            # print(last_10_class_labels)

            violation_ids = {}
            for track_id, class_label in zip(last_N_track_ids, last_N_class_labels):
                if class_label in self.violation_classes:
                    if track_id not in violation_ids:
                        violation_ids[track_id] = [1, class_label]   # [track_id_counter, class_label]
                    else:
                        violation_ids[track_id][0] += 1
            print(violation_ids)

            for track_id, v in violation_ids.items():
                track_id_counter = v[0]
                class_label = v[1]
                if track_id_counter == self.frames_threshold:
                    start_time = time.time()
                    print("self.global_violation_ids: ", self.global_violation_ids)
                    if track_id not in self.global_violation_ids:
                        self.global_violation_ids[track_id] = {}
                        self.global_violation_ids[track_id][class_label] = []
                        self.global_violation_ids[track_id][class_label].append(start_time)
                        self.send_to_payload(violation_id=self.violation_classes[class_label])
                    elif class_label not in self.global_violation_ids[track_id].keys():
                        self.global_violation_ids[track_id][class_label] = []
                        self.global_violation_ids[track_id][class_label].append(start_time)
                        self.send_to_payload(violation_id=self.violation_classes[class_label])
                    # else:
                    #     for XXXX:
                    #         if elif time.time() - self.global_violation_ids[track_id][class_label][0] >= self.time_betw_trigs:   # compute difference in time between violations to trigger new payload
                    #             self.global_violation_ids[track_id][class_label][0] = start_time
                    #             self.send_to_payload(violation_id=self.violation_classes[class_label])
                    #             print("current time: ", self.global_violation_ids[track_id])
                            
            self.track_ids.popleft()
            self.class_labels.popleft()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            self._reset()
            return {}

        self.frame_counter += 1
        _ = self.smart_notif(inputs)
        return {"frame_counts": self.frame_counter}


    def send_to_payload(self, violation_id):

        # url = 'http://52.221.211.53/SendNotification'

        # dt = datetime.datetime.now()
        # dt = int(time.mktime(dt.timetuple())) + 8*60*60   # convert to GMT+8 hours to seconds
        # instance = 1
        # vidURL = 'https://drive.google.com/uc?export=download&id=1CQwjM0m3vgoelz3Xr7VwOJs0UY8cStqm'

        # payload = {
        #     "alarms" : [
        #         {
        #             "camId": '1',
        #             "time": dt,
        #             "startTime": dt,
        #             "endTime": dt+10,
        #             "instance": instance,
        #             "violationType": violation_id,
        #             "videoUrl": vidURL
        #         }
        #     ]
        # }

        # x = requests.post(url, json = payload, verify = False)

        print("triggered send_to_payload:", "violation_id =", violation_id)






# self.global_violation_ids = {0: [[t, 'no ppe'], [t, 'helmet'], [t, 'mask'], [t, 'all ppe']], 1: [[t, 'no ppe'], [t, 'helmet']], 2: [[t, 'no ppe']}

# {0: {'no ppe': [], 'mask': []}, 1: {'no ppe': [], 'mask': []}}
