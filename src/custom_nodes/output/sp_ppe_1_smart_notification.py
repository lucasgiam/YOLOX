from typing import Any, Dict
from collections import deque
import cv2
import os
import requests
import datetime 
import time
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from peekingduck.pipeline.nodes.abstract_node import AbstractNode


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # Assumptions:
        # 1. The class label (anything except 'all ppe') must remain the same for at least N frames before it counts as a violation -> rationale is to determine exactly which of the 7 PPE violations it is
        # 2. The tracking id for each unique person does not change with time

        # User configs:
        self.frames_threshold = 100         # min number of frames of consecutive PPE non-compliance to trigger a violation alert
        self.time_betw_trigs = 30           # time (in seconds) between two instances of same PPE non-compliance to trigger a violation alert
        self.frames_percent_trig = 0.90     # percentage of self.frames_threshold to trigger a violation alert, e.g. if self.frames_threshold = 100, then as long as any 90 frames are PPE non-compliance, it will trigger a violation alert 
        self.violation_classes = {"no ppe": 0, "helmet": 2, "mask": 3, "vest": 4, "mask + vest": 5, "helmet + mask": 6, "helmet + vest": 7}   # dict mapping class_label to violation_id

        # Do not config:
        self.frame_counter = 0
        self.track_ids = deque([])      # empty queue that is 2d, e.g. [[0, 1], [1, 0]] -> the outer list represents the 2 frames, while each inner list represents the 2 object tracking ids
        self.begin = False 
        self.class_labels = deque([])   # empty queue that is 2d, e.g. [['no ppe' 'mask'], ['mask' 'no ppe']] -> the outer list represents the 2 frames, while each inner list represents the class labels of the 2 objects
        self.global_violation_ids = {}

        
    def smart_notif(self, inputs):
        obj_attrs_ids = inputs["obj_attrs"]["ids"]

        self.track_ids.append(obj_attrs_ids)
        self.class_labels.append(inputs["bbox_labels"])
        # print(self.track_ids)
        # print(self.class_labels)

        begin = self.frame_counter == self.frames_threshold

        if begin:
            self.begin = True
        if self.begin:
            last_N_track_ids = list(self.track_ids)
            last_N_class_labels = list(self.class_labels)
            last_N_track_ids = [item for sublist in last_N_track_ids for item in sublist]
            last_N_class_labels = [item for sublist in last_N_class_labels for item in sublist]
            # print(last_N_track_ids)
            # print(last_N_class_labels)

            violation_ids = {}

            for track_id, class_label in zip(last_N_track_ids, last_N_class_labels):
                if class_label in self.violation_classes:
                    if track_id not in violation_ids:
                        violation_ids[track_id] = {}
                        if class_label not in violation_ids[track_id].keys():
                            violation_ids[track_id][class_label] = 1    # set counter = 1
                        else:
                            violation_ids[track_id][class_label] += 1   # add 1 to the counter if class_label is already present in violation_ids
                    else:
                        if class_label not in violation_ids[track_id].keys():
                            violation_ids[track_id][class_label] = 1    # set counter = 1
                        else:
                            violation_ids[track_id][class_label] += 1   # add 1 to the counter if class_label is already present in violation_ids
            # print('violation_ids:', violation_ids)

            for track_id, inner_dict in violation_ids.items():
                for class_label, counter in inner_dict.items():
                    if counter >= int(self.frames_percent_trig * self.frames_threshold):
                        start_time = time.time()
                        # print('self.global_violation_ids:', self.global_violation_ids)
                        if track_id not in self.global_violation_ids:
                            self.global_violation_ids[track_id] = {}   # inner dict within self.global_violation_ids dict
                            self.global_violation_ids[track_id][class_label] = []
                            self.global_violation_ids[track_id][class_label].append(start_time)
                            self.send_to_payload(violation_id=self.violation_classes[class_label])
                        elif class_label not in self.global_violation_ids[track_id].keys():
                            self.global_violation_ids[track_id][class_label] = []
                            self.global_violation_ids[track_id][class_label].append(start_time)
                            self.send_to_payload(violation_id=self.violation_classes[class_label])
                        else:
                            if start_time - self.global_violation_ids[track_id][class_label][0] >= self.time_betw_trigs:
                                self.global_violation_ids[track_id][class_label][0] = start_time
                                self.send_to_payload(violation_id=self.violation_classes[class_label])

            self.track_ids.popleft()
            self.class_labels.popleft()


    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            self._reset()
            return {}

        self.frame_counter += 1
        self.smart_notif(inputs)
        return {"frame_counts": self.frame_counter}


    def send_to_payload(self, violation_id):

        # vid_name, vid_path = self.img_to_vid()
        # vidURL = self.upload_to_google_drive(vid_name, vid_path)

        # url = 'http://52.221.211.53/SendNotification'

        # dt = datetime.datetime.now()
        # dt = int(time.mktime(dt.timetuple())) + 8*60*60   # convert to GMT+8 hours to seconds
        # instance = 1

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

        # requests.post(url, json = payload, verify = False)

        print("triggered send_to_payload:", "violation_id =", violation_id)
        # print("self.global_violation_ids: ", self.global_violation_ids)


    def img_to_vid(self):
        img_dir = './pkd_outputs/output_images'
        img_files = [os.path.splitext(filename)[0] for filename in os.listdir(img_dir)]
        vid_dir = './pkd_outputs/compiled_videos'
        vid_name = img_files[0] + '.mp4'
        vid_path = os.path.join(vid_dir, vid_name)

        images = [img for img in os.listdir(img_dir) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(img_dir, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(vid_path, fourcc, 20, (width, height))

        for img in images:
            video.write(cv2.imread(os.path.join(img_dir, img)))

        video.release()

        return vid_name, vid_path


    def upload_to_google_drive(self, vid_name, vid_path):

        # Authenticate Google Drive
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("mycreds.txt")   # try to load saved client credentials
        if gauth.credentials is None:
            gauth.LocalWebserverAuth()             # need to authenticate if mycreds.txt is not there, also ensure that client_secrets.json is in the same directory as this script
        elif gauth.access_token_expired:
            gauth.Refresh()                        # refresh credentials if expired
        else:
            gauth.Authorize()                      # initialize the saved credentials
        gauth.SaveCredentialsFile("mycreds.txt")   # save the current credentials to a file
        drive = GoogleDrive(gauth)

        # Locate folder in Google Drive
        fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file in fileList:
            if(file['title'] == "sp_ppe_1_videos"):
                fileID = file['id']

        # Upload video to Google Drive folder
        vid_upload = drive.CreateFile({'title': vid_name, 'parents': [{'kind': 'drive#fileLink', 'id': fileID}]})
        vid_upload.SetContentFile(vid_path)
        vid_upload.Upload()
        print('Uploaded file %s with mimeType %s' % (vid_upload['title'], vid_upload['mimeType']))

        # Generate URL to file
        # vidURL = vid_upload['alternateLink']
        vidURL = 'https://drive.google.com/uc?export=download&id=' + str(vid_upload['id'])
        
        return vidURL