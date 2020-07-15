from .seven_scenes import SevenScenes

from ..registry import DATASETS


@DATASETS.register_module(name='UnoRobot')
class UnoRobot(SevenScenes):
    '''dataset for Uno robot
    the data was collected and processed according to 7-scene
    dataset, specially:

    - UnoRobot/
        - scene1/
            - seq-1/
            - seq-2/
            - ...
            - TestSplit.txt
            - TrainSplit.txt
        - scene2/
            - ...
        - ...

    for each sub-directory in scene, (eg. scene1/seq-1/), a series of
    3 kind files are included:
        - `frame-{:06d}.color.png`: color frames
        - `frame-{:06d}.depth.png`: depth frames (optional)
        - `frame-{:06d}.pose.txt`: pose info represented by rotation matrix
    '''
    DATASET_NAME = 'UnoRobot'

    def __init__(self, *args, **kwargs):
        super(UnoRobot, self).__init__(*args, **kwargs)
