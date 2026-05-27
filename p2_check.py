# Check your ultralytics version and if the yaml exists
import ultralytics, pathlib
print('ultralytics:', ultralytics.__version__)
cfg = pathlib.Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8-p2.yaml'
print('yolov8-p2.yaml exists:', cfg.exists())
cfg_n = pathlib.Path(ultralytics.__file__).parent / 'cfg/models/v8/yolov8n-p2.yaml'
print('yolov8n-p2.yaml exists:', cfg_n.exists())
