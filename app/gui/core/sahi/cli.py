import fire

from gui.core.sahi import __version__ as sahi_version
from gui.core.sahi.predict import predict, predict_fiftyone
from gui.core.sahi.scripts.coco2fiftyone import main as coco2fiftyone
from gui.core.sahi.scripts.coco2yolov5 import main as coco2yolov5
from gui.core.sahi.scripts.coco_error_analysis import analyse
from gui.core.sahi.scripts.coco_evaluation import evaluate
from gui.core.sahi.scripts.slice_coco import slice
from gui.core.sahi.utils.import_utils import print_enviroment_info

coco_app = {
    "evaluate": evaluate,
    "analyse": analyse,
    "fiftyone": coco2fiftyone,
    "slice": slice,
    "yolov5": coco2yolov5,
}

sahi_app = {
    "predict": predict,
    "predict-fiftyone": predict_fiftyone,
    "coco": coco_app,
    "version": sahi_version,
    "env": print_enviroment_info,
}


def app() -> None:
    """Cli app."""
    fire.Fire(sahi_app)


if __name__ == "__main__":
    app()
