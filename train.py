import multiprocessing as mp

from ultralytics import YOLO


def main():

    model = YOLO(r"ultralytics/cfg/models/dream/DREAM.yaml")

    model.train(
        data=r"R:\Dream\dataset\visdrone2021\visdrone-yolo11.yaml",  # dataset
        epochs=300,
        imgsz=640,
        device="0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
        project="runs\paper",
        name="Dream",
        exist_ok=True,
        batch=8,
        cfg="ultralytics/cfg/hyp_visdrone.yaml",
        workers=5,
        resume=False,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
