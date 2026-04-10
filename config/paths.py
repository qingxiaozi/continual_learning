import os
from config.parameters import Config


class Paths:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    TRAJECTORY_DIR = os.path.join(DATA_DIR, "PortoTaxi")

    # 结果子目录
    RESULTS_PNG_DIR = os.path.join(RESULTS_DIR, "png")
    RESULTS_NPY_DIR = os.path.join(RESULTS_DIR, "npy")
    RESULTS_PTH_DIR = os.path.join(RESULTS_DIR, "pth")
    RESULTS_COM_EXP_DIR = os.path.join(RESULTS_DIR, "com_exp")

    # 轨迹可视化
    ALL_TRAJECTORIES_PNG = os.path.join(RESULTS_PNG_DIR, "all_trajectories.png")

    # 确保基础目录存在
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RESULTS_PNG_DIR, RESULTS_NPY_DIR, RESULTS_PTH_DIR, RESULTS_COM_EXP_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def get_dataset_dir(sub_dir="com_exp"):
        dataset_name = Config.CURRENT_DATASET
        if sub_dir == "com_exp":
            return os.path.join(Paths.RESULTS_COM_EXP_DIR, dataset_name)
        elif sub_dir == "png":
            return os.path.join(Paths.RESULTS_PNG_DIR, dataset_name)
        elif sub_dir == "npy":
            return os.path.join(Paths.RESULTS_NPY_DIR, dataset_name)
        elif sub_dir == "pth":
            return os.path.join(Paths.RESULTS_PTH_DIR, dataset_name)
        return Paths.RESULTS_DIR

    @staticmethod
    def get_drl_model_path():
        dataset_name = Config.CURRENT_DATASET
        model_dir = os.path.join(Paths.RESULTS_PTH_DIR, dataset_name)
        model_filename = f"trained_drl_model_{dataset_name}_3.pth"
        full_path = os.path.join(model_dir, model_filename)
        os.makedirs(model_dir, exist_ok=True)
        return full_path

    @staticmethod
    def get_visualizer_path():
        dataset_name = Config.CURRENT_DATASET
        return os.path.join(Paths.RESULTS_PNG_DIR, dataset_name)
