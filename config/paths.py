import os

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

    # 模型保存路径
    TRAINED_DRL_MODEL_PATH = os.path.join(RESULTS_PTH_DIR, "trained_drl_model.pth")

    # 轨迹可视化
    ALL_TRAJECTORIES_PNG = os.path.join(RESULTS_PNG_DIR, "all_trajectories.png")

    # 确保目录存在
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RESULTS_PNG_DIR, RESULTS_NPY_DIR, RESULTS_PTH_DIR, RESULTS_COM_EXP_DIR]:
        os.makedirs(dir_path, exist_ok=True)

