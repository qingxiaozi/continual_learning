import os

class Paths:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')

    # 确保目录存在
    for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

paths = Paths()