## 一、项目结构

continual_learning/
├── main.py
├── config/
│   ├── __init__.py
│   ├── parameters.py
│   └── paths.py
├── environment/
│   ├── __init__.py
│   ├── vehicle_env.py
│   ├── communication.py
│   └── data_simulator.py
├── models/
│   ├── __init__.py
│   ├── neural_networks.py
│   ├── mab_selector.py
│   └── drl_agent.py
├── learning/
│   ├── __init__.py
│   ├── continual_learner.py
│   ├── cache_manager.py
│   └── evaluator.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
└── experiments/
    ├── __init__.py
    ├── baseline_comparison.py
    └── ablation_study.py