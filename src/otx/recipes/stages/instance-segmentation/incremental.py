_base_ = ["./train.py", "../_base_/models/detectors/detector.py"]

task = "instance-segmentation"

task_adapt = dict(
    type="default_task_adapt",
    op="REPLACE",
    efficient_mode=False,
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.001),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)

ignore = True
adaptive_validation_interval = dict(
    max_interval=5,
    enable_adaptive_interval_hook=True,
    enable_eval_before_run=True,
)
