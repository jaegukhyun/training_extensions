_base_ = ["./train.py", "../_base_/models/detectors/detector.py"]

task_adapt = dict(
    type="default_task_adapt",
    op="REPLACE",
    efficient_mode=False,
    use_adaptive_anchor=True,
)

evaluation = dict(interval=1, metric="pascal_voc/mAP")

custom_hooks = [
    dict(
        type="LazyEarlyStoppingHook",
        start=3,
        patience=10,
        iteration_patience=0,
        metric="mAP",
        interval=1,
        priority=75,
    ),
    dict(
        type="EMAHook",
        priority="ABOVE_NORMAL",
        momentum=0.1,
    ),
]

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3333333333333333, by_epoch=False, begin=0, end=5),
    dict(type="ReduceOnPlateauLR", monitor="pascal_voc/mAP", patience=4, begin=5, min_value=1e-6, rule="greater"),
]

ignore = True
adaptive_validation_interval = dict(
    max_interval=5,
    enable_adaptive_interval_hook=True,
    enable_eval_before_run=True,
)
