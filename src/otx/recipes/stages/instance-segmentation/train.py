_base_ = [
    "../_base_/default.py",
    "../_base_/logs/log.py",
    "../_base_/optimizers/sgd.py",
    "../_base_/schedules/plateau.py",
]

default_scope = "mmdet"

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.001),
)

evaluation = dict(
    interval=1,
    metric="pascal_voc/mAP",
)
early_stop_metric = "pascal_voc/mAP"

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3333333333333333, by_epoch=False, begin=0, end=5),
    dict(type="ReduceOnPlateauLR", monitor="pascal_voc/mAP", patience=4, begin=5, min_value=1e-6, rule="greater"),
]

# Check all of these hooks are needed
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=100),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, save_best="pascal_voc/mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

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
        type="AdaptiveTrainSchedulingHook",
        enable_adaptive_interval_hook=False,
        enable_eval_before_run=True,
    ),
    dict(type="LoggerReplaceHook"),
]
