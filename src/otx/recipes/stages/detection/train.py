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
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    ),
)

evaluation = dict(
    interval=1,
    metric="pascal_voc/mAP",
)
early_stop_metric = "pascal_voc/mAP"

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
        metric="bbox_mAP",
        interval=1,
        priority=75,
    ),
]
