_base_ = ["./dist/dist.py"]

randomness = dict(seed=5, deterministic=False)
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl", linear_scale_lr=True),
)

task_adapt = dict(op="REPLACE")

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
