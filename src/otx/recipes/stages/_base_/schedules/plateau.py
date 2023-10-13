_base_ = "./schedule.py"

param_scheduler = [
    dict(type="LinearLR", start_factor=0.3333333333333333, by_epoch=False, begin=0, end=5),
    dict(type="ReduceOnPlateauLR", monitor="pascal_voc/mAP", patience=4, begin=5, min_value=1e-6, rule="greater"),
]
