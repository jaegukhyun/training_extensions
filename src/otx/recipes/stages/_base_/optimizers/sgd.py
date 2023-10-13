_base_ = "./optimizer.py"

optim_wrapper = dict(type="OptimWrapper", optimizer=dict(type="SGD", lr=0.03, momentum=0.9))
