from cfno.losses.data_loss import LpLoss, VectorNormLoss, LOSSES_CLASSES as DATA_LOSSES
from cfno.losses.physics_loss import LOSSES_CLASSES as PINN_LOSSES

LOSSES_CLASSES = {**DATA_LOSSES, **PINN_LOSSES}