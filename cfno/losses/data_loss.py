import torch

# Store all loss classes in a dictionary using the register decorator 
LOSSES_CLASSES = {}

def register(cls):
    assert hasattr(cls, '__call__') and callable(cls), "loss class must implement the __call__ method"
    LOSSES_CLASSES[cls.__name__] = cls
    return cls

@register
class LpLoss(object):
    """
    Model loss
    Args:
        d (int): dimension
        p (int): order of norm 
        size_average (bool): take average
        reduction (bool): perform reduction
    """
    def __init__(self, 
                 d:int=2,
                 p:int=2,
                 size_average:bool=True, 
                 reduction:bool=True,
                 device=None
    ):
        super().__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, pred, ref, inp=None):
        return self.rel(pred, ref)


@register
class VectorNormLoss(object):
    """
    Vector norm model loss
    Args:
        p (int): order of norm 
        out: model prediction of shape 
        ref: data reference of shape 
        dim: dimension of FFT
    """
    def __init__(self, 
                 p:int=2,
                 absolute=False,
                 dim=2,
                 device=None
    ):
        super().__init__()
        # Lp-norm type is positive
        assert p > 0
        assert dim in (2,3), "Only FNO2D and FNO3D supported"
        self.dim = dim
        self.p = p
        if absolute:
            self.__class__.__call__ = self.__class__.__call__ABS
        self.absolute = absolute

    def vectorNorm(self, x):
        if self.dim == 2:
            axis = (-2,-1)
        else:
            axis = (-3,-2,-1)
        return torch.linalg.vector_norm(x, ord=self.p, dim=axis)
    
    def __call__(self, pred, ref, inp=None):
        refNorms = self.vectorNorm(ref)
        diffNorms = self.vectorNorm(pred-ref)
        return torch.mean(diffNorms/refNorms)

    def __call__ABS(self, pred, ref, inp=None):
        return torch.mean(self.vectorNorm(pred-ref))

