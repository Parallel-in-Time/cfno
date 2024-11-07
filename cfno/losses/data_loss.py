import torch

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
                 reduction:bool=True
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

    def __call__(self, x, y):
        return self.rel(x, y)


class VectormNormLoss(object):
    """
    Vector norm model loss
    Args:
        p (int): order of norm 
        out: model prediction of shape [nBatch,nVar,nX,nZ]
        ref: data reference of shape [nBatch,nVar,nX,nZ]
    """
    def __init__(self, 
                 p:int=2,
    ):
        super().__init__()
        # Lp-norm type is postive
        assert p > 0
        self.p = p


    def vectorNorm(self, x, dim=(-2,-1)):
        return torch.linalg.vector_norm(x, ord=self.p, dim=dim)
    
    def __call__(self, out, ref):
        refNorms = self.vectorNorm(ref)
        diffNorms = self.vectorNorm(out-ref)
        return torch.mean(diffNorms/refNorms)

