import time

import torch as th
import gp_apis

class MatrixMuliplication_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_MatrixMuliplication(input1, input2, dim_0, dim_1, device0)
        ctx.backward_cache = input1, input2
        return res

    @staticmethod
    def backward(ctx, dZ):
        X, W = ctx.backward_cache
        dX = th.mm(dZ, W.t())
        dW = th.mm(X.t(), dZ)
        return dX, dW, None, None, None

def MatrixMuliplication(input1, input2, dim_0, dim_1, device0):
    return MatrixMuliplication_impl.apply(input1, input2, dim_0, dim_1, device0)

