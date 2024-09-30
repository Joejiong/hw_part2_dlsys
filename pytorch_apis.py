import torch as th
import gp_apis
import time

class addVec_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, device0):
        res = gp_apis.gp_addVec(input1, input2, dim_0, device0)
        ctx.save_for_backward(input1, input2)
        return res

    @staticmethod
    def backward(ctx, dZ):
        grad_input1 = dZ
        grad_input2 = dZ
        return grad_input1, grad_input2, None, None


def addVec(input1, input2, dim_0, device0):
    return addVec_impl.apply(input1, input2, dim_0, device0)

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

