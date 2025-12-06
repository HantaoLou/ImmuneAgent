"""
losses for VoxelMorph
"""

# Third party inports
# import tensorflow as tf
# import tensorflow.backend as K
# import tensorflow.keras.layers as KL
# from tensorflow.keras.layers import ReLU

import numpy as np
import sys
import torch
import torch.nn.functional as F

# import utils

# class NCC():
#     """
#     local (over window) normalized cross correlation
#     """
#
#     def __init__(self, win=None, num_ch=1, eps=1e-5, central=True, smoonth=False):
#         self.win = win
#         self.eps = eps
#         self.eyes = tf.reshape(tf.eye(num_ch), [1] * 3 + [num_ch, num_ch])
#         self.central = central
#         self.ndims = 3
#         self.strides = [1] * (self.ndims + 2)
#         # set window size
#         if self.win is None:
#             self.win = [8] * self.ndims
#         if smoonth:
#             self.kernels = self._build_kernel(std=.5)
#         self.smoonth = smoonth
#         self.sum_filt = self._build_kernel(std=0.)
#
#     def _build_kernel(self, std=0.):
#         if std == 0.:
#             return tf.ones([*self.win, 1, 1]) * self.eyes
#         else:
#             tail = int(np.ceil(std)) * 3
#             k = tf.exp([-0.5 * x ** 2 / std ** 2 for x in range(-tail, tail + 1)])
#             kernel = k / tf.reduce_sum(k)
#             return tf.reshape(kernel, [-1, 1, 1, 1, 1]) * tf.reshape(kernel, [1, -1, 1, 1, 1]) * tf.reshape(kernel,
#                                                                                                             [1, 1, -1,
#                                                                                                              1,
#                                                                                                              1]) * self.eyes
#
#     def ncc(self, I, J, label=None):
#         # get dimension of volume
#         # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
#         # ndims = len(I.get_shape().as_list()) - 2
#         # assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
#         padding = 'SAME'
#
#         # compute filters
#
#         # get convolution function
#         conv_fn = getattr(tf.nn, 'conv%dd' % self.ndims)
#         if self.smoonth:
#             I = conv_fn(I, self.kernels, self.strides, padding)
#             J = conv_fn(J, self.kernels, self.strides, padding)
#         # compute CC squares
#         I2 = I * I
#         J2 = J * J
#         IJ = I * J
#
#         if self.central:
#             # compute local sums via convolution
#             I_sum = conv_fn(I, self.sum_filt, self.strides, padding)
#             J_sum = conv_fn(J, self.sum_filt, self.strides, padding)
#             I2_sum = conv_fn(I2, self.sum_filt, self.strides, padding)
#             J2_sum = conv_fn(J2, self.sum_filt, self.strides, padding)
#             IJ_sum = conv_fn(IJ, self.sum_filt, self.strides, padding)
#             # compute cross correlation
#             win_size = np.prod(self.win)
#
#             # u_I = I_sum/win_size
#             # u_J = J_sum/win_size
#             # cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
#             # I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
#             # J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size
#             cross = tf.reduce_sum(IJ_sum, axis=-1) - tf.reduce_sum(I_sum * J_sum, axis=-1) / win_size
#             I_var = tf.reduce_sum(I2_sum, axis=-1) - tf.reduce_sum(I_sum * I_sum, axis=-1) / win_size
#             J_var = tf.reduce_sum(J2_sum, axis=-1) - tf.reduce_sum(J_sum * J_sum, axis=-1) / win_size
#         else:
#             # compute local sums via convolution
#             I2_sum = conv_fn(I2, self.sum_filt, self.strides, padding)
#             J2_sum = conv_fn(J2, self.sum_filt, self.strides, padding)
#             IJ_sum = conv_fn(IJ, self.sum_filt, self.strides, padding)
#
#             cross = tf.reduce_sum(IJ_sum, axis=-1)
#             I_var = tf.reduce_sum(I2_sum, axis=-1)
#             J_var = tf.reduce_sum(J2_sum, axis=-1)
#
#         cc = (cross * cross / (I_var * J_var + self.eps))
#         if label is not None:
#             label = tf.reduce_sum(tf.cast(label, dtype='float32'), axis=-1)
#             cc = tf.reduce_mean(cc * label, axis=[1, 2, 3], keepdims=False) / (
#                         tf.reduce_mean(label, axis=[1, 2, 3], keepdims=False) + self.eps)
#             # cc= cc
#             # return negative cc.
#         return tf.reduce_mean(cc)
#
#     def loss(self, I, J, label=None):
#         return - self.ncc(I, J, label=label)

EPS=1e-7

class Grad(torch.nn.Module):
    """
    N-D gradient loss
    """

    def __init__(self, penalty=['l1'],ndims=2, eps=1e-8, outrange_weight=100, detj_weight=50, apear_scale=5, dist=1, sign=1):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.eps = eps
        self.outrange_weight = outrange_weight
        self.detj_weight=detj_weight
        self.apear_scale = apear_scale
        self.ndims=ndims
        self.max_sz = torch.reshape(torch.tensor([0.7]*ndims, dtype=torch.float32) , [1]+[ndims]+[1]*(ndims))
        self.act = torch.nn.ReLU(inplace=True)
        self.dist=dist
        self.sign=sign

    def _diffs(self, y,dist=None):
        if dist is None:
            dist=self.dist
        # vol_shape = y.size()[2:]
        # vol_shape = y.get_shape().as_list()[1:-1]
        # ndims = len(vol_shape)

        df = [None] * self.ndims
        for i in range(self.ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, self.ndims + 2)]
            y = y.permute(r)
            dfi = y[dist:, ...] - y[:-dist, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, self.ndims + 2)]
            df[i] = dfi.permute(r)
        return df

    def _eq_diffs(self, y,dist=None):
        if dist is None:
            dist=self.dist
        # vol_shape = y.get_shape().as_list()[1:-1]
        vol_shape = y.size()[2:]
        ndims = len(vol_shape)
        pad = [0, 0] * (ndims + 1) +[dist, 0]
        pad1 = [0, 0] * (ndims + 1) +[0, dist]
        df = [None, None] * ndims
        for i in range(ndims):
            # df_full = tf.zeros_like(y, dtype=float)
            d = i + 2
            # permute dimensions to put the ith dimension first
            # r = [d, *range(d), *range(d + 1, ndims + 2)]
            # y = K.permute_dimensions(y, r)
            yt = y.permute([d, *range(d), *range(d + 1, ndims + 2)])
            # dfi = y[1:, ...] - y[:-1, ...]
            # df_full=tf.pad(yt[1:, ...] - yt[:-1, ...], pad)
            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            # r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            # df[i] = K.permute_dimensions(df_full, r)
            dy=yt[dist:, ...] - yt[:-dist, ...]
            df[2*i] = (F.pad(dy, pad,mode='constant',value=0)).permute([*range(1, d + 1), 0, *range(d + 1, ndims + 2)])
            df[2*i+1] = (F.pad(dy, pad1, mode='constant', value=0)).permute([*range(1, d + 1), 0, *range(d + 1, ndims + 2)])
        return df

    def _outl_dist(self, y):
        self.device = y.device
        vol_shape = y.size()[2:]
        self.max_sz=self.max_sz.to(self.device)
        # self.act.to(self.device)
        # [h, w] = x.size()[2:]
        # [h,w]=img_sz
        # self.img_sz = torch.reshape((torch.tensor(vol_shape, device=self.device) - 1) / 2., [1, 1, 1, 2])
        # ndims = len(vol_shape)
        select_loc = [s // 2 for s in vol_shape]
        # disp=y[select_loc]

        # disp=tf.gather_nd(y,)
        if self.ndims==3:
            return torch.mean(self.act(torch.abs(y[:,:, select_loc[0], select_loc[1], select_loc[2]]) - self.max_sz))
        elif self.ndims == 2:
            return torch.mean(self.act(torch.abs(y[:, :, select_loc[0], select_loc[1]]) - self.max_sz))

        # return tf.reduce_mean(act(tf.abs(y[:,select_loc[:],:])-tf.expand_dims(y.get_shape()[1:-1], 0)))

    def _eval_detJ(self, disp=None, weight=None):
        # ndims = 3
        # label = vol2 > thresh
        # label=vol2[...,0]>thresh
        # label=np.ones_like(vol2[...,0])
        # a=np.stack(np.gradient(disp,axis=[-2,-3,-4]),-1)
        # b=np.sum(label)
        # rescale_factor = 2
        # disp = zoom(disp, [rescale_factor] * ndims + [1], mode='nearest')
        # label=zoom(label, rescale_factor, mode='nearest')
        # a=np.stack(np.gradient(disp,axis=[-2,-3,-4]),-1)
        # b = np.linalg.det(a)
        # weight = 1 if weight is None else weight[...,0]
        weight = 1
        torch.stack(disp, -1)
        if self.ndims==3:
            detj = (disp[0][..., 0] * disp[1][..., 1] * disp[2][..., 2]) + (
                    disp[0][..., 1] * disp[1][..., 2] * disp[2][..., 0]) + (
                           disp[0][..., 2] * disp[1][..., 0] * disp[2][..., 1]) - (
                           disp[0][..., 2] * disp[1][..., 1] * disp[2][..., 0]) - (
                           disp[0][..., 0] * disp[1][..., 2] * disp[2][..., 1]) - (
                           disp[0][..., 1] * disp[1][..., 0] * disp[2][..., 2])
        elif self.ndims==2:
            detj = (disp[0][..., 0] * disp[1][..., 1]) - (disp[0][..., 1] * disp[1][..., 0])

        return torch.mean(torch.nn.ReLU()(-detj) * weight)
        # weight = 1 if weight is None else weight
        # return tf.reduce_mean(tf.nn.relu(-tf.linalg.det(tf.stack(disp, -1))) * weight)

    def forward(self,  y_pred=None,x=None, img=None):
        reg_loss = 0
        if img is None:
            # if self.penalty == 'l1':
            if 'l1' in self.penalty:
                df = [torch.mean(torch.abs(f)) for f in self._diffs(y_pred)]
                reg_loss += sum(df) / len(df)
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
            # elif self.penalty == 'l2':
            if 'l2' in self.penalty:
                # assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
                df = [torch.mean(f * f) for f in self._diffs(y_pred)]
                reg_loss += sum(df) / len(df)
                # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
        # else:
        #     dg = tf.exp(-self.apear_scale * tf.add_n(
        #         [tf.reduce_sum(g * g, axis=-1, keepdims=True) for g in self._eq_diffs(img)]) / tf.reduce_sum(
        #         tf.square(.2 + img), axis=-1, keepdims=True))
        #     if 'l1' in self.penalty:
        #         df = [tf.reduce_mean(tf.abs(f) * dg) for f in self._eq_diffs(y_pred)]
        #         reg_loss += tf.add_n(df) / len(df)
        #         # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)
        #     # elif self.penalty == 'l2':
        #     if 'l2' in self.penalty:
        #         # assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
        #         df = [tf.reduce_mean(f * f * dg) for f in self._eq_diffs(y_pred)]
        #         reg_loss += tf.add_n(df) / len(df)
        #         # return (tf.add_n(df) / len(df)) + self.outrange_weight * self._outl_dist(y_pred)

        if 'detj' in self.penalty:
            df = self.detj_weight*self._eval_detJ(self._eq_diffs(y_pred))  # , dg[...,0])
            reg_loss += df  # 0.5*df
        if 'range' in self.penalty:
            reg_loss += self.outrange_weight * self._outl_dist(y_pred)
        if 'param' in self.penalty:
            # diff=self._diffs(y_pred)
            # self.dist = 30
            mean_dim=list(range(1, self.ndims + 2))
            dg = torch.exp(-self.apear_scale * torch.nn.ReLU(inplace=True)(.1-sum([torch.sum(g * g, dim=1, keepdim=True) for g in self._eq_diffs(img,dist=1)]) / torch.sum(torch.square(.1 + img), dim=1, keepdim=True)))
            dg = dg/(EPS+torch.mean(dg,dim=mean_dim,keepdim=True))

            # from torchvision.utils import save_image
            # img_save_pth = "data/images"
            # save_image(dg[0][0], img_save_pth + "/weight_map.png", nrow=5, normalize=True)

            # dg=1

            for id,d in enumerate(self.dist):
                # df = [torch.mean(torch.abs(x[:,id,...]-torch.mean(torch.abs(f),dim=mean_dim,keepdim=True))) for f in self._eq_diffs(y_pred,dist=d)]
                df= torch.mean(torch.abs(sum([torch.mean(torch.abs(f),dim=mean_dim,keepdim=True) for f in self._eq_diffs(y_pred,dist=d)])-torch.abs(x[:,id,...]))*dg)
                reg_loss += 1 * (df)/len(self.dist)
            # self.dist = 50
            # df = sum([torch.mean(torch.abs(f), dim=list(range(1, self.ndims + 2)),keepdim=True) for f in self._diffs(y_pred,dist=self.dist[1])])
            # reg_loss += self.sign * torch.mean(df - x[:, 1, ...])

        return reg_loss




class SED(torch.nn.Module):
    def __init__(self,kern_rad=2,ndims=1,channel=20,ceter_scale=5):
        super(SED, self).__init__()
        self.channel=channel
        self.kern_rad=kern_rad
        self.ceter_scale = ceter_scale
        self.kernel=self._build_kernel()
        self.conv=getattr(F,'conv%dd' % ndims)
        self.thresh = 1/channel/kern_rad/2

    def _build_kernel(self):
        # kernel=np.array(list(range(1,self.kern_rad))+list(range(self.kern_rad,0,-1)))
        kernel=np.array(list(range(1,self.kern_rad-1))+[self.kern_rad*self.ceter_scale]+list(range(self.kern_rad,0,-1)))
        kernel=kernel/np.sum(kernel)
        return torch.tensor(kernel,dtype=torch.float32).view([1,1,2*self.kern_rad-1])

    def _compensate_onehot(self, seq):
        return torch.cat([seq,1-torch.sum(seq,dim=1)])
    def _smooth(self,seq,sharpness_scale=10):
        if sharpness_scale>0 and sharpness_scale is not None:
            seq=F.softmax(sharpness_scale*seq,dim=1)
        sz = seq.size()
        # a=self.conv(seq.view([sz[0]*sz[1],1,*sz[2:]]),self.kernel, bias=None, stride=1, padding=self.kern_rad-1, dilation=1, groups=1).view(sz)
        return self.conv(seq.reshape([sz[0]*sz[1],1,*sz[2:]]),self.kernel.to(seq.device), bias=None, stride=1, padding=self.kern_rad-1, dilation=1, groups=1).view(sz)

    def forward(self,seq1,seq2):
        # a=torch.mean(torch.sum(torch.abs(self._smooth(seq1) - self._smooth(seq2)), dim=1))
        # return torch.mean(torch.sum(torch.abs(self._smooth(seq1)-self._smooth(seq2)),dim=1))  #edited by jz 26/10/2022
        # return torch.sqrt(torch.mean(torch.sum(F.relu(torch.square(self._smooth(seq1) - self._smooth(seq2))-self.thresh**2,inplace=True), dim=1))) #edited by jz 26/10/2022
        gt_seq=self._smooth(seq2)
        # return torch.mean(torch.sum(F.relu(torch.abs(self._smooth(seq1) - gt_seq)-self.thresh**2,inplace=True), dim=1)/(torch.sum(gt_seq,dim=1)+1)) #edited by jz 26/10/2022
        return torch.mean(torch.square(torch.mean(
            torch.sum(F.relu(torch.abs(self._smooth(seq1) - gt_seq) - self.thresh ** 2, inplace=True), dim=1) / (
                        torch.sum(gt_seq, dim=1) + 1),dim=-1)))  # edited by jz 29/03/2023

class RMSE(torch.nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self,seq1,seq2):
        return torch.sqrt(self.mse(seq1, seq2))

class ClampMSELoss(torch.nn.Module):
    def __init__(self,min=0,max=10,loss=torch.nn.L1Loss,reduction='mean',*argv,**kwargs):
        super(ClampMSELoss, self).__init__()
        self.max=max
        self.min=min
        self.reduction=reduction
        self.loss= loss(reduction='none',*argv,**kwargs)
        return
    def forward(self, input=None, target=None, *argv,**kwargs):
        # loss=self.loss(torch.clamp(input,max=self.max),torch.clamp(target,max=self.max),*argv,**kwargs)
        loss=torch.where(target < self.max, self.loss(input, target,*argv,**kwargs), torch.nn.functional.relu(self.max - input))
        if self.reduction== None or self.reduction=='none':
            return loss
        else:
            return torch.mean(loss)

class MskCombLoss(torch.nn.Module):
    def __init__(self,loss_list=[torch.nn.BCELoss(reduction='none'),torch.nn.BCELoss(reduction='none'),ClampMSELoss(reduction=None)],weights=[1,1,1]):
        super(MskCombLoss, self).__init__()
        self.loss_list=loss_list
        self.weights=weights
        self.reduce_func=torch.mean
        self.eps=1e-7
        return
    def forward(self, input=[None], target=[None], mask=None, *argv,**kwargs):
        if mask is None:
            mask = [1]*len(input)
        losses=[w*self.reduce_func(m*loss(x,y))/(self.eps+(m if type(m) is int else self.reduce_func(m))) for loss,w,x,y,m in zip(self.loss_list,self.weights,input,target,mask)]
        # for loss,w,x,y,m in zip(self.loss_list,self.weights,input,target,mask):
        #     print(loss(x,y))
        #     print(m*loss(x,y))
        #     print(self.reduce_func(m*loss(x,y))/(self.eps+(m if type(m) is int else self.reduce_func(m))))
        return sum(losses)



if __name__=="__main__":
    import utils
    def one_hot(s):
        return torch.unsqueeze(torch.tensor(np.transpose(utils.one_hot_encoder(s=s)), dtype=torch.float32), dim=0)
    sed=SED()
    print(sed.kernel)
    seq="ACDEFGHIKLMNPQRSTVWYBXZJUO"
    seq_vec=one_hot(s=seq)
    print(seq)
    num=3
    # random editing
    # seq_edit = utils.random_edit_str(in_str=seq, num=num, skip_num=1)
    # print(seq)
    # print(seq_edit)
    # seq_edit_vec=one_hot(s=seq_edit)
    # print(sed(seq_vec,seq_edit_vec))

    # random deleting
    seq_edit = utils.random_edit_str(in_str=seq, num=num, skip_num=1, alphabet="")
    print(seq_edit)
    seq_del_vec=one_hot(s=seq_edit)
    # random inserting
    seq_edit = utils.random_edit_str(in_str=seq_edit, num=num, skip_num=0)
    print(seq_edit)
    seq_ins_vec=one_hot(s=seq_edit)

    seq_edit_vec1=one_hot(s=seq_edit)

    # random deleting
    seq_edit = utils.random_edit_str(in_str=seq_edit, num=num, skip_num=1, alphabet="")
    # random inserting
    seq_edit = utils.random_edit_str(in_str=seq_edit, num=num, skip_num=0)
    seq_edit_vec2=one_hot(s=seq_edit)

    print(sed(seq_vec,seq_vec))
    print(sed(seq_vec,seq_edit_vec1))
    print(sed(seq_vec,seq_edit_vec2))

    mskloss=MskCombLoss()
    input1=torch.rand(3, requires_grad=True)
    target1=torch.randint_like(input1,low=0,high=2)
    input2=input1
    target2=1-target1
    input3=torch.randn(3, requires_grad=True)*10
    target3 = torch.randn(3, requires_grad=True)*10

    l=mskloss(input=[input1,input2,input3],target=[target1,target2,target3],mask=[1,1-target1,target2])

