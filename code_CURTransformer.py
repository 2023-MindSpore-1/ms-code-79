'''
MODEL AND TRAIN
'''
#本模块用于读取数据，生成加载器，相关参数设置如下：

left_path='./data/rgb'
right_path='./data/noise'
split_ratio=0.8

import random
import cv2
import mindspore.dataset as ds
import os
import numpy as np
import glob
import mindspore as mds
import mindspore.nn as nn
import numpy as np
import mindspore.numpy as mdsnp
from mindspore import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
mds.context.set_context(device_target='Ascend',mode=mds.context.PYNATIVE_MODE)#  GRAPH_MODE

def load_data_path(left_path,right_path,split_ratio=0.03):
    #load data path
    left_images=os.listdir(left_path)
    right_images=os.listdir(right_path)
    left_images.sort()
    right_images.sort()
    left_images_path = sorted(glob.glob(left_path + "/*.png"))
    right_images_path = sorted(glob.glob(right_path + "/*.png"))
    #print(left_images)
    #print(right_images)
    #print(left_images)
    
    #left_images_path=[left_path+'/'+img for img in left_images ]
    #right_images_path=[right_path+'/'+img for img in right_images]
    
    #print(left_images_path[:10])
    #print(right_images_path[:10])
    
    #split data
    data_length=len(left_images_path)
    temp=[(l,r) for l,r in zip(left_images_path,right_images_path)]
    #print(temp[:10])  
    random.shuffle(temp)
    num_traindata=int(split_ratio*data_length)
    train_data_path=temp[0:num_traindata]
    #print(num_traindata)  638
    val_data_path=temp[num_traindata:]
    
    return train_data_path,val_data_path


            
class imgDataset():
    def __init__(self, tra):
        super(imgDataset, self).__init__()
        self.tra=tra
        
        #for data,label in enumerate(self.data_generator(self.tra)):
        #   self.imgs.append(data)
        #   self.labels.append(label)
        
    def __getitem__(self, index):
        
        return self.data_generator(self.tra[index])

    def __len__(self):
        return len(self.tra)
    
    def data_generator(self,data_path,is_train=True):
        #input data_path:list of tuple with (left,right)
        #output  dataset generator
        #
        #print('******************reading data*****************')
        left_img=cv2.imread(data_path[0])
        right_img=cv2.imread(data_path[1])
        return [left_img, right_img]
        
    # def _centerImage_(self,img):
    #     img = img.astype(np.float32)
    #     return img
    # def _rotateImage_(self,img):
    #     (h, w) = img.shape[:2]
    #     center=(w/2-0.5,h/2-0.5)
    #     M = cv2.getRotationMatrix2D(center, 180, 1.0)
    #     rotated = cv2.warpAffine(img, M, (w, h))
    #     return rotated
        
    # def _getGeometryFeat_(self,img_shape):
    #     H = img_shape[0]
    #     W = img_shape[1]
    #     feat = np.zeros((H,W,2))
    #     for j in range(H):
    #         for i in range(W):
    #             feat[j,i,0]=np.min([j-0,H-1-j])/(H-1)*1.0            
    #             feat[j,i,1]=np.min([i-0,W-1-i])/(W-1)*1.0
    #     return feat
from einops import rearrange, repeat

# import random
# import glob
# import os
# import cv2
import mindspore.dataset as ds
import mindspore as mds
import mindspore.nn as nn
import mindspore.numpy as mdsnp
from mindspore import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import numpy as np

class CyclicShift(nn.Cell):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def construct(self, x):
        return nn.Roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, **kwargs):
        #print('before res:',x.shape)
        x1 = self.fn(x, **kwargs)
        #print('after res:' , x1.shape)
        x1 = x1.asnumpy()
        x1 = rearrange(x1, 'b h_n w_n (c p1 p2) ->b c (h_n p1) (w_n p2)', p1=1, p2=1)
        x1 = mds.Tensor(x1)

        return x1 + x


class PreNorm(nn.Cell):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        shape1 = x.shape[1:]
        m = nn.LayerNorm(shape1)
        return m(x)


class Feedconstruct(nn.Cell):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.SequentialCell(
            [nn.Dense(dim, hidden_dim),
            nn.GELU(),
            nn.Dense(hidden_dim, dim)]
        )

    def construct(self, x):
        x = rearrange(x, 'b c (h_n p1) (w_n p2) ->b h_n w_n (c p1 p2)', p1=1, p2=1)
        return self.net(x)


def create_mask(window_size, displacement, upper_lower, left_right):
    print('mask')
    mask =  mds.Tensor((window_size ** 2, window_size ** 2))
    print('mask')
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = mds.Tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class PatchMerging(nn.Cell):
    def __init__(self, in_channels, out_channels=32, downscaling_factor=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1_1 = mds.ops.Conv2D(64, 3, pad_mode='same')
        self.conv2_1 = mds.ops.Conv2D(32, 3, pad_mode='same')
        self.conv3_1 = mds.ops.Conv2D(out_channels, 3, pad_mode='same')
        self.conv1_2 = mds.ops.Conv2D(64, 3, pad_mode='same')
        self.conv2_2 = mds.ops.Conv2D(32, 3, pad_mode='same')
        self.conv3_2 = mds.ops.Conv2D(out_channels, 3, pad_mode='same')
        self.conv1_3 = mds.ops.Conv2D(64, 3, pad_mode='same')
        self.conv2_3 = mds.ops.Conv2D(32, 3, pad_mode='same')
        self.conv3_3 = mds.ops.Conv2D(out_channels, 3, pad_mode='same')
        self.weight11 = mds.Tensor(np.ones([64, in_channels, 3, 3]), mds.float32)
        self.weight21 = mds.Tensor(np.ones([32, 64, 3, 3]), mds.float32)
        self.weight31 = mds.Tensor(np.ones([out_channels, 32, 3, 3]), mds.float32)
        self.weight12 = mds.Tensor(np.ones([64, in_channels, 3, 3]), mds.float32)
        self.weight22 = mds.Tensor(np.ones([32, 64, 3, 3]), mds.float32)
        self.weight32 = mds.Tensor(np.ones([out_channels, 32, 3, 3]), mds.float32)
        self.weight13 = mds.Tensor(np.ones([64, in_channels, 3, 3]), mds.float32)
        self.weight23 = mds.Tensor(np.ones([32, 64, 3, 3]), mds.float32)
        self.weight33 = mds.Tensor(np.ones([out_channels, 32, 3, 3]), mds.float32)
        self.patch_merge = nn.Unfold(ksizes=[1, downscaling_factor, downscaling_factor, 1], strides=[1,downscaling_factor, downscaling_factor, 1], rates=[1, downscaling_factor, downscaling_factor, 1])
#         self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        
        self.downscaling_factor = downscaling_factor
    def construct(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x1 = self.conv3_1(self.relu(self.conv2_1(self.relu(self.conv1_1(x,self.weight11)),self.weight21)),self.weight31)
        x2 = self.conv3_2(self.relu(self.conv2_2(self.relu(self.conv1_2(x,self.weight12)),self.weight22)),self.weight32)
        x3 = self.conv3_3(self.relu(self.conv2_3(self.relu(self.conv1_3(x,self.weight13)),self.weight23)),self.weight33)
        x = mds.ops.Concat(1)((x1,x2,x3))
        x = mds.ops.Transpose()(x, (0, 2, 3, 1))
        return x

class WindowAttention(nn.Cell):
    def __init__(self, in_channels, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads
        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=dim)

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.ParameterUpdate(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.ParameterUpdate(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        #self.to_qkv = nn.Dense(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = mds.ops.StandardNormal(2 * window_size - 1, 2 * window_size - 1)
        else:
            self.pos_embedding = mds.ops.StandardNormal(window_size ** 2, window_size ** 2)

        self.to_out = nn.Dense(inner_dim, dim)

    def construct(self, x):
        #经过cnn，得到qkv
        x = self.patch_partition(x)
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, ss, h = *x.shape, self.heads
        qkv = [x[:,:,:,:ss//3].asnumpy(), x[:,:,:,ss//3:ss//3*2].asnumpy(), x[:,:,:,ss//3*2:].asnumpy()]
#         qkv = x.chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = np.einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        q = mds.Tensor(q)
        k = mds.Tensor(k)
        v = mds.Tensor(v)
#         if self.relative_pos_embedding:
#             dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
#         else:
#             dots += self.pos_embedding

#         if self.shifted:
#             dots[:, :, -nw_w:] += self.upper_lower_mask
#             dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        dots = mds.Tensor(dots)
#         attn = dots.softmax(dim=-1)
        attn = mds.ops.Softmax()(dots)
        out = np.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)


        return out



class SwinBlock(nn.Cell):
    def __init__(self, in_channels, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(in_channels=in_channels,
                                                                     dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        #self.mlp_block = Residual(PreNorm(dim, Feedconstruct(dim=dim, hidden_dim=mlp_dim)))

    def construct(self, x):
        #print('before swinblock:',x.shape)
        x = self.attention_block(x)
        #print("after_attrntion:",x.shape)
        #x = self.mlp_block(x)
        #print("after_swin:", x.shape)
        return x





class StageCell(nn.Cell):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.downscaling_factor = downscaling_factor
        self.layers = nn.CellList([])
        for _ in range(layers // 2):
            self.layers.append(nn.CellList([
                SwinBlock(in_channels=in_channels,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(in_channels=in_channels,dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def construct(self, x):
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x
class merage(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = mds.ops.Conv2D(out_channels, 3, pad_mode='same')
        self.relu = nn.ReLU()
    def construct(self, x, y):
        x = mds.ops.Concat(1)(x, y)
        #x = torch.add(x, y).permute(0, 2, 3, 1)

        x = self.relu(self.conv(x))
        return x

class ResBlock(nn.Cell):

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = mds.ops.Conv2D(channels, 5, pad_mode='same')
        self.weight1 = mds.Tensor(np.ones([channels, 64, 5, 5]), mds.float32)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = mds.ops.Conv2D(channels, 5, pad_mode='same')
        self.weight2 = mds.Tensor(np.ones([channels, channels, 5, 5]), mds.float32)
        # self.bn2 = nn.BatchNorm2d(channels)

    def construct(self, x):
        residual = x

        out = self.conv1(x,self.weight1)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out,self.weight2)
        # out = self.bn2(out)

        out += residual
        # out = self.relu(out)

        return out

class Head(nn.Cell):
    """ Head consisting of convolution layers
    Extract features from corrupted images, mapping N3HW images into NCHW feature map.
    """

    def __init__(self, in_channels, out_channels=64, channels=32):
        super(Head, self).__init__()
        self.weight = mds.Tensor(np.ones([out_channels, 3, 3, 3]), mds.float32)
        self.conv1 = mds.ops.Conv2D(out_channels, 3, pad_mode='same')
        # self.bn1 = nn.BatchNorm2d(out_channels) if task_id in [0, 1, 5] else nn.Identity()
        # self.relu = nn.ReLU()
        self.resblock1 = ResBlock(out_channels)
        self.resblock2 = ResBlock(out_channels)
        self.conv2 = mds.ops.Conv2D(channels, 1, pad_mode='same')
        self.weight2 = mds.Tensor(np.ones([channels, out_channels, 1, 1]), mds.float32)

    def construct(self, x):
        out = self.conv1(x, self.weight)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.conv2(out,self.weight2)
        return out

class CURTransformer(nn.Cell):
    def __init__(self, *, hidden_dim, layers, heads, channels=32, num_classes=1000, head_dim=8, window_size=28,
                 downscaling_factors=1, relative_pos_embedding=False,scale_factor=0):
        super().__init__()
        self.sr = scale_factor
        self.stage1 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.merage1 = merage(in_channels=channels * 2, out_channels=channels)
        self.merage2 = merage(in_channels=channels * 2, out_channels=channels)
        self.merage3 = merage(in_channels=channels * 2, out_channels=channels)
        self.up_stage1 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.up_stage2 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.up_stage3 = StageCell(in_channels=channels, hidden_dimension=hidden_dim, layers=2,
                                  downscaling_factor=downscaling_factors, num_heads=heads, head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.headsets = Head(3, 64, channels)
        self.tailsets = mds.ops.Conv2D(16, 3, pad_mode='same')
        #self.mlp_head = nn.Dense(96, 48)

        # up-sampling
        #assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([mds.ops.Conv2D(hidden_dim,hidden_dim* (2 ** 2), kernel_size=3, padding=1),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # elif scale_factor == 3 :
        #     self.upscale = nn.Sequential(
        #         mds.ops.Conv2D(hidden_dim, hidden_dim * (scale_factor ** 2), kernel_size=3, padding=1),
        #         nn.PixelShuffle(scale_factor)
        #     )

        self.conv2 = mds.ops.Conv2D(3, 3, pad_mode='same')

    def construct(self, img):
        # return img
        #b 3 hw->b 16 hw
        record = img
        x = self.headsets(img)
        #print(x.shape)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        #print(x1.shape, x2.shape, x3.shape, x4.shape, x.shape)
        x = self.merage1(x4, x3)
        #print(x.shape)
        x = self.up_stage1(x)
        #print(x.shape)
        #print(x.shape)
        x = self.merage2(x, x2)
        #print(x.shape)
        x = self.up_stage2(x)
        #print(x.shape)
        x = self.merage3(x, x1)

        x = self.up_stage3(x)
        # if self.sr != 0:
        #     x = self.upscale(x)
        x = self.conv2(x)
        return x+img


###加载数据  设置batchsize
# yern
#print(tra)
tra,val=load_data_path(left_path,right_path,split_ratio)
print(tra[:2],len(tra))
dataset=imgDataset(tra)
#traindataset=ds.GeneratorDataset(data_generator(tra),column_names=['img','label'],num_parallel_workers=4)
traindataset=ds.GeneratorDataset(dataset,column_names=['img','label'],num_parallel_workers=4)
print(1)
traindataset=traindataset.batch(1)
print(2)
NET=CURTransformer(
        hidden_dim=32,
        layers=(2, 2, 6, 2),
        heads=1,
        channels=32,
        num_classes=3,
        head_dim=32,
        window_size=14,
        downscaling_factors=1,
        relative_pos_embedding=False
    )
print(3)
input_x = mds.Tensor(np.ones([1, 3, 224, 224]), mds.float32)
print(NET(input_x).shape)


class MISS_1(nn.Cell):
    def __init__(self):
        super(MISS_1,self).__init__()
        self.lossfn=nn.SSIM()
    def construct(self,data,label):
        ssim=self.lossfn(data,label)
        ones=mdsnp.full_like(ssim,1)
        return ones-ssim
class SLoss(nn.Cell):
    def __init__(self, base_num_filter=8):
        super(SLoss, self).__init__()
        self.exp = ops.Exp()
#         self.conv2d = nn.Conv2d(in_channels = 1, out_channels = 3, kernel_size = 3)

    def construct(self, pred, true):
        loss = self.getloss(pred, true)
        return loss

    def getloss(self, pred, true):
        
        y_true = true[:, 0:6, 10:-10, 10:-10]
        y_pred = pred[:, 0:6, 10:-10, 10:-10]
        y_true_V = y_true[:, 0:1, :, :]
        y_true_U = y_true[:, 1:2, :, :]
        y_true_Y = y_true[:, 2:3, :, :]
        y_true_reverse_V = y_true[:, 3:4, :, :]
        y_true_reverse_U = y_true[:, 4:5, :, :]
        y_true_reverse_Y = y_true[:, 5:6, :, :]

        y_pred_V = y_pred[:, 0:1, :, :]
        y_pred_U = y_pred[:, 1:2, :, :]
        y_pred_Y = y_pred[:, 2:3, :, :]
        y_pred_reverse_V = y_pred[:, 3:4, :, :]
        y_pred_reverse_U = y_pred[:, 4:5, :, :]
        y_pred_reverse_Y = y_pred[:, 5:6, :, :]
        
        #print(true)
        #print(y_pred)
        #ssim1=1
        ssim1 = self.tf_ssim011(y_pred_Y, y_true_Y, max_val=255.0)
        ssim2 = self.tf_ssim(y_pred_reverse_V, y_true_reverse_V, max_val=255.0)
        ssim3 = self.tf_ssim(y_pred_reverse_U, y_true_reverse_U, max_val=255.0)

        ssim = (ssim1 + ssim2 + ssim3) / 3.0
        return 1 - ssim

    def tf_ssim(self,img1, img2, max_val=1, cs_map=False, mean_metric=True):
        K1 = 0.01
        K2 = 0.03
        L = max_val  # depth of image (255 in case the image has a different scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        mu1 = self.conv2d(img1)
        mu2 = self.conv2d(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2
        if cs_map:
            value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                  (sigma1_sq + sigma2_sq + C2)),
                     (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                 (sigma1_sq + sigma2_sq + C2))
        reducemean = ops.ReduceMean()
        if mean_metric:
            value = reducemean(value)
        return value
    
    def tf_ssim011(self, img1, img2, max_val=1, mean_metric=True):
        K1 = 0.01
        K2 = 0.03
        L = max_val  # depth of image (255 in case the image has a different scale)
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2
        #print(img1)
        mu1 = self.conv2d(img1)
        mu2 = self.conv2d(img2)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.conv2d(img1 * img1) - mu1_sq
        sigma2_sq = self.conv2d(img2 * img2) - mu2_sq
        sigma12 = self.conv2d(img1 * img2) - mu1_mu2
        value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        if mean_metric:
            value = value.mean()
        return value
    
testloss = SLoss()

###training 
#Loss=MISS_1()
Loss2=SLoss()
trainDataset=imgDataset(tra)
trainData=ds.GeneratorDataset(trainDataset,column_names=['img','label'],num_parallel_workers=1)
trainData=trainData.batch(1)

optim=nn.RMSProp(params=NET.trainable_params(), learning_rate=0.001)
trainnet=Model(NET,loss_fn=Loss2,optimizer=optim)
loss_cb = LossMonitor(per_print_times=1)
ckpt_config = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
ckpoint_cb = ModelCheckpoint(prefix='coloring', directory='./model', config=ckpt_config)
print('start    training')
trainnet.train(8,trainData)