'''
#推理部分
'''
from mindspore import load_checkpoint, load_param_into_net, Tensor
from codecccnn import Cyclecolornet, SLoss, nn, Model, Imgdataset, ds, load_data_path

left_path = './data/color'
right_path = './data/denoise'
split_ratio = 0.8
ckpt_file_name = "./model/CURTransformer.ckpt"
param_set = load_checkpoint(ckpt_file_name)
net = CURTransformer(
        hidden_dim=32,
        layers=(2, 2, 6, 2),
        heads=1,
        channels=32,
        num_classes=3,
        head_dim=32,
        window_size=14,
        downscaling_factors=1,
        relative_pos_embedding=True
    )
load_param_into_net(net, param_set)
loss3 = SLoss()
tra_data, val_data = load_data_path(left_path, right_path, split_ratio)
optim = nn.RMSProp(params=net.trainable_params(), learning_rate=0.001)
model = Model(net, loss_fn=loss3, optimizer=optim)

test_dataset = Imgdataset(val_data)
testdataset = ds.GeneratorDataset(test_dataset,
                                  column_names=['img', 'label'],
                                  num_parallel_workers=4)
testdataset = testdataset.batch(1)

testdata_iter = testdataset.create_dict_iterator()
testdata = next(testdata_iter)
#print(Tensor(testdata['img']).shape)
predicted = model.predict(Tensor(testdata['img']))
predicted = predicted[0]
