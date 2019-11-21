class Config(object):
    def __init__(self):
        self.lr = 0.0001 # 0.000025 #0.00025
        self.data_dir = '../CaNet/VOCdevkit/VOC2012/'
        self.fold = 0
        self.checkpoint_dir = './checkpoint_Support2Conv/'
        self.gpu = '0, 1'
        self.train_batch_size = 32
        self.val_batch_size = 32
        self.IMG_MEAN = [0.485, 0.456, 0.406]
        self.IMG_STD = [0.229, 0.224, 0.225]
        self.num_epoch = 200
        self.input_size = (321, 321)
        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.power = 0.9
        self.category = [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle'],
                     ['bus', 'car', 'cat', 'chair', 'cow'],
                     ['diningtable', 'dog', 'horse', 'motorbike', 'person'], 
                     ['potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
                        ]
        
    