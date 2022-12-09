class Config():
    # General hyper-parameters
    lr = 5e-3
    n_epochs = 200
    momentum = 0.9
    log_interval = 10
    random_seed = 1
    weight_decay= 1e-4
    seed = 100
    batch_size = 64
    test_batch_size = 5000
    num_workers = 4
    train_split = 0.8
    val_split = 0.2
    image_size = 20
    patch_size = 4
    train_sigmoid_threshold = [0.1, 0.2, 0.3]
    shuffle_dataset = True
    shuffle = True

    # ViT hyper-parameters
    models = ['ViT']
    emb_dropout = 0
    dropout = 0
    mlp_expand = 4
    heads = 12
    depth = 6
    dim = 384
    mlp_dim = 1536
    pretrained = True
    pretrained_model_path = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\09_12_2022____19_09_57--\best_model_epoch74.pth'

    # Test hyper-parameters and settings
    test_model = 'ResNet18'
    test_dir = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\09_12_2022____18_12_39--'
    test_best_model = 'best_model_epoch64.pth'
    test_sigmoid_threshold = 0.2

