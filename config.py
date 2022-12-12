class Config():
    # General hyper-parameters
    lr = 5e-3
    n_epochs = 250
    momentum = 0.9
    log_interval = 10
    random_seed = 1
    weight_decay= 1e-4
    batch_size = 32
    num_workers = 4
    train_split = 0.8
    val_split = 0.2
    image_size = 20
    train_sigmoid_threshold = [0.1]
    shuffle_dataset = True
    warmup_epochs = 0
    step = 1
    log_save_interval = 5

    models = ['ResNet50',]

    # ViT hyper-parameters
    patch_size = 4
    emb_dropout = 0.1
    dropout = 0.1
    mlp_expand = 4
    heads = 6
    depth = 12
    dim = 384
    mlp_dim = 1536
    pretrained = True

    # Test hyper-parameters and settings
    test_model = 'ResNet50'
    test_dir = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\11_12_2022____20_04_50'
    test_best_model = 'Best_ResNet50_0.1_228.pth'
    test_sigmoid_threshold = 0.1
    test_batch_size = 5000

