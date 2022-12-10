class Config():
    # General hyper-parameters
    lr = 5e-3
    n_epochs = 250
    momentum = 0.9
    log_interval = 10
    random_seed = 1
    weight_decay= 1e-4
    seed = 100
    batch_size = 32
    num_workers = 4
    train_split = 0.8
    val_split = 0.2
    image_size = 20
    patch_size = 4
    train_sigmoid_threshold = [0.1]
    shuffle_dataset = True
    shuffle = True
    warmup_epochs = 1
    log_save_interval = 5

    models = ['ResNet18']

    # ViT hyper-parameters
    emb_dropout = 0.1
    dropout = 0.1
    mlp_expand = 4
    heads = 6
    depth = 12
    dim = 384
    mlp_dim = 1536
    pretrained = True
    # pretrained_model_path = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\09_12_2022____22_10_40--\best_model_epoch5.pth'
    # pretrained_optimizer_path = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\09_12_2022____22_10_40--\optimizer.pth'
    # pretrained_scheduler_path = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\09_12_2022____22_10_40--\scheduler.pth'

    # Test hyper-parameters and settings
    test_model = 'ResNet50'
    test_dir = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\10_12_2022____09_38_51'
    test_best_model = 'model_epoch400.pth'
    test_sigmoid_threshold = 0.2
    test_batch_size = 5000

