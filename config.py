class Config():
    # General hyper-parameters
    lr = 5e-3
    n_epochs = 200
    momentum = 0.9
    log_interval = 10
    random_seed = 1
    weight_decay= 1e-4
    seed = 100
    batch_size = 16
    test_batch_size = 5000
    num_workers = 4
    train_split = 0.8
    val_split = 0.2
    image_size = 20
    patch_size = 4
    test_sigmoid_threshold = 0.2
    train_sigmoid_threshold = [0.2, 0.5, 0.8]
    shuffle_dataset = True
    shuffle = True

    # cometml settings
    api_key = "c3qk7qgByuTxwIjTdsG261cy9"
    project_name = "face/non_face"
    workspace = "noamberg"
    models = ['ResNet50','ViT']

    # model settings
    emb_dropout = 0
    dropout = 0
    mlp_expand = 4
    heads = 12
    depth = 6
    dim = 384
    mlp_dim = 1536

    test_dir = r'C:\Users\Noam\PycharmProjects\Jubaan\logs\08_12_2022____15_41_22'


