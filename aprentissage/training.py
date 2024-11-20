from aprentissage.transfer_learning import ResNet50_transfer_learning

dataset_path = '../ASL_dataset/asl_alphabet_train/asl_alphabet_train'
ResNet50_transfer_learning(
    dataset_path,
    num_epochs=7,
    learning_rate=0.001,
    batch_size=32
)