import os
from PIL import Image
import torch
from transfer_learning import load_model, load_index_to_class, get_transforms

index_to_class = load_index_to_class('outputs/index_to_class.csv')
num_classes = len(index_to_class)
model = load_model('outputs/model_little_dataset.pth', num_classes)
test_transform = get_transforms(num_classes, 'test')

test_set = '../ASL_dataset/asl_alphabet_test/asl_alphabet_test'
for img in os.listdir(test_set):
    img_path = os.path.join(test_set, img)
    true_label = img.split('_')[0]  # get the label from the image name
    image = Image.open(img_path)
    image = test_transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        pred = output.argmax(1).item()
    classe = index_to_class[pred]

    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
    if true_label == classe:
        print(f"{GREEN}True label: {true_label}, Predicted label: {classe}{END}")
    else:
        print(f"{RED}True label: {true_label}, Predicted label: {classe}{END}")
