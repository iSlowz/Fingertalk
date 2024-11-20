import os
from PIL import Image
import torch
from torch.nn.functional import softmax
from aprentissage.transfer_learning import load_model, load_index_to_class, get_transforms

index_to_class = load_index_to_class(os.path.join((os.path.dirname(__file__)), 'outputs/index_to_class.csv'))
num_classes = len(index_to_class)
model = load_model(os.path.join(os.path.dirname(__file__), 'outputs/model_accuracy98.pth'), num_classes)
test_transform = get_transforms(num_classes, 'test')


def predict_class(image):
    image = test_transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probas = softmax(output, dim=1)
        pred = probas.argmax(1).item()
        conf = probas[0][pred].item()
    classe = index_to_class[pred]
    return classe, conf


def testset_predictions(test_set='../ASL_dataset/asl_alphabet_test/asl_alphabet_test'):
    for img_name in os.listdir(test_set):
        true_classe = img_name.split('_')[0]
        img_path = os.path.join(test_set, img_name)
        img = Image.open(img_path)
        classe, conf = predict_class(img)
        conf = round(conf * 100, 1)

        GREEN = '\033[92m'
        RED = '\033[91m'
        END = '\033[0m'
        if true_classe == classe:
            color = GREEN
        else:
            color = RED
        print(f"{color}True classe: {true_classe}, Predicted classe: {classe}, confidence: {conf}%{END}")


if __name__ == '__main__':
    testset_predictions()
