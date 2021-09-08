import torch
import torchvision.transforms as transforms
import model
from PIL import Image
import sys

DIR="data/models/"
MODEL="model-100-epochs-adam-0003-lr-cpu.pth"


def get_model(PATH, model):
    device = torch.device('cpu')
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    return model

def load_img(PATH):
    img = Image.open(PATH)
    img.load()
    return img

def load_apply_preprocessing(PATH):

    test_transforms = transforms.Compose([

                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5],
                                [0.5, 0.5, 0.5])])
    img = load_img(PATH)
    img = test_transforms(img)
    img = torch.unsqueeze(img, 0)
    return img

def predict(model, img):
    with torch.no_grad():
        pred = model(img)
    
    idx = pred.argmax()
    prob = torch.nn.functional.softmax(pred, dim=1)[0][idx].item()
    res = (f"Cat {prob}%") if pred.argmax()==0 else (f"Dog {prob}%")
    return res

if __name__ == "__main__":

    sample_img = sys.argv[1] #"data/cat.jpg"
    model = model.Classifier()
    model = get_model(DIR+MODEL, model)
    
    img = load_apply_preprocessing(sample_img)
    result = predict(model, img)
    print("Image:",sample_img," ---> ",result)