from torch import load as tload
from torchvision.io import read_image, ImageReadMode
from torch import argmax as targmax
from torch import max as tmax
from Neural_Network import Net
from torchvision.transforms import Normalize, Resize, CenterCrop
import streamlit as st

NUM_CATEGORIES = 43
IMG_SIZE = 32
NUM_CHANNELS = 3

label_map = {
        '0': '20 Speed',
        '1': '30 Speed',
        '2': '50 Speed',
        '3': '60 Speed',
        '4': '70 Speed',
        '5': '80 Speed',
        '6': '80 Lifted',
        '7': '100 Speed',
        '8': '120 Speed',
        '9': 'No Overtaking General',
        '10': 'No Overtaking Trucks',
        '11': 'Right of Way Crossing',
        '12': 'Right of Way General',
        '13': 'Give Way',
        '14': 'Stop',
        '15': 'No Way General',
        '16': 'No Way Trucks',
        '17': 'No Way One Way',
        '18': 'Attention General',
        '19': 'Attention Left Turn',
        '20': 'Attention Right Turn',
        '21': 'Attention Curvy',
        '22': 'Attention Bumpers',
        '23': 'Attention Slippery',
        '24': 'Attention Bottleneck',
        '25': 'Attention Construction',
        '26': 'Attention Traffic Light',
        '27': 'Attention Pedestrian',
        '28': 'Attention Children',
        '29': 'Attention Bikes',
        '30': 'Attention Snowflake',
        '31': 'Attention Deer',
        '32': 'Lifted General',
        '33': 'Turn Right',
        '34': 'Turn Left',
        '35': 'Turn Straight',
        '36': 'Turn Straight Right',
        '37': 'Turn Straight Left',
        '38': 'Turn Right Down',
        '39': 'Turn Left Down',
        '40': 'Turn Circle',
        '41': 'Lifted No Overtaking General',
        '42': 'Lifted No Overtaking Trucks'
    }

picture = st.file_uploader("Choose a picture of a handwritten digit:", type=['png', 'jpg'])

if picture:
    with open("uploaded.jpg", "wb") as f:
        f.write(picture.read())

    img = read_image("uploaded.jpg", ImageReadMode.RGB)
    c, h, w = img.shape

    if h <= w:
        img = CenterCrop(int(h))(img)
        st.image(img.permute(1, 2, 0).numpy(), width=300)
    else:
        img = CenterCrop(int(w))(img)
        st.image(img.permute(1, 2, 0).numpy(), width=300)

    img = Resize(size=(IMG_SIZE, IMG_SIZE))(img)

    img = img / 255
    img = Normalize(img.mean([1, 2]), img.std([1, 2]))(img)

    layers = [(IMG_SIZE ** 2) * NUM_CHANNELS, 4000, 1500, NUM_CATEGORIES]
    cnn_layers = [NUM_CHANNELS, 100, 150, 250]
    cnn_kernels = [5, 3, 3]
    model = Net(layers, cnn_layers, cnn_kernels, IMG_SIZE)
    model.load_state_dict(tload("model.pt"))
    model.eval()

    z = model(img.reshape(1, 3, IMG_SIZE, IMG_SIZE))

    st.write(f"I think this is '{label_map[str(int(targmax(z)))]}' sign with probility of {round(float(10 ** (tmax(z)) * 100), 2)} %.")
