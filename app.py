from PIL import Image
from io import BytesIO
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import shutil
import pandas as pd
import imageio
import streamlit as st
import os
from datetime import datetime
from zipfile import ZipFile
# returns current date and time
now = datetime.now()



# Streamlit configuration
st.set_page_config(page_title='Super Resolution')
hide_menu_style = """
    <style>
    @MainMenu {visibility: hidden; }
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)
# Rest of dependancies
buf = BytesIO()  # Setting up BytesIO to download results


# Global variables and directroies
input_folder = 'Real-ESRGAN/inputs/'
result_folder = 'Real-ESRGAN/results/'
model_folder = './Real-ESRGAN/'
home = str(os.getcwd())  # Backup of Home Directory Path, we'll need it!


def clear():  # for emptying inputs and results folder
  os.chdir('Real-ESRGAN/')
  if os.path.exists('inputs/') == True:
    shutil.rmtree('inputs/')
    os.mkdir('inputs/')
  else:
    os.mkdir('inputs/')

  if os.path.exists('results/') == True:
    shutil.rmtree('results/')
    os.mkdir('results/')
  else:
    os.mkdir('results/')
  os.chdir(home)


# Setting up pretrained model


def load_model():
  if os.path.exists(model_folder) == True:
    shutil.rmtree(model_folder)
  elif os.path.exists(model_folder) == False:
    # os.system ('pip install -r requirements.txt')
    os.system("git clone https://github.com/xinntao/Real-ESRGAN.git")
    st.write('Loading model')
    os.chdir('Real-ESRGAN/'); os.system('python setup.py develop')
    os.system('wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models')
    st.write('Model Loaded')
    os.chdir(home)
    if os.path.exists(input_folder) == True:
      shutil.rmtree(input_folder)
      os.mkdir(input_folder)
    os.mkdir(result_folder)


# Localising files
if (st.button('Load/Reinstall model')):
  load_model()
if (st.button('Clear inputs and results')):
  clear()

# Caching image for code reruns (works better outside colab)


@st.cache
def load_image(image_file):
  img = Image.open(image_file)
  return img


st.title("Super Resolution")
# Saving uploaded image in input folder for processing


def save_image(image_file):
  if image_file is not None:
    filename = image_file.name
    img = load_image(image_file)
    st.image(image=img, width=None)
    with open(os.path.join(input_folder, filename), "wb") as f:
      f.write(image_file.getbuffer())
      st.success("Succesfully uploaded file for processing".format(filename))


# Taking input image from the user
image_file = st.file_uploader(
    "Upload Image", type=['png', 'jpeg', 'jpg', 'webp'])
save_image(image_file)

# Hyperparameter Tuning
# Necessary (Default: 4, but 3.5 worked better with me)
scale = st.number_input('Outscale', value=3.5)
# Whether to use half precision during inference
half = st.radio('Half Precision', options=('On', 'Off'))
if half == 'On':
  half = ''
else:
  half = '--fp32'
# Whether to use GFPGAN to enhance face
face = st.radio('GFPGAN face enhance', options=('On', 'Off'))
if face == 'On':
  face = '--face_enhance'
else:
  face = ''
# optional: Horizontal Radio buttons
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


# Runnning the model and upscalling user input

if (st.button('Enhance')):
  filesinputlist = []
  # os.chdir('Real-ESRGAN/')
  for filename in os.listdir('Real-ESRGAN/inputs/'):
    if os.path.isfile(f"{filename.lower()}_out.") not in os.listdir('Real-ESRGAN/results/'):
      os.system(
          "python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i Real-ESRGAN/inputs/{0} -o Real-ESRGAN/results --outscale {1} {2} {3}".format(filename,scale, half, face))
    else:
      # skip file it already exists in output (unless specfied not to)
      continue
  for filename in os.listdir('Real-ESRGAN/results/'):
    if '_out.' in filename.lower():
      target = filename
      filesinputlist.append(filename)
      if '.jpg' in target.lower():
        format = 'JPEG'
      elif '.jpeg' in target.lower():
        format = 'JPEG'
      elif '.png' in target.lower():
        format = 'PNG'
      else:
        format = 'webp'

  # First file in list
  res = load_image(f'Real-ESRGAN/results/{filesinputlist[0]}')
  st.image(res, width=None)
  res.save(buf, format=format)
  byte_img = buf.getvalue()
  with ZipFile(f'Output {filesinputlist[0]}.zip', 'w') as zipObj2:
    for i in os.listdir("Real-ESRGAN/results/"):
      zipObj2.write(f"Real-ESRGAN/results/{i}")
    # if (st.download_button(label='Download Upscaled Image', data=zipObj2)):
    #   st.success('Image Saved!')


  with open(f'Output {filesinputlist[0]}.zip', "rb") as fp:
      btn = st.download_button(
          label="Download ZIP",
          data=fp,
          file_name=f"Output {now}.zip",
          mime="application/zip")
