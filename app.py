import os
import uuid
import torch
import cv2
import replicate
import requests
import PIL
import gradio as gr

from PIL import Image
from io import BytesIO
import numpy as np
from fastapi import FastAPI
from torch import autocast

from matplotlib import pyplot as plt
from torchvision import transforms
from clipseg.models.clipseg import CLIPDensePredT


CUSTOM_PATH = "/gradio"
app = FastAPI()


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

os.environ["REPLICATE_API_TOKEN"] = "9b398c46835c17d59b6ba9e138594692bec10e2e"
pipe = replicate.models.get("stability-ai/stable-diffusion-inpainting")
version = pipe.versions.get("c28b92a7ecd66eee4aefcd8a94eb9e7f6c3805d5f06038165407fb5cb355ba67")

model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval()
model.load_state_dict(torch.load('./clipseg/weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)

transform = transforms.Compose(
    [
    
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      transforms.Resize((512, 512)),
])

def predict(radio, dict, word_mask, prompt=""):
    if(radio == "draw a mask above"):
        with autocast("cpu"):
            init_image = dict["image"].convert("RGB").resize((512, 512))
            mask = dict["mask"].convert("RGB").resize((512, 512))
    else:
        img = transform(dict["image"]).unsqueeze(0)
        word_masks = [word_mask]

        with torch.no_grad():

            preds = model(img.repeat(len(word_masks),1,1,1), word_masks)[0]

        init_image = dict['image'].convert('RGB').resize((512, 512))

        filename = f"{uuid.uuid4()}.png"
        plt.imsave(filename,torch.sigmoid(preds[0][0]))
        img2 = cv2.imread(filename)

        gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
        mask = Image.fromarray(np.uint8(bw_image)).convert('RGB')
        os.remove(filename)

    
    mask.save("mask_image.png")
    init_image.save("init_image2.png")

    input_requests = {
                    
                    'prompt': str(prompt),
                    'image': open("init_image2.png", "rb"),
                    'mask': open("mask_image.png", "rb")
                    
                    }

    output = version.predict(**input_requests)
    print("Output =  ", output)
    res = str(output)[1:-1]
    a2 = eval(res)
    # and saves it in a variable
    data = requests.get(a2).content
    f = open('img.jpg','wb')
    # Storing the image data inside the data variable to the file
    f.write(data)
    f.close()
    
    # Opening the saved image and displaying it
    img = Image.open('img.jpg')

    return img
# examples = [[dict(image="init_image.png", mask="mask_image.png"), "A panda sitting on a bench"]]

css = ''''''

def swap_word_mask(radio_option):
    if(radio_option == "type what to mask below"):
        return gr.update(interactive=True, placeholder="A cat")
    else:
        return gr.update(interactive=False, placeholder="Disabled")

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(
        """
        """
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload").style(height=400)
            with gr.Box(elem_id="mask_radio").style(border=False):
                radio = gr.Radio(["draw a mask above", "type what to mask below"], value="draw a mask above", show_label=False, interactive=True).style(container=False)
                word_mask = gr.Textbox(label = "What to find in your image", interactive=False, elem_id="word_mask", placeholder="Disabled").style(container=False)
            prompt = gr.Textbox(label = 'Your prompt (what you want to add in place of what you are removing)')
            radio.change(fn=swap_word_mask, inputs=radio, outputs=word_mask,show_progress=False)
            radio.change(None, inputs=[], outputs=image_blocks, _js = """
            """)
            btn = gr.Button("Run")
        with gr.Column():
            result = gr.Image(label="Result")
        btn.click(fn=predict, inputs=[radio, image, word_mask, prompt], outputs=result)
    gr.HTML(
            """
           """
        )
# demo.launch()

app = gr.mount_gradio_app(app, image_blocks, path=CUSTOM_PATH)
