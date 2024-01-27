import imageio.v3 as iio
import base64
import requests
import openai
import os


def video_to_frames(video_file, save=True):
    file_dir = os.path.dirname(video_file)
    frames = iio.imread(video_file)
    start_frame, end_frame = frames[0], frames[-1]
    return start_frame, end_frame


def gptv_call(frames, model="gpt-4-vision-preview"):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    print(response.choices[0])


# See https://platform.openai.com/docs/guides/vision
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# video_to_frames()