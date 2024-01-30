import imageio.v3 as iio
import base64
import openai
import os
from eurekaplus.utils.extract_task_code import file_to_string


def video_to_frames(video_file):
    file_dir = os.path.dirname(video_file)
    frames = iio.imread(video_file)
    start_frame, end_frame = frames[5], frames[-1]
    start_frame_path, end_frame_path = (
        f"{file_dir}/start_frame.png",
        f"{file_dir}/end_frame.png",
    )
    iio.imwrite(start_frame_path, start_frame)
    iio.imwrite(end_frame_path, end_frame)
    paths = [start_frame_path, end_frame_path]
    return paths


class BehaviorCaptioner:
    def __init__(self, init_sys_prompt, model="gpt-4-vision-preview") -> None:
        self.init_sys_prompt = file_to_string( init_sys_prompt)
        self.model = model

    def describe(self, image_paths, task: str = ""):
        image_contents = [
            self.make_image_content(image_path) for image_path in image_paths
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.init_sys_prompt},
                    ],
                },
                {"role": "user", "content": [*image_contents, task]},
            ],
            max_tokens=4096,
            n=1,
        )
        msg = response.choices[0]
        return msg

    def conclude(self, image_paths, task: str = ""):
        msg = self.describe(image_paths=image_paths, task=task)
        description = msg["message"]["content"]
        succ = None
        if "FAIL" in description:
            succ = True
        elif "SUCCESS" in description:
            succ = False
        return succ

    def make_image_content(self, image_path):
        image_base64 = self._encode_image(image_path)
        return self._wrap_image_content(image_base64)

    def _wrap_image_content(self, image_base64):
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        }

    # See https://platform.openai.com/docs/guides/vision
    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

