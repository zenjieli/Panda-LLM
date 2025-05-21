import os.path as osp
from openai import OpenAI

def prepare_messages(query=None, img_filename=None):
    message = {"role": "user", "content": query}

    if img_filename:
        message["function_call"] = {"type": "image", "image": img_filename}

    return [message]


def predict_streaming():
    # create a request activating streaming response
    messages = prepare_messages()
    for chunk in client.chat.completions.create(model='Unknown',
                                                messages=messages,
                                                top_p=0.8,
                                                stream=True):
        if hasattr(chunk.choices[0].delta, 'content'):
            print(chunk.choices[0].delta.content, end='', flush=True)

    print()


def predict_nostream(query: str, img_filename: str):
    messages = prepare_messages(query, img_filename)
    response = client.chat.completions.create(model='Unknown', messages=messages, stream=False, temperature=0)

    return (response.choices[0].message.content) if len(response.choices) > 0 else ''


def upload_file_fast(filepath: str):
    from pathlib import Path

    response = client.files.create(file=Path(filepath).open("rb"), purpose="assistants")

    return response.filename


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8112)
    options = parser.parse_args()
    return options


if __name__ == '__main__':
    args = parse_arguments()

    client = OpenAI(base_url=f"http://{args.server_name}:{args.server_port}/v1", api_key="none")
    img_filename = upload_file_fast(osp.expanduser("~/Pictures/dolphin_screenshot.png"))
    prompt = "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction"
    print(predict_nostream(prompt + "\n" + "Click on \"Settings\" menu.", img_filename))
