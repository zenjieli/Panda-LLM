import os.path as osp

SUPPORTED_MODELS_TEXT = '<center><font size=2><b>Supported models</b>: Mistral, QWen, QWen-VL, Yi, MiniCPM-V </center>'

DOWNLOAD_MODEL_INSTRUCTION = 'Hugging Face username/model path, for instance: facebook/galactica-125m. To specify a branch, add it at the end after a \":\" character like this: facebook/galactica-125m:main.'

def model_text(model_path):
    return f"""\
        <p align="center">
            <table border="0" style="margin-left: auto; margin-right: auto;">
                <tr>
                    <td><img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Creative-Tail-Animal-panda.svg" style="height: 40px"/></td>
                    <td><font size=3>{osp.basename(model_path) if model_path else ''}</font></td>
                </tr>
            </table>
        </p>"""

    # return f'<center><font size=3>Model: {osp.basename(model_path)}</center>'