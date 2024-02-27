import os.path as osp

HEADER_TEXT = """\
        <p align="center">
            <table border="0" style="margin-left: auto; margin-right: auto;">
                <tr>
                    <td><img src="https://upload.wikimedia.org/wikipedia/commons/f/fc/Creative-Tail-Animal-panda.svg" style="height: 80px"/></td>
                    <td><font size=6>Panda Chatbot</font></td>
                </tr>
            </table>
        </p>"""

SUPPORTED_MODELS_TEXT = '<center><font size=2><b>Supported models</b>: Mistral-GPTQ, QWen1.5-GPTQ, Yi-GPTQ, Yi-GGUF </center>'

def model_text(model_path):
    return f'<center><font size=3>Model: {osp.basename(model_path)}</center>'