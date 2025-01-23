from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='none')


def prepare_messages():
    query = 'Hi!'
    return [{'role': 'user', 'content': query}]


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


def predict_nostream():
    messages = prepare_messages()
    response = client.chat.completions.create(model='Unknown', messages=messages, stream=False, temperature=0)

    return (response.choices[0].message.content) if len(response.choices) > 0 else ''

if __name__ == '__main__':
    print(predict_nostream())
