from openai import OpenAI

client = OpenAI(base_url='http://localhost:8000/v1', api_key='none')


# create a request activating streaming response
query = 'Hi!'
print(f'Input: {query}')
print('Output:', end=' ')
for chunk in client.chat.completions.create(model='Unknown',
                                            messages=[
                                                {'role': 'user', 'content': query}
                                            ],
                                            top_p=0.8,
                                            stream=True):
    if hasattr(chunk.choices[0].delta, 'content'):
        print(chunk.choices[0].delta.content, end='', flush=True)

print()