# SPDX-License-Identifier: Apache-2.0

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="Qwen/Qwen3-4B")
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    # Add example params
    parser.add_argument("--chat-template-path", type=str)

    return parser


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    chat_template_path = args.pop("chat_template_path")

    # Create an LLM
    llm = LLM(**args)

    # Create sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k    

    def print_outputs(outputs):
        print("\nGenerated Outputs:\n" + "-" * 80)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}\n")
            print(f"Generated text: {generated_text!r}")
            print("-" * 80)

    print("=" * 80)

    # In this script, we demonstrate how to pass input to the chat method:
    conversation = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {
            "role": "user",
            "content": "Write an essay about the importance of higher education.",
        },
    ]
    outputs = llm.chat(conversation, sampling_params, use_tqdm=False)
    print_outputs(outputs)    


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    args["max_model_len"] = 8192
    args["max_tokens"] = 1024
    args["tensor_parallel_size"] = 2
    main(args)