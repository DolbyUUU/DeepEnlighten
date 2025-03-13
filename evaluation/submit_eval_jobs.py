import os
import argparse

configs = [
    {
        'output-dir': "outputs/llama3binstruct-step1000",
        'model-path': "../checkpoints/DeepEnlighten/Llama3BInstruct-SocialIQa-REINFORCE++-4RTX4090-202503111620/actor/global_step_1000",
        'tokenizer-path': "../checkpoints/DeepEnlighten/Llama3BInstruct-SocialIQa-REINFORCE++-4RTX4090-202503111620/actor/global_step_1000",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-deepenlighten-llama3binstruct-step1000'
    },
    {
        'output-dir': "outputs/llama3binstruct",
        'model-path': "meta-llama/Llama-3.2-3B-Instruct",
        'tokenizer-path': "meta-llama/Llama-3.2-3B-Instruct",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-vanilla-llama3binstruct'
    },
    {
        'output-dir': "outputs/qwen3binstruct-step1000",
        'model-path': "../checkpoints/DeepEnlighten/Qwen3BInstruct-SocialIQa-REINFORCE++-4RTX4090-202503120726/actor/global_step_1000",
        'tokenizer-path': "../checkpoints/DeepEnlighten/Qwen3BInstruct-SocialIQa-REINFORCE++-4RTX4090-202503120726/actor/global_step_1000",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-deepenlighten-qwen3binstruct-step1000'
    },
    {
        'output-dir': "outputs/qwen3binstruct",
        'model-path': "Qwen/Qwen2.5-3B-Instruct",
        'tokenizer-path': "Qwen/Qwen2.5-3B-Instruct",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-vanilla-qwen3binstruct'
    },
    {
        'output-dir': "outputs/qwen3b-step1000",
        'model-path': "../checkpoints/DeepEnlighten/Qwen3B-SocialIQa-REINFORCE++-4RTX4090-202503121336/actor/global_step_1000",
        'tokenizer-path': "../checkpoints/DeepEnlighten/Qwen3B-SocialIQa-REINFORCE++-4RTX4090-202503121336/actor/global_step_1000",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-deepenlighten-qwen3b-step1000'
    },
    {
        'output-dir': "outputs/qwen3b",
        'model-path': "Qwen/Qwen2.5-3B",
        'tokenizer-path': "Qwen/Qwen2.5-3B",
        'model-size': "3b",
        'overwrite': True,
        'use-vllm': True,
        'no-markup-question': False,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-vanilla-qwen3b'
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int ,default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-gpus", type=int, default=8)
    args = parser.parse_args()

    # 遍历所有配置并运行对应的实验
    for conf in configs:
        cmd = "python run_subset_parallel.py"
        for key, val in conf.items():
            if key == 'expname':  # 跳过 expname，因为它仅用于标识实验
                continue
            if isinstance(val, str):
                cmd += f" --{key} {val}"
            elif val:
                cmd += f" --{key}"
        cmd += f" --test-conf {conf['test-conf']}"
        cmd += f" --n-repeats {args.n_repeats}"
        cmd += f" --temperature {args.temperature}"
        cmd += f" --ngpus {args.n_gpus}"
        cmd += f" --rank {0} &"
        
        # 打印并执行命令
        print(f"Running experiment: {conf['expname']}")
        print(cmd, flush=True)
        os.system(cmd)

if __name__ == '__main__':
    main()
