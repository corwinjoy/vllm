from vllm import LLM, SamplingParams

import random

def random_line(lines):
    return random.choice(lines).strip()

def get_text_lines(filename):
    with open(filename) as f_in:
        lines = filter(None, (line.rstrip() for line in f_in))
        return list(lines)

txt_lines = get_text_lines('model/t8.shakespeare.txt')

mini_model = False
fb_model = True

# Original from bug
rem = """
llm = LLM(model = model_path, tensor_parallel_size=2, max_model_len=12*1024, enable_prefix_caching=True)
local_request_outputs= llm.genearate(prompts, 
     sampling_params=SamplingParams(temperature=0.7, max_tokens=1500, n=3, best_of=3)
)
"""

if fb_model:
    llm = LLM(model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
              tensor_parallel_size=2,
              max_model_len=12*1024,
              enable_prefix_caching=True,
              gpu_memory_utilization=0.90)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=1500, n=3, best_of=3)




# nroggendorff/smallama
# facebook/opt-125m
if mini_model:
    llm = LLM(model="crumb/nano-mistral",
              tensor_parallel_size=1,
              enable_prefix_caching=True,
              distributed_executor_backend = "ray", # "ray" is default, "mp" is python multiprocessing for single node
              scheduling_policy = "priority"
              # ,max_num_seqs = 4
              , disable_sliding_window = True
              ) # max_num_batched_tokens=2048*2
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30, n=3, best_of=3)

rem = """
Details for Distributed Inference and Serving
vLLM supports distributed tensor-parallel and pipeline-parallel inference and serving. Currently, we support 
Megatron-LM’s tensor parallel algorithm. We manage the distributed runtime with either Ray or python native 
multiprocessing. Multiprocessing can be used when deploying on a single node, multi-node inferencing currently 
requires Ray.

Multiprocessing will be used by default when not running in a Ray placement group and if there are sufficient GPUs 
available on the same node for the configured tensor_parallel_size, otherwise Ray will be used. This default can be 
overridden via the LLM class distributed-executor-backend argument or --distributed-executor-backend API server 
argument. Set it to mp for multiprocessing or ray for Ray. It’s not required for Ray to be installed for the multiprocessing case.

To run multi-GPU inference with the LLM class, set the tensor_parallel_size argument to the number of GPUs you want to use.
"""

# Problem could be with Ray cluster. Should check with mp.
rem = """
After you start the Ray cluster, you’d better also check the GPU-GPU communication between nodes. It can be non-trivial 
to set up. Please refer to the sanity check script for more information. If you need to set some environment variables 
for the communication configuration, you can append them to the run_cluster.sh script, e.g. -e NCCL_SOCKET_IFNAME=eth0. 
Note that setting environment variables in the shell (e.g. NCCL_SOCKET_IFNAME=eth0 vllm serve ...) only works for the 
processes in the same node, not for the processes in the other nodes. Setting environment variables when you create the 
cluster is the recommended way. See the discussion for more information.
"""

if False:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

random.seed(42)
for j in range(0, 2000):
    print("------------------------------------------------------------------------\n")
    print(f"Prompt batch: {j}")
    prompts = []
    for i in range(50):
        lines = []
        for _ in range(200):
            lines.append(random_line(txt_lines))
        prompt = "\n".join(lines)
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")


