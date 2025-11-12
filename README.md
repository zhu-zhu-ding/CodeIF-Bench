## About
+ **CodeIF-Bench** is a benchmark for evaluating the instruction-following ability of LLM in interactive code generation tasks.
+ **CodeIF-Bench** contains 9 verifiable instruction strategies collected from code review tasks.
+ **CodeIF-Bench** contains 900+ verifiable instructions with test cases that cover both **SA** and **Non-SA** programming tasks and support **Multi-Turn** dialogue.

## Getting Started

### 1. Prepare the data

- The original repositories can be downloaded from  [link](https://figshare.com/s/aa2ec81006727d9ddb0c). 
- The data file can be finded in /data.

### 2. Environment Setup

```plain
conda create --name xxx --file environment.txt
conda activate xxx
pip install -r requirement.txt
```

### 3. Static_Conversation

- run `inference.sh`. Note that you should set the LLM settings (such as, url or keys) in `llm_factory.py`.
- run `run_metrics.sh` to get metrics.

### 4. Dynamic_Conversation

- run `inference_mbpp.sh`or `inference_repo.sh`. Note that you should set the LLM settings (such as, url or keys) in `multi_turn_xxx_eval.py`.
- run `run_metrics.sh` to get metrics.

### 5. Evaluation

- **IA:** The LLM's ability to follow current instructions

- **CA:** The  LLM's ability to follow instructions throughout the entire conversation

- **IFR:** The proportion of instructions an LLM forgets during the conversation

- **CIF:** The number of instructions last followed in a dynamic conversation

  For further details, please refer to our [paper](https://arxiv.org/abs/2503.22688). New version is coming soon!

## Citation

If you have any questions or suggestions, please email us at wangpeiding@buaa.edu.cn

If you find this repository useful, please cite our paper:

```
@misc{wang2025codeifbenchevaluatinginstructionfollowingcapabilities,
      title={CodeIF-Bench: Evaluating Instruction-Following Capabilities of Large Language Models in Interactive Code Generation}, 
      author={Peiding Wang and Li Zhang and Fang Liu and Lin Shi and Minxiao Li and Bo Shen and An Fu},
      year={2025},
      eprint={2503.22688},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2503.22688}, 
}
```

