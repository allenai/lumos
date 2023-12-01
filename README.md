# 🪄 Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs
<p align="center">
  <a href="https://allenai.github.io/lumos/">
    <img src="https://img.shields.io/badge/🌐-Website-red">
  </a>
  <a href="https://arxiv.org/abs/2311.05657">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="https://huggingface.co/datasets?sort=trending&search=ai2lumos">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>
  <a href="https://huggingface.co/models?sort=trending&search=ai2lumos">
    <img src="https://img.shields.io/badge/🤗-Model-green">
  </a> 
</p>

🖋 **Authors:** [Da Yin](https://wadeyin9712.github.io/), [Faeze Brahman](https://fabrahman.github.io/), [Abhilasha Ravichander](https://www.cs.cmu.edu/~aravicha/), [Khyathi Chandu](https://www.cs.cmu.edu/~kchandu/), [Kai-Wei Chang](http://web.cs.ucla.edu/~kwchang/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/), [Bill Yuchen Lin](https://yuchenlin.xyz/)

We introduce 🪄**Lumos**, Language Agents with **Unified** Data Formats, **Modular** Design, and **Open-Source** LLMs. **Lumos** unifies a suite of complex interactive tasks and achieves competitive performance with GPT-4/3.5-based and larger open-source agents. 

‼️ **Lumos** has following features:
* 🧩 **Modular Architecture**:
  - 🧩 **Lumos** consists of planning, grounding, and execution modules built based on LLAMA-2-7B and off-the-shelf APIs.
  - 🤗 **Lumos** utilizes a unified data format that encompasses multiple task types, thereby enabling the developed agent framework to conveniently support a range of interactive tasks.
* 🌍 **Diverse Training Data**:
  - 🌍 **Lumos** is trained with ~40K diverse high-quality subgoal/action annotations from ground-truth reasoning steps in existing benchmarks with GPT-4.
  - ⚒️ **Lumos** data can be instrumental for future research in developing open-source agents for complex interactive tasks.
* 🚀 **Competitive Performance**:
  - 🚀 **Lumos** is comparable or even beats **GPT-series** agents on web/complex QA tasks Mind2Web and HotpotQA, and **larger open agents** on math tasks.
  - 🚀 **Lumos** exceeds contemporaneous agents that have been **fine-tuned** with in-domain HotpotQA and Mind2Web annotations, such as **FiReAct** and **AgentLM**.
  - 🚀 **Lumos** performs better than open agent baseline formulations including **chain-of-thoughts** and **integrated** training.
  - 🚀 **Lumos** surpasses larger open LLM agents and domain-specific agents on an unseen task, WebShop.
 
## 🤩 Citation

If you find this work is relevant with your research, please feel free to cite our work!
```
@article{yin2023lumos,
  title={{Lumos: Learning Agents with Unified Data, Modular Design, and Open-Source LLMs}},
  author={Yin, Da and Brahman, Faeze and Ravichander, Abhilasha and Chandu, Khyathi and Chang, Kai-Wei and Choi, Yejin and Lin, Bill Yuchen},
  journal={arXiv preprint arXiv:2311.05657},
  year={2023}
}
```

## 🔥 News
- **[2023, Nov 8]** We release the important items for training and evaluating **Lumos**:
  - 💻 **Lumos** code for annotation generation, training and evaluation
  - 🤗 **Lumos** checkpoints with 7B model size
  - 🤗 **Lumos** training annotations and their raw data
 
## 🧩 Architecture
<p align="center">
<img src=assets/lumos.png width=800/>
</p>

## 🛠️ Setup
```
./setup.sh
```
Please make sure that the cudatoolkit version in `setup.sh` aligns with your local cuda version.

## Training
### 📈 Training Data Download
We collect all the training annotations, raw data and prompt converted annotations in a single [Google Drive folder](https://drive.google.com/drive/folders/1ASFhOkhezgewVxR01dQg-8KUVR8IdBlY?usp=sharing). It can be downloaded by
```
cd data
python -c "import gdown; gdown.download_folder('https://drive.google.com/drive/folders/1ASFhOkhezgewVxR01dQg-8KUVR8IdBlY?usp=sharing', quiet=True)" 
```

We also provide generated annotations for planning and grounding modules in 🤗 Huggingface Datasets. 
| Dataset Names | 🤗 Huggingface Links  |
|----------------|----------------|
| lumos_complex_qa_iterative    |  [Planning](https://huggingface.co/datasets/ai2lumos/lumos_complex_qa_plan_iterative), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_complex_qa_ground_iterative)        |
| lumos_complex_qa_onetime         |  [Planning](https://huggingface.co/datasets/ai2lumos/lumos_complex_qa_plan_onetime), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_complex_qa_ground_onetime)         |
| lumos_web_agent_iterative  |  [Planning](https://huggingface.co/datasets/ai2lumos/lumos_web_agent_plan_iterative), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_web_agent_ground_iterative)     |
| lumos_maths_iterative         |  [Planning](https://huggingface.co/datasets/ai2lumos/lumos_maths_plan_iterative), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_maths_ground_iterative)       |
| lumos_maths_onetime         |  [Planning](https://huggingface.co/datasets/ai2lumos/lumos_maths_plan_onetime), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_maths_ground_onetime)    |
| lumos_unified_iterative     | [Planning](https://huggingface.co/datasets/ai2lumos/lumos_unified_plan_iterative), [Grounding](https://huggingface.co/datasets/ai2lumos/lumos_unified_ground_iterative)    |

### 🧑‍🎓️ Train Modules with Generated Annotation
```
./train.sh [MODULE] [FORMULATION]
```
`[MODULE]` can be either `plan` or `ground`. `[FORMULATION]` can be either `iterative` or `onetime`.

You can adjust the fine-tuning hyperparameters and specific task you want to fine-tune in the training scripts such as `finetune_llama2_plan_iterative.sh` in [`scripts/train`](./scripts/train).

We also provide the fine-tuned planning and grounding module checkpoints in 🤗 Huggingface.
| Model Names | 🤗 Huggingface Links  |
|----------------|----------------|
| lumos_complex_qa_iterative    |  [Planning](https://huggingface.co/ai2lumos/lumos_complex_qa_plan_iterative), [Grounding](https://huggingface.co/ai2lumos/lumos_complex_qa_ground_iterative)        |
| lumos_complex_qa_onetime         |  [Planning](https://huggingface.co/ai2lumos/lumos_complex_qa_plan_onetime), [Grounding](https://huggingface.co/ai2lumos/lumos_complex_qa_ground_onetime)         |
| lumos_web_agent_iterative  |  [Planning](https://huggingface.co/ai2lumos/lumos_web_agent_plan_iterative), [Grounding](https://huggingface.co/ai2lumos/lumos_web_agent_ground_iterative)     |
| lumos_maths_iterative         |  [Planning](https://huggingface.co/ai2lumos/lumos_maths_plan_iterative), [Grounding](https://huggingface.co/ai2lumos/lumos_maths_ground_iterative)       |
| lumos_maths_onetime         |  [Planning](https://huggingface.co/ai2lumos/lumos_maths_plan_onetime), [Grounding](https://huggingface.co/ai2lumos/lumos_maths_ground_onetime)    |
| lumos_unified_iterative     |  [Planning](https://huggingface.co/ai2lumos/lumos_unified_plan_iterative), [Grounding](https://huggingface.co/ai2lumos/lumos_unified_ground_iterative)    |

## ✅ Evaluation
Evaluation scripts for different datasets are under [`scripts/eval`](./scripts/eval). For example, you can evaluate Lumos on HotpotQA by running:
```
./scripts/eval/hotpotqa.sh
```

## Others
### 📈 Data Annotation Generation
We provide the code for generating training annotations based on raw existing benchmarks from scratch. 

Before generating annotations, we first need to download the existing benchmarks providing ground-truth intermediate reasoning steps. 
The raw data are can be downloaded via this [Google Drive folder](https://drive.google.com/drive/folders/1ASFhOkhezgewVxR01dQg-8KUVR8IdBlY?usp=sharing).
```
python -m data.prompt_convertion \
  --domain DOMAIN \
  --data_fn DATA_FN \
  --convert_all
```
`domain` covers maths, complex QA and web agent. `data_fn` is the path where raw benchmarks are stored.

## ❤️ Acknowledgement
We greatly thank Tulu team for providing awesome [code](https://github.com/allenai/open-instruct) to finetune LLAMA-2. We also sincerely appreciate the contributors of [zeno-build](https://github.com/zeno-ml/zeno-build), [Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web), and [WebShop](https://github.com/princeton-nlp/WebShop) for providing fast GPT prompting, HTML preprocessing and evaluation docker environment.
