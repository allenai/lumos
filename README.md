# 🪄 Lumos
<p align="center">
  <a href="">
    <img src="https://img.shields.io/badge/🌐-Website-red">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/📝-Paper-blue">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/🤗-Model-green">
  </a> 
</p>

We introduce 🪄**Lumos**, Language Agents with **Unified** Formats, **Modular** Design, and **Open-Source** LLMs. **Lumos** unifies a suite of complex interactive tasks and achieves competitive performance with GPT-4/3.5-based and larger open-source agents. 

**Lumos** has following features:
* 🧩 **Modular Architecture**:
  - **Lumos** consists planning, grounding, and execution modules built based on open LLMs such as LLAMA-2.
* 🌍 **Diverse Training Data**:
  - **Lumos** is trained with ~40K high-quality annotations from ground-truth reasoning steps in existing benchmarks with GPT-4. 
* 🚀 **Competitive Performance**:
  - 🚀 **Lumos** outperforms **GPT-4/3.5-based** agents on complex QA and web agent tasks, and larger open agents on maths tasks.
  - 🚀 **Lumos** performs better than open agent baseline formulations including **chain-of-thoughts** and **unmodularized** training.
  - 🚀 **Lumos** surpasses larger open LLM agents and domain-specific agents on an unseen task, WebShop.
 
### Citation

If you find this work is relevant with your research, please feel free to cite our work!
```
@article{yin2023lumos,
  title={Lumos: Towards Language Agents that are Unified, Modular, and Open Source},
  author={Yin, Da and Brahman, Faeze and Ravichander, Abhilasha and Chandu, Khyathi and Chang, Kai-Wei and Choi, Yejin and Lin, Bill Yuchen},
  year={2023}
}
```

## 🔥 News
- **[2023, Oct ]** We release the important items for training and evaluating **Lumos**:
  - 💻 **Lumos** code for annotation generation, training and evaluation
  - 🤗 **Lumos** checkpoints with 7B model size
  - 🤗 **Lumos** training annotations and their raw data
 
## 🧩 Architecture
<p align="center">
<img src=assets/lumos.png width=800/>
</p>

## Setup
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

We also provide generated annotations for planning and grounding modules in 🤗 Huggingface Datasets. Here're the annotations we release:
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
`\[MODULE\]` can be either `plan` or `ground`. `\[FORMULATION\]` can be either `iterative` or `onetime`.

You can adjust the fine-tuning hyperparameters and specific task you want to fine-tune in the training scripts such as `finetune_llama2_plan_iterative.sh` in [`scripts/train`](./scripts/train).

## Evaluation
Evaluation scripts for different datasets are under [`scripts/eval`](./scripts/eval). For example, you can evaluate Lumos on HotpotQA by running:
```
./hotpotqa.sh
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
