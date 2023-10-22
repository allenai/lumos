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

## 🔥 News
- **[2023, Oct ]** We release the important items for training and evaluating **Lumos**:
  - 💻 **Lumos** code for annotation generation, training and evaluation
  - 🤗 **Lumos** checkpoints with 7B model size
  - 🤗 **Lumos** training annotations and their raw data
 
## 🧩 Architecture
<p align="center">
<img src=assets/lumos.png />
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
| Dataset Names | 🤗 Huggingface Datasets Link  |
|----------------|----------------|
| lumos-complex_qa    |  Planning: , Grounding:        |
| lumos-complex_qa-onetime         |  Planning: , Grounding:         |
| lumos-web_agent  |  Planning: , Grounding:     |
| lumos-maths         |  Planning: , Grounding:        |
| lumos-maths-onetime         |  Planning: , Grounding:        |
| lumos-unified     | Planning: , Grounding:       |

### 🧑‍🎓️ Train Modules with Generated Annotation



## Evaluation


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
`data_fn` is the path you store the raw benchmarks.
