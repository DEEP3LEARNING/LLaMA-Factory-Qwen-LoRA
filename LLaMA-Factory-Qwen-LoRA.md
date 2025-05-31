

**步骤 1: 环境准备 (Ubuntu)**



1.  **NVIDIA驱动和CUDA Toolkit:**

    *   确保你的Ubuntu系统安装了合适的NVIDIA驱动。检查：`nvidia-smi`

    *   安装CUDA Toolkit。推荐版本与PyTorch兼容（例如CUDA 11.8 或 12.1）。

        *   可以从NVIDIA官网下载：[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

        *   验证安装：`nvcc --version`



2.  **Anaconda/Miniconda (推荐用于环境管理):**

    *   下载并安装Miniconda：

        ```bash

        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

        bash Miniconda3-latest-Linux-x86_64.sh

        source ~/.bashrc # 或者重启终端

        ```



3.  **创建并激活Conda环境:**

    ```bash

    conda create -n llama_factory python=3.10 -y

    conda activate llama_factory

    ```



4.  **安装Git:**

    ```bash

    sudo apt update

    sudo apt install git -y

    ```



5.  **安装PyTorch:**

    *   根据你的CUDA版本从PyTorch官网 ([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)) 获取安装命令。例如，对于CUDA 11.8:

        ```bash

        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

        ```

    *   验证PyTorch GPU支持：

        ```python

        import torch

        print(torch.cuda.is_available())

        print(torch.cuda.get_device_name(0))

        ```



***上面极力推荐使用阿里云的大模型服务器，全都是预装好的，而且不需要环境之间的相互匹配（怀念docker），ModelScope中现在注册还有新人奖励，能免费玩几个小时。***



**步骤 2: LLaMA-Factory 安装**



1.  **克隆LLaMA-Factory仓库:**

    ```bash

    git clone https://github.com/hiyouga/LLaMA-Factory.git

    cd LLaMA-Factory

    ```



2.  **安装依赖:**

    ```bash

    pip install -r requirements.txt

    ```

    *   **可选但推荐 (为了更好的性能和功能):**

        *   **FlashAttention-2 (如果GPU支持):**

            ```bash

            pip install flash-attn --no-build-isolation

            ```

        *   **Bitsandbytes (用于QLoRA - 4bit/8bit量化训练):**

            ```bash

            # Linux (推荐)

            pip install bitsandbytes

            # Windows (可能需要特定版本或从源码编译)

            # pip install bitsandbytes-windows # (可能会过时，检查官方文档)

            ```

        *   **DeepSpeed (用于大规模模型训练):**

            ```bash

            pip install deepspeed

            ```



**步骤 3: 数据准备**



LLaMA-Factory支持多种数据格式，最常用的是Alpaca格式的JSON文件。



1.  **创建数据目录:**

    ```bash

    mkdir data

    ```



2.  **准备数据集文件 (例如 `my_qwen_data.json`):**

    文件应放在 `LLaMA-Factory/data/` 目录下。

    格式示例 (Alpaca):

    ```json

    [

      {

        "instruction": "请介绍一下你自己。",

        "input": "",

        "output": "我是一个由LLaMA-Factory微调的Qwen模型。"

      },

      {

        "instruction": "广州塔的高度是多少？",

        "input": "",

        "output": "广州塔，又称小蛮腰，总高度为600米，其中塔身主体高454米，天线桅杆高146米。"

      },

      {

        "instruction": "将以下句子从中文翻译成英文。",

        "input": "今天天气真好。",

        "output": "The weather is really nice today."

      }

    ]

    ```

    *   **`instruction`**: 用户的指令或问题。

    *   **`input`**: (可选) 额外的上下文信息。如果指令本身足够清晰，可以为空。

    *   **`output`**: 模型期望的回答。



3.  **在 `dataset_info.json` 中注册你的数据集:**

    编辑 `LLaMA-Factory/data/dataset_info.json` 文件，添加你的数据集信息。

    例如，在文件中添加如下条目：

    ```json

    {

      // ... 其他数据集 ...

      "my_qwen_dataset": {

        "file_name": "my_qwen_data.json", // 对应你在 data/ 目录下创建的文件名

        "columns": {

          "prompt": "instruction",

          "query": "input",

          "response": "output"

        }

      }

      // ... 其他数据集 ...

    }

    ```

    *   `my_qwen_dataset` 是你给数据集起的名字，后续训练时会用到。

    *   `file_name` 指向你实际的数据文件。

    *   `columns` 映射了JSON字段到LLaMA-Factory内部的 "prompt", "query", "response" 角色。



**步骤 4: 模型下载 (可选)**



LLaMA-Factory会在首次运行时自动从Hugging Face Hub下载模型。如果你想预先下载或使用本地模型：



*   **从Hugging Face Hub下载:**

    例如，下载 `Qwen/Qwen-7B-Chat` (你需要有足够的磁盘空间)：

    ```bash

    # 你可能需要先登录 Hugging Face CLI

    # pip install huggingface_hub

    # huggingface-cli login

    

    # 使用 git lfs (确保已安装: sudo apt install git-lfs; git lfs install)

    git lfs install

    git clone https://huggingface.co/Qwen/Qwen-7B-Chat /path/to/your/models/Qwen-7B-Chat

    ```

    然后在训练命令中，`--model_name_or_path` 指定为本地路径 `/path/to/your/models/Qwen-7B-Chat`。



**步骤 5: 配置微调参数并开始微调**



**使用Web UI (`webui.py`):**



1.  启动Web UI:

    ```bash

    python src/webui.py

    ```

2.  在浏览器中打开 `http://localhost:7860` (或指定的端口)。

3.  在UI界面中：

    *   **模型名称:** 选择或输入 `Qwen/Qwen-7B-Chat` (或其他Qwen模型)。

    *   **微调方法:** 选择 `lora`。

    *   **数据集:** 选择你定义的 `my_qwen_dataset`。

    *   **Prompt模板:** 选择 `qwen`。

    *   **LoRA模块 (lora_target):** 输入 `c_attn` 或 `qwen`。

    *   **输出目录:** 指定一个路径。

    *   **其他超参数:** 如学习率、批大小、Epoch数、FP16/BF16、量化 (Quantization) 等，根据你的需求和硬件进行调整。

    *   点击 "开始"。



**步骤 6: 监控训练**



训练过程中，终端会输出日志信息，包括loss、learning rate等。你可以在输出目录中找到保存的checkpoint。



**步骤 7: 模型导出/合并 (可选)**



LoRA微调后得到的是一个适配器 (adapter)，而不是完整的模型。你可以将这个适配器与基础模型合并，得到一个可以直接使用的完整模型。