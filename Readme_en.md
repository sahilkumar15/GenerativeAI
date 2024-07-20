# Problem Summary

<details open=True>
<summary>Table of Contents</summary>

- [Problem Summary](#problem-summary)
  - [I. Download Issues](#i-download-issues)
    - [1.1 Code Download](#11-code-download)
    - [1.2 Weight Download](#12-weight-download)
    - [1.3 Network Download](#13-network-download)
    - [1.4 Clone Voice Weights](#14-clone-voice-weights)
  - [II. Environment Configuration Issues](#ii-environment-configuration-issues)
    - [2.1 GPU Environment](#21-gpu-environment)
    - [2.2 CPU Environment](#22-cpu-environment)
    - [2.3 VRAM Issues](#23-vram-issues)
  - [III. Runtime Issues](#iii-runtime-issues)
    - [3.1 File Not Found](#31-file-not-found)
    - [3.2 FFMPEG Issues](#32-ffmpeg-issues)
    - [3.3 Path Issues](#33-path-issues)
    - [3.4 GFPGANer is not defined](#34-gfpganer-is-not-defined)
    - [3.5 Microsoft Visual C++ 14.0 is required](#35-microsoft-visual-c-140-is-required)
    - [3.6 Multiple Server Deployment](#36-multiple-server-deployment)
    - [3.7 GeminiPro Proxy Parameter Settings](#37-geminipro-proxy-parameter-settings)
    - [3.8 Project Update Directions](#38-project-update-directions)
    - [3.9 version GLIBCXX_3.4.* not found](#39-version-glibcxx_34_not-found)
    - [3.10 Gradio Connection errored out](#310-gradio-connection-errored-out)
    - [3.11 gr.Error("No clone environment or clone model weights, unable to clone voice", e)](#311-grerror-no-clone-environment-or-clone-model-weights-unable-to-clone-voice-e)
    - [3.12 OSError: WinError 127 The specified program could not be found](#312-oserror-winerror-127-the-specified-program-could-not-be-found)
    - [3.13 LLM Dialogue Step Error: "Sorry, your request encountered an error, please try again."](#313-llm-dialogue-step-error-sorry-your-request-encountered-an-error-please-try-again)
    - [3.14 Startup Error: SadTalker Error: invalid load key, 'v'.](#314-startup-error-sadtalker-error-invalid-load-key-v)
    - [3.15 File is not a zip file](#315-file-is-not-a-zip-file)
  - [IV. Usage Issues](#iv-usage-issues)
    - [4.1 Can TTS Voice Be Adjusted](#41-can-tts-voice-be-adjusted)
    - [4.2 Voice Cloning Operations](#42-voice-cloning-operations)
  - [V. Feature Iteration Issues](#v-feature-iteration-issues)
    - [5.1 LLM Model Update](#51-llm-model-update)
    - [5.2 Clone Voice Model Replacement](#52-clone-voice-model-replacement)
    - [5.3 LLM API Usage Issues](#53-llm-api-usage-issues)
    - [5.4 Linly-Talker Effect Issues](#54-linly-talker-effect-issues)
  - [VI. Community Group Issues](#vi-community-group-issues)

</details>

## I. Download Issues

### 1.1 Code Download

The code can be downloaded from GitHub at [https://github.com/Kedreamix/Linly-Talker](https://github.com/Kedreamix/Linly-Talker) or from Gitee at [https://gitee.com/kedreamix/Linly-Talker](https://gitee.com/kedreamix/Linly-Talker).

GitHub will have the latest code, while Gitee will be updated periodically.

### 1.2 Weight Download

Weights can be downloaded from the following four sources, detailed in the README:

- [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)
- [Huggingface](https://huggingface.co/Kedreamix/Linly-Talker)
- [Modelscope](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/summary)
- [Quark(夸克网盘)](https://pan.quark.cn/s/f48f5e35796b)

SadTalker code can be downloaded from [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`). You can also run the shell script `bash scripts/download_models.sh` to automatically download all models and move them to the appropriate directory (suitable for Linux).

Wav2Lip model code can be downloaded from One Drive. You only need to download the first or second model:

| Model                        | Description                                           | Link to the model                                            |
| ---------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Wav2Lip                      | Highly accurate lip-sync                              | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW) |
| Wav2Lip + GAN                | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |
| Expert Discriminator         | Weights of the expert discriminator                   | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQRvmiZg-HRAjvI6zqN9eTEBP74KefynCwPWVmF57l-AYA?e=ZRPHKP) |
| Visual Quality Discriminator | Weights of the visual disc trained in a GAN setup     | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EQVqH88dTm1HjlK11eNba5gBbn15WMS0B0EZbDBttqrqkg?e=ic0ljo) |

GPT-SoVITS model code can be downloaded from the following links. For details, see [https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#预训练模型](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/cn/README.md#预训练模型).

Download the pre-trained models from [GPT-SoVITS Models](https://huggingface.co/lj1995/GPT-SoVITS) and place them in `GPT_SoVITS\pretrained_models`.

Chinese users can download the two models by clicking "Download Copy" on the following link:

- [GPT-SoVITS Models](https://www.icloud.com.cn/iclouddrive/056y_Xog_HXpALuVUjscIwTtg#GPT-SoVITS_Models)

Additionally, MuseTalk models can be downloaded from Modelscope. It is recommended to use Modelscope for faster download speeds in China.

```bash
python scripts/modelscope_download.py
```

### 1.3 Network Download

Sometimes, when downloading code, network issues may arise, such as with downloading large models from `huggingface`. To address this, I have also provided a [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`) link. You can download files locally and place them in the appropriate folders to achieve the same functionality.

> If you encounter issues with any file downloads, feel free to suggest them to me, and I will upload them to Baidu Netdisk.

### 1.4 Voice Cloning Weights

To protect user privacy, I have not provided voice cloning weights as this may involve copyright issues. If you are interested, you can try training using the same methods or contact me privately. Thank you for your understanding.

Some voice cloning and reference audio files are available on [Quark(夸克网盘)](https://pan.quark.cn/s/f48f5e35796b) for you to download.

## II. Environment Configuration Issues

### 2.1 GPU Environment

I use Pytorch version 2.0+ since the voice cloning model requires a relatively high version of Pytorch. The specific download command can be set according to the Pytorch official commands at [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). Sometimes, using Anaconda for installation can be convenient for management and installing other dependencies.

```bash
conda create -n linly python=3.10  
conda activate linly

# Pytorch Installation Method 1: Install via Conda
# CUDA 11.7
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
# CUDA 11.8
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Pytorch Installation Method 2: Install via Pip
# CUDA 11.7
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

conda install -q ffmpeg # ffmpeg==4.2.2

# Upgrade pip
python -m pip install --upgrade pip
# Change pypi source to speed up library installation
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install tb-nightly -i https://mirrors.aliyun.com/pypi/simple
pip install -r requirements_webui.txt

# Install dependencies for MuseTalk
pip install --no-cache-dir -U openmim
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"
```

> GPU environments sometimes require configuring CUDA. There are many online resources available for this, so I won’t go into detail here.

### 2.2 CPU Environment

You can replace the GPU with a CPU, but this may be slower. When installing Pytorch, do not install the GPU version, and it should still work. However, the results might be subpar due to the demands of running large models, so a GPU environment is recommended.

### 2.3 Memory Issues

Based on my tests, SadTalker with a default `Batch Size = 1` and without importing large models uses about 2GB of memory, while importing SadTalker uses about 4-6GB. Therefore, a computer with at least 6-8GB of memory should be able to run this in a GPU environment.

Using MuseTalk will require more memory, ideally around 11GB. However, if the uploaded image resolution is high, it may still cause memory overflow issues, so this needs attention.

## III. Runtime Issues

### 3.1 File Not Found

If a `FileNotFound` error occurs and it relates to weights, refer back to section 1.2 and redownload them. Make sure to check the README for the correct folder structure.

### 3.2 FFMPEG Issues

If you encounter `ffmpeg` issues when generating videos, it may be due to an incorrect `ffmpeg` installation. There are two ways to fix this.

The first method is to install `ffmpeg` using Conda, requiring version ffmpeg>=4.2.2:

```bash
conda install -q ffmpeg # ffmpeg==4.2.2
```

The second method is to install `ffmpeg` directly:

```bash
# Linux installation
sudo apt install ffmpeg
```

For Windows installation, download `ffmpeg` from the official site [https://ffmpeg.org/](https://ffmpeg.org/). You can follow this guide: [Windows Installation of ffmpeg](https://zhuanlan.zhihu.com/p/118362010).

### 3.3 Path Issues

If files are not placed correctly, you need to set the corresponding path in `config.py`. You can also change the port, which is set to 7860 by default but can be changed to any available port.

### 3.4 GFPGANer is not defined

If this issue arises, it indicates a need for the enhanced `gfpgan` module. Install the `gfpgan` library with the following command:

```bash
pip install gfpgan
```

### 3.5 Microsoft Visual C++ 14.0 is required

If you encounter this issue, it means that Windows needs certain dependencies. This article can help resolve it: [Microsoft Visual C++ 14.0 is required solution](https://zhuanlan.zhihu.com/p/126669852).

![Microsoft Visual C++ 14.0 error](https://picx.zhimg.com/80/v2-d25b289827fc989f419df70f650b44e9.png)

### 3.6 Multiple Server Deployment

For multiple servers, consider deploying the large model on another server. I have written a FastAPI version that allows model usage via API deployment.

Alternatively, you can deploy it locally first, which avoids loading the large model every time, thus saving some waiting time.

### 3.7 GeminiPro Proxy Settings

For GeminiPro proxy settings, the `proxy_url` parameter can be set. I have set it to `http://127.0.0.1:7890` since I use clash, which uses port 7890. You can replace it with your corresponding port.

### 3.8 Project Update Direction

To add other models or directions, you can add the corresponding algorithms in the `ASR`, `TTS`, `THG`, and `LLM` folders. Feel free to suggest updates, and I will incorporate them when I have time. Contributions are welcome!

> I will keep updating the project. Sometimes it takes a while to come up with good ideas before implementing them. Contributions and PRs are always welcome!

### 3.9 version GLIBCXX_3.4.* not found

If you encounter this issue, it may be due to version problems with certain libraries. Refer to ["`GLIBCXX_3.4.32' not found" error at runtime. GCC 13.2.0](https://stackoverflow.com/questions/76974555/glibcxx-3-4-32-not-found-error-at-runtime-gcc-13-2-0).

```bash
/lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

I found two potential solutions. Firstly, the Python version may resolve the issue. I did not encounter this error with Python 3.10, but did with 3.9.

The second solution is to downgrade the `pyopenjtalk` and `opencc` libraries:

```bash
pip install pyopenjtalk==0.3.1
pip install opencc==1.1.1
```

You can also check the GLIBCXX version on your machine with:

```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

### 3.10 Gradio Connection errored out

I haven't encountered this issue myself, but some people have, especially on Windows. It seems less stable there. If anyone has suggestions or common solutions, please let me know, as the information available online doesn’t seem very helpful.

### 3.11 gr.Error("No cloning environment or no cloning model weights, unable to clone voice", e)

This is related to functional iterations, specifically the cloning environment and cloning model weights. First, make sure the cloning environment is set up correctly:

```bash
pip install -r VITS/requirements.txt
```

Then, follow the instructions in [4.2 Replacing Cloned Voice Model](https://github.com/Kedreamix/Linly-Talker/blob/main/%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98%E6%B1%87%E6%80%BB.md#42-%E5%85%8B%E9%9A%86%E8%AF%AD%E9%9F%B3%E6%A8%A1%E5%9E%8B%E6%9B%BF%E6%8D%A2) to modify the model weights.

### 3.12 OSError: [WinError 127] The specified program could not be found

This error typically occurs when trying to run a program or command on a Windows OS, but the system cannot find the specified executable. Generally, it means the corresponding library wasn't installed correctly. Reinstall the library as per the error message to fix it.

### 3.13 LLM Dialogue Error: "Sorry, there was an error with your request. Please try again."

This error is caused by compatibility issues with large models. Reinstall the necessary libraries to resolve it:

```bash
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
```

### 3.14 Startup Error: SadTalker Error: invalid load key, 'v'.

If you encounter this error while starting up, it’s likely due to an incorrect download of the model weights, especially the two `pth` files related to `mapping`. These files should be 174MB in size. The correct sizes are:

```bash
149M checkpoints/mapping_00109-model.pth.tar
149M checkpoints/mapping_00229-model.pth.tar
```

To resolve this issue, redownload these two files. Here are three sources to download the weights from, as detailed in the `README`:

- [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)
- [huggingface](https://huggingface.co/Kedreamix/Linly-Talker)
- [modelscope](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/summary)
- [Quark(夸克网盘)](https://pan.quark.cn/s/f48f5e35796b)

If `git lfs clone` encounters bugs, you can download the files directly using `wget` and re-upload them to the `checkpoints` folder:

```bash
wget -c https://modelscope.cn/api/v1/models/Kedreamix/Linly-Talker/repo?Revision=master&FilePath=checkpoints%2Fmapping_00109-model.pth.tar
wget -c https://modelscope.cn/api/v1/models/Kedreamix/Linly-Talker/repo?Revision=master&FilePath=checkpoints%2Fmapping_00229-model.pth.tar
```

### 3.15 File is not a zip file

Network issues can cause problems when `nltk` tries to automatically download models, rendering them unusable. To avoid this, manually download the `nltk_data` files. First, find the cache path for `nltk_data`:

```python
import nltk
print(nltk.data.path)
```

After locating the path, place the downloaded files into the cache path. If there are existing download caches, replace and overwrite them. The `nltk_data` files are available on [Quark(夸克网盘)](https://pan.quark.cn/s/9e7af40d8a26), including `corpora` and `taggers`.

## IV. Usage Issues

### 4.1 Can TTS Voices Be Adjusted?

Linly-Talker has built-in TTS models, including `EdgeTTS` and `PaddleTTS`. `EdgeTTS` offers better and more diverse voice options. You can adjust voices in the `TTS Method Voice Adjustment` section.

![TTS Voice Adjustment](https://pica.zhimg.com/v2-5c768befccf3e7d7ea055d52398ea8c4.png)

There are many voices, including different languages and Taiwanese voices. For voices starting with `zh-CN`, `Xiao` is female and `Yun` is male.

![EdgeTTS](https://pic1.zhimg.com/v2-0be853527468de483e5e818b5c64be51.png)

`PaddleTTS` offers fewer options, but you can choose according to your preference.

![PaddleTTS](https://picx.zhimg.com/v2-cc67b4fa7a0c7f2fc94e6031bfdcc93d.png)

### 4.2 Voice Cloning Operation

For voice cloning, refer to [GPT-SoVITS on GitHub](https://github.com/RVC-Boss/GPT-SoVITS). There are also many tutorial videos on Bilibili, along with some public weights. I have also uploaded some to Quark Netdisk. Note the following steps for using cloned voices:

1. Load model weights.
2. Upload reference audio.
3. Transcribe the reference audio text.
4. Input target text, i.e., the question.
5. Generate the cloned voice.

![Voice Cloning](https://pica.zhimg.com/v2-a93d3b3ef38d7837be404f3442d1ac93.png)

## Section Five: Feature Iteration Issues

### 5.1 LLM Model Updates

To add a new LLM model, place the chosen model in the LLM folder.

Here is a Chinese template suitable for any large language model (LLM). This template is designed to be flexible and easy to configure while providing a consistent interface for different models.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMTemplate:
    def __init__(self, model_name_or_path, mode='offline'):
        """
        Initialize the LLM template.

        Args:
            model_name_or_path (str): Model name or path.
            mode (str, optional): Mode, 'offline' for offline mode, 'api' for using API mode. Defaults to 'offline'.
        """
        self.mode = mode
        # Model initialization
        self.model, self.tokenizer = self.init_model(model_name_or_path)
        self.history = None
    
    def init_model(self, model_name_or_path):
        """
        Initialize the language model.

        Args:
            model_name_or_path (str): Model name or path.

        Returns:
            model: Loaded language model.
            tokenizer: Loaded tokenizer.
        """
        # TODO: Load the model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                     device_map="auto", 
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return model, tokenizer   
    
    def generate(self, prompt, system_prompt=""):
        """
        Generate dialogue response.

        Args:
            prompt (str): Dialogue prompt.
            system_prompt (str, optional): System prompt. Defaults to "".

        Returns:
            str: Dialogue response.
        """
        # TODO: Model prediction
        # This section should be adjusted according to the specific model, as it uses inference templates borrowed from HuggingFace.
        if self.mode != 'api':
            try:
                response, self.history = self.model.chat(self.tokenizer, prompt, history=self.history, system=system_prompt)
                return response
            except Exception as e:
                print(e)
                return "Sorry, your request encountered an error. Please try again."
        else:
            return self.predict_api(prompt)
    
    def predict_api(self, prompt):
        """
        Predict dialogue response using an API.

        Args:
            prompt (str): Dialogue prompt.

        Returns:
            str: Dialogue response.
        """
        # Placeholder for the API version, similar to Linly-API, can be implemented if interested.
        pass 
    
    def chat(self, system_prompt, message):
        response = self.generate(message, system_prompt)
        self.history.append((message, response))
        return response, self.history
    
    def clear_history(self):
        self.history = []
```

### 5.2 Cloning and Replacing Voice Models

Clone voice models can be placed in the `GPT_SoVITS/pretrained_models` folder. This makes it easy to see and modify the specified models in the WebUI interface.

![WebUI](docs/WebUI3.png)

### 5.3 Using LLM APIs

We can use available LLM APIs. To do this, modify the files under the LLM directory. Here, we take ChatGPT as an example. We can implement the `generate` and `chat` functions, and then modify the selection in `__init__.py`.

```python
'''
pip install openai
'''

import os
import openai

class ChatGPT:
    def __init__(self, model_path='gpt-3.5-turbo', api_key=None, proxy_url=None, prefix_prompt='Please answer the following question in less than 25 words\n\n'):
        if proxy_url:
            os.environ['https_proxy'] = proxy_url
            os.environ['http_proxy'] = proxy_url
        openai.api_key = api_key
        self.model_path = model_path
        self.prefix_prompt = prefix_prompt
        self.history = []
        
    def generate(self, message, system_prompt="You are a helpful assistant."):
        self.history.append({"role": "user", "content": self.prefix_prompt + message})
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_path,
                messages=messages
            )
            answer = response['choices'][0]['message']['content']
            if 'sorry' in answer.lower():
                return 'Sorry, your request encountered an error. Please try again.'
            return answer
        except Exception as e:
            print(e)
            return 'Sorry, your request encountered an error. Please try again.'
        
    def chat(self, system_prompt="You are a helpful assistant.", message=""):
        response = self.generate(message, system_prompt)
        self.history.append({"role": "assistant", "content": response})
        return response, self.history
    
    def clear_history(self):
        self.history = []

if __name__ == '__main__':
    API_KEY = '******'  # Replace with your API key
    llm = ChatGPT(model_path='gpt-3.5-turbo', api_key=API_KEY, proxy_url=None)
    answer, history = llm.chat(message="How to cope with stress?")
    print(answer)
    
    from time import sleep
    sleep(5)  # Simulate delay in conversation
    
    answer, history = llm.chat(message="Can you elaborate further?")
    print(answer)
```

### 5.4 Linly-Talker Effectiveness Issues

Linly-Talker has some issues with its effectiveness, which will be continuously updated and improved. I welcome your suggestions and will work hard to make improvements. Any bugs found will be promptly addressed and responded to.

## Section Six: Discussion Group Issues

A WeChat group has been created for everyone to exchange and learn.

If you have any thoughts, you can leave a message under the video or send me a private message. I will read them all. If the discussion group link expires, you can add me on WeChat: `pikachu2biubiu`.

Since the discussion group has over 200 members, I need to add you manually. Please add me first, and I will add you to the group. Thank you for your attention!