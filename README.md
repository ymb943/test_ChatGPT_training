# References  
This source code is originated from ```https://github.com/airobotlab/KoChatGPT```  
I tested the code in this version based on the above originated code.

# Environment
```
ubuntu:20.04
conda create -n py37 python=3.7 -y
pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116&&
pip install transformers==4.35.2&&
pip install accelerate==0.24.1&&
pip install colossalai==0.2.7&&
pip install openai&&
pip install langchain==0.0.113&&
pip install pandas>=1.4.1&&
pip install datasets&&
pip install loralib==0.1.2&&
pip install jupyterlab
```

# Run files
1. Vanilla GPT ```python 000_GPT_text_completion_model.py```  
2. Supervised Fine Tuning ```python 001_SFT.py```  
3. Reward model ```python 002_RM.py```  
4. PPO training ```python 003_PPO.py```