conda activate chatglm3-demo

https://zhipu-ai.feishu.cn/wiki/OwCDwJkKbidEL8kpfhPcTGHcnVc   硬件需求


pip install -r requirements.txt


python cli_demo.py

python web_demo_gradio.py

streamlit run web_demo_streamlit.py

streamlit run web_demo_streamlit.py --server.port 7000

pip install transformers==4.40.0  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install sentencepiece   -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install peft -i https://pypi.tuna.tsinghua.edu.cn/simple


No module named 'tqdm.contrib'



pip install numpy==1.19 -i https://pypi.tuna.tsinghua.edu.cn/simple




A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.1 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "E:\ChatGLM3\basic_demo\cli_demo.py", line 3, in <module>
    from transformers import AutoTokenizer, AutoModel
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\__init__.py", line 26, in <module>
    from . import dependency_versions_check
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\dependency_versions_check.py", line 16, in <module>
    from .utils.versions import require_version, require_version_core
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\utils\__init__.py", line 34, in <module>
    from .generic import (
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\utils\generic.py", line 462, in <module>
    import torch.utils._pytree as _torch_pytree
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\__init__.py", line 2120, in <module>
    from torch._higher_order_ops import cond
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\_higher_order_ops\__init__.py", line 1, in <module>
    from .cond import cond
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\_higher_order_ops\cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\_subclasses\functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\_subclasses\functional_tensor.py", line 258, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\torch\_subclasses\functional_tensor.py:258: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\torch\csrc\utils\tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
Traceback (most recent call last):
  File "E:\ChatGLM3\basic_demo\cli_demo.py", line 13, in <module>
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\models\auto\tokenization_auto.py", line 878, in from_pretrained
    tokenizer_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\dynamic_module_utils.py", line 502, in get_class_from_dynamic_module
    final_module = get_cached_module_file(
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\dynamic_module_utils.py", line 327, in get_cached_module_file
    modules_needed = check_imports(resolved_module_file)
  File "D:\Anaconda3\envs\chatglm3-demo\lib\site-packages\transformers\dynamic_module_utils.py", line 182, in check_imports
    raise ImportError(
ImportError: This modeling file requires the following packages that were not found in your environment: sentencepiece. Run `pip install sentencepiece`