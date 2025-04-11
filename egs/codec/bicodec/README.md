# Bicodec 使用说明

## 数据准备

Bicodec的dataloader通过jsonl格式进行数据的加载管理，jsonl格式的文件中每一行是一个json对象，json对象中需要包含两个必要keys:
    - index (str): sample的index
    - wav_path (Path): 对应音频的绝对路径

