FROM python:3.10
LABEL authors="hl"
# 创建工作目录
RUN mkdir -p /code

# 指定pip源, 因为需要安装poetry
COPY pip.conf /root/.pip/pip.conf

# 将当前工作目录设置为 /code
WORKDIR /code

# 添加文件到容器中
ADD . /code

# 安装poetry
RUN pip install poetry
# 生成requirements.txt
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# 正式安装依赖
# !!! 注意--trusted-host 的值为tool.poetry.source的域名
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --trusted-host pypi.tuna.tsinghua.edu.cn

# 暴露端口 8501
EXPOSE 8501

CMD ["streamlit","run", "app.py"]
