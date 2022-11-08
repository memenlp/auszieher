FROM registry.shdocker.tuya-inc.top/ai-platform-public/ubuntu-py3.7:v1.0

RUN sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list \
    && sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list \
    && apt-get update && apt-get install -y libboost-all-dev wget unzip vim less \
    && rm -rf /var/lib/apt/list/* \
    &&apt-get clean

RUN pip install --upgrade pip && \
    pip config set global.index-url https://maven.tuya-inc.top/repository/pypi-group/simple

ENV LANG=zh_CN.utf-8
ENV LC_ALL=zh_CN.utf-8

RUN apt-get install -y language-pack-zh-hans fonts-droid-fallback ttf-wqy-zenhei ttf-wqy-microhei fonts-arphic-ukai fonts-arphic-uming liblzma-dev libbz2-dev
RUN echo -e 'LANG=zh_CN.UTF-8\nLANGUAGE=zh_CN:zh:en_US:en' > /etc/environment
RUN echo -e 'en_US.UTF-8 UTF-8 \n zh_CN.UTF-8 UTF-8 \n zh_CN.GBK GBK \n zh_CN GB2312' > /var/lib/locales/supported.d/local
RUN locale-gen

ENV PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.7/site-packages/:/workspace/site-packages"
COPY ./requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

RUN cp /usr/lib/python3.6/lib-dynload/_bz2.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python3.7/lib-dynload/_bz2.cpython-37m-x86_64-linux-gnu.so
RUN chmod +x /usr/local/lib/python3.7/lib-dynload/_bz2.cpython-37m-x86_64-linux-gnu.so

COPY ./third_party/lzma.py /usr/local/lib/python3.7/lzma.py
COPY ./auszieher /workspace/auszieher
COPY ./data /workspace/data
COPY ./server.py /workspace
WORKDIR /workspace

# RUN pip config set global.index-url https://pypi.doubanio.com/simple/
# RUN pip install xgboost==1.6.2
# RUN pip config set global.index-url https://maven.tuya-inc.top/repository/pypi-group/simple

# RUN python -m textblob.download_corpora

CMD ["python",  "-m", "server"]
