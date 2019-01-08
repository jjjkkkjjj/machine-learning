FROM ubuntu:16.04
MAINTAINER Junnosuke Kado

# パッケージのインストールとアップデート
RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install build-essential
RUN apt-get -y install git vim curl wget
RUN apt-get -y install zlib1g-dev \
                       libssl-dev \
                       libreadline-dev \
                       libyaml-dev \
                       libxml2-dev \
                       libxslt-dev \
                       libncurses5-dev \
                       libncursesw5-dev


# pyenv のインストール
RUN git clone git://github.com/yyuu/pyenv.git /root/.pyenv
RUN git clone https://github.com/yyuu/pyenv-pip-rehash.git /root/.pyenv/plugins/pyenv-pip-rehash
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN echo 'eval "$(pyenv init -)"' >> .bashrc

# anaconda のインストール
ENV ANACONDA_VER 2.2.0
RUN pyenv install anaconda3-$ANACONDA_VER
RUN pyenv global anaconda3-$ANACONDA_VER
ENV PATH $PYENV_ROOT/versions/anaconda3-$ANACONDA_VER/bin:$PATH
RUN conda create -n py2 python=2.7 anaconda
RUN echo 'alias activate="source $PYENV_ROOT/versions/anaconda3-$ANACONDA_VER/bin/activate"' >> .bashrc

# ライブラリのアップデート
RUN conda update -y conda
RUN pip install --upgrade pip
 
# install opencv
RUN conda install -y ffmpeg
#RUN conda config --remove channels conda-forge
RUN conda config --add channels conda-forge  
RUN conda install -y opencv

RUN conda install -n py2 -y ffmpeg
#RUN conda config --remove channels conda-forge
#RUN conda config -n py2 --add channels conda-forge  
RUN conda install -n py2 -y opencv -c conda-forge
RUN apt-get install -y libgl1-mesa-glx
RUN conda install -n py2 -y scipy scikit-learn cvxopt
#CMD source activate py2
