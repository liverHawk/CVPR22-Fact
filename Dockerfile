FROM mcr.microsoft.com/azure-cli:latest

ENV HTTP_PROXY=http://proxy11.omu.ac.jp:8080/
ENV HTTPS_PROXY=http://proxy11.omu.ac.jp:8080/
ENV http_proxy=http://proxy11.omu.ac.jp:8080/
ENV https_proxy=http://proxy11.omu.ac.jp:8080/

RUN tdnf install -y tar git make

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /home