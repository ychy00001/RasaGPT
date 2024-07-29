# x86 img
FROM rasa/rasa:latest
# mac image
#FROM khalosa/rasa-aarch64:3.5.2

COPY . /app

WORKDIR /app
USER root
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-dotenv rasa-sdk==3.5.0 requests
EXPOSE 5055

CMD ["run", "actions", "--debug"]