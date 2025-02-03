**Purpose**
We can working with docker as standard environment, cross platform. Also compatible with cloud notebook as SageMaker

**Build docker image**

```bash
# run from root of project
docker build -t <image-name>:<tag> -f devops/images/training/dockerfile .
```

**Run docker image**

```bash
# run from root of project
docker run -it -p 8888:8888 --gpus all <image-name>:<tag>
```