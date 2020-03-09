docker run --runtime=nvidia -it -v $PWD:/tmp -w /tmp pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime bash /tmp/train_script.sh
