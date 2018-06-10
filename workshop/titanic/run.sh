docker image build -t titanic .
docker container run --rm -v $(pwd):/src titanic python model_01_separate_file.py
