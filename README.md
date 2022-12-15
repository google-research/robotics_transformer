# Robotics Transformer

*This is not an officially supported Google product.*


This repository is a collection code files and artifacts for running
Robotics Transformer or RT-1.

## Features

* Film efficient net based image tokenizer backbone
* Token learner based compression of input tokens
* Transformer for end to end robotic control
* Testing utilities

## Getting Started

### Installation
Clone the repo
```bash
git clone https://github.com/google-research/robotics_transformer.git
pip install -r robotics_transformer/requirements.txt
python -m robotics_transformer.tokenizers.action_tokenizer.test
```

### Running Tests

To run RT-1 tests, you can clone the git repo and run
[bazel](https://bazel.build/):

```bash
git clone https://github.com/google_research/robotics_transformer.git
cd robotics_transformer
bazel test ...
```

### Using trained checkpoints
Checkpoints are included in trained_checkpoints/ folder for three models:
1. [RT-1 trained on 700 tasks](trained_checkpoints/rt1main)
2. [RT-1 jointly trained on EDR and Kuka data](trained_checkpoints/rt1multirobot)
3. [RT-1 jointly trained on sim and real data](trained_checkpoints/rt1simreal)

They are tensorflow SavedModel files. Instructions on usage can be found [here](https://www.tensorflow.org/guide/saved_model)


## Future Releases

The current repository includes an initial set of libraries for early adoption.
More components may come in future releases.

## License

The Robotics Transformer library is licensed under the terms of the Apache
license.
