## MI-reflection-generation
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 1.7.1](https://img.shields.io/badge/pytorch-1.7.1-green.svg?style=plastic)
![CUDA 11.0](https://img.shields.io/badge/cuda-11.0-green.svg?style=plastic)
![License MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=plastic)


This repository contains the implementation of the following paper:

> **Counseling-Style Reflection Generation Using Generative Pretrained Transformers with Augmented Context**<br>
> Siqi Shen, Charles Welch, Rada Mihalcea, Verónica Pérez-Rosas<br>
> https://www.aclweb.org/anthology/2020.sigdial-1.2/
>
> **Abstract:** *We introduce a counseling dialogue system that seeks to assist counselors while they are learning and refining their counseling skills. The system generates counselors’reflections – i.e., responses that reflect back on what the client has said given the dialogue history. Our method builds upon the new generative pretrained transformer architecture and enhances it with context augmentation techniques inspired by traditional strategies used during counselor training. Through a set of comparative experiments, we show that the system that incorporates these strategies performs better in the reflection generation task than a system that is just fine-tuned with counseling conversations. To confirm our findings, we present a human evaluation study that shows that our system generates naturally-looking reflections that are also stylistically and grammatically correct.*


## Resources

Resources related to this work. 
The dataset used in the paper is not shared due to IRB restrictions. However, we have another dataset on Motivational Interviewing linked below. 

- Paper: https://www.aclweb.org/anthology/2020.sigdial-1.2.pdf
- Video: https://www.youtube.com/watch?v=Y9dOYM98rqI&ab_channel=SIGDIAL2020
- MI Dataset: http://web.eecs.umich.edu/~mihalcea/downloads/HighLowQualityCounseling.zip

## System requirements

Refer to [requirement.txt](https://github.com/shensq/MI-reflection-generation/blob/master/requirements.txt) for more details. Run
``` 
pip install -r requirements.txt
```
The code is checked for the following settings.
* Python 3.8 64-bit. 
* Pytorch 1.7.1 or newer with GPU support.
* Transformers v4.0.1 or newer.
* One NVIDIA GPUs with at least 11GB of DRAM. A 2080Ti is used in the experiments. 
* NVIDIA driver 450.57 or newer, CUDA toolkit 11.0 or newer.

## Documentation
### Relevant Response Retrieval
In this step, a model trained on the dataset is used to predict a good match of context and response. A training sample will first be matched with relevant sessions based on tf-idf. Then, the model will give a matching score for each candidate response from a counselor in the relevant sessions. 
The candidate with the highest score will be chosen as the augment response. 
```
cd code && sh run_retrieval.sh
```
* `num_turns`: number of utterances to consider for the matching. 

### Model Finetuning && Text Generation
With the augmented input, a GPT2 model is finetuned with LM loss. The finetuned model is used to generate potential response using top-p sampling.
```
cd code && sh run_finetuning.sh
```
**Finetuning hyper-parameters**
* `model_dir`: The directory of the model to be loaded, the default is 'gpt2'.
* `output_dir`: The output directory where the model predictions and checkpoints will be written.
* `num_train_epocs`: The training epoches.
* `train_batch_size`: The amount of samples in each batch during training.

**Generation hyper-parameters**
* `model_dir`: The directory of the finetuned model.
* `output_dir`: The directory where the generated sentences are stored.
* `length`: The maximum length for generated sentences.
* `temperature`: The temprature applied to softmax probability.
* `top_p`: The model only samples from the set of words whose cumulative probability exceeds the probability p.
## Citation

```bibtex
@inproceedings{shen-etal-2020-counseling,
    title = "Counseling-Style Reflection Generation Using Generative Pretrained Transformers with Augmented Context",
    author = "Shen, Siqi  and
      Welch, Charles  and
      Mihalcea, Rada  and
      P{\'e}rez-Rosas, Ver{\'o}nica",
    booktitle = "Proceedings of the 21th Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2020",
    address = "1st virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.sigdial-1.2",
    pages = "10--20",
}
```

