# The Importance of Context in Intent Classification: A Comparative Study of Encoder-Decoder Architectures

[Full paper](https://pamplemousse.ensae.fr/index.php?p=100)

Abstract :

>*This study presents a benchmark of various encoder and decoder architectures for intent classification in dialogue systems, using the DailyDialog corpus. The role of context in classification accuracy is explored, with a particular focus on the importance of capturing dynamic structures of context in real-world applications. Our results demonstrate that including context significantly improves classification performance, and that the choice of decoder architecture is important both for their architecture and the level of context they use. We also demonstrated that analyzing accuracy according to the context - past, none, and full - provides valuable insights into the impact of context on real-world applications.*


### Prerequisites
Install the required packages using the following command :
```
pip install -r requirements.txt
```

### Walkthrough

The directory `src` contains the source code to process DailyDialog data with different encoders. The different model architectures are presented in `models.py`. Finally in `utils.py`, we find different functions for training as well as statistics on the models and their performance.

The DataFrame `encoders.csv` contains the different encoders we used with a brief description of their particularity and the size of the embedding. More details on [sbert.net](https://www.sbert.net/docs/pretrained_models.html).

The directory `models` contains all the trained models we used in our analyses. In particular the folder `model_parameters` contains the models for the performance analysis with a fixed number of parameters.

The notebook `result_analysis.ipynb` presents the results of the different models built, as well as analyses to understand the importance of the context but also of the architecture of the model used. Moreover, we also show the superiority of some encoders (gtr-t5-large and all-roberta-large-v1). 

