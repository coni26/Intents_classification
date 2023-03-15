# Mettre le titre final

Abstract :

>*Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sed neque nisl. Aliquam purus purus, ornare at tellus quis, aliquam finibus ante. Nam elementum volutpat nisl vel maximus. Nullam convallis ipsum ac magna facilisis, sit amet sagittis mauris ullamcorper. In viverra neque at mauris facilisis sollicitudin. Integer quis commodo lacus. Fusce congue, odio nec feugiat pellentesque, risus urna facilisis enim, quis facilisis nisl turpis at eros. Curabitur ut condimentum orci, sed laoreet diam.*


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

