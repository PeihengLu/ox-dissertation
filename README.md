# ox-dissertation

## Web App Usage

Limited by the scale and funding of this project, the web application is not deployed to a public server at the current stage. However, it can still be tested locally using Python with Django, Pytorch and Tensorflow installed (all requirements are listed in `requirements.txt`) by executing:

```bash
python webtool/manage.py runserver
```

suppose that the user is in the root directory of the project. The web tool can then be accessed by visiting [http://127.0.0.1:8000/](http://127.0.0.1:8000/). 

When testing on personal devices with limited computational resources, HEK293T - PE2 combination is not recommended, as the model needs to first load the large DeepPrime dataset into memory to make predictions, which could take up to 10 minutes.
