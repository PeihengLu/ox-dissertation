# ox-dissertation

## Web App Setup

Limited by the scale and funding of this project, the web application is not deployed to a public server at the current stage. However, it can still be tested locally using Python with Django, Pytorch and Tensorflow installed (all requirements are listed in `requirements.txt`) by executing:

```bash
python webtool/manage.py runserver
```

suppose that the user is in the root directory of the project. The web tool can then be accessed by visiting [http://127.0.0.1:8000/](http://127.0.0.1:8000/). 

When testing on personal devices with limited computational resources, HEK293T - PE2 combination is not recommended, as the model needs to first load the large DeepPrime dataset into memory to make predictions, which could take up to 10 minutes.

## Web App Usage

The user can supply their target edit in the PRIDICT format of \{at-lesat-100bp\}-(before-edit/after-edit)-\{at-lesat-100bp\} in the input box, and can also use the example provided by clicking the `Use Test Example` button. The PE and Cell Line should also be selected from the dropdown menu above the text box before running the prediction. Any missing input would result in an error message delivered to the user with browser notification, while format error in the input sequence would likely result in prediction failure, which is also communicated to the user.

<img src='https://github.com/user-attachments/assets/47c54cae-cf78-465b-b756-b335ab843b65' width=650>

The response of the model is parsed into a table of possible pegRNA designs, ranked by their predicted efficiency:

<img src='https://github.com/user-attachments/assets/d3375025-476b-4e33-9c00-e559b919a7f8' width=650>

To better understand the design of each pegRNA, user can then click on the `Visualize Sequence` button to produce a visualization of the pegRNA's relative position to the input sequence

<img src='https://github.com/user-attachments/assets/5f06d918-b18d-4ffc-83df-81fddaa331ef' width=650>
