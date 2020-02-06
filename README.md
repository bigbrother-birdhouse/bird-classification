# WTB
Automatic classification of birds for this project : http://www.cabane-oiseaux.org/

## Installation
This project requires python 3.6+
Then, just clone this project and run this command from the main dir :

```
pip install -r requirements.txt
```

## Tensorflow optimization
Depending on your hardware configuration, you can get an optimized version of tensorflow.
I used a pre-compiled version for Intel CPU :

```
pip install https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
```

> If are lucky enough to own a NVIDIA GPU card, you will get best performances with a specific version.
Then, use the "tensorflow-gpu" package with a single `pip install tensorflow-gpu`.

## Data
### Pictures and classes
The training is a supervised one. That means that you must have labellised pictures of birds: each picture of a class must be stored in a folder which name is the class name. All these folders must be stored themselves in a main folder that will be the entry point of the training system.

### Balance the data
As the training system relies on test and training data, the project comes with a tool that can split your original data folder into test, validation and training folders. This tool will also generate new pictures for sub-represented classes (using crop, rotation, ... on original files).
For this, use the `run_prepare_data.py` script from the folder that contains your class folders.
By default, 10% of your original dataset will be used for both test and validation data. 

```bash
python run_prepare_data.py --source=data
```

## Training
The training process relies on this folder structure:

```
photos/
    |__ photos_test/
        |__ accenteur_mouchet/
            |__ photo01.jpg                
            |__ photo02.jpg                
            |__ ...                
        |__ mesange_bleu/
            |__ photo01.jpg                
            |__ photo02.jpg                
            |__ ...                        
        |__ mesange_charbonniere/
            |__ photo01.jpg                
            |__ photo02.jpg                
            |__ ...                
        |__ ...
    |__ photos_train/
        |__ accenteur_mouchet/
            |__ ...                
        |__ mesange_bleu/
            |__ ...                        
        |__ mesange_charbonniere/
            |__ ...                
        |__ ...        
    |__ photos_valid/
        |__ accenteur_mouchet/
            |__ ...                
        |__ mesange_bleu/
            |__ ...                        
        |__ mesange_charbonniere/
            |__ ...                
        |__ ...
```

From such a main folder, you can train a new model from the command line

```
python run_train.py --train_dir=data_train --validation_dir=data_valid
```

This script produces a model that is stored in the `models/` folder with a hdf5 extension.

## Prediction
To predict the class of a given image, one may use the `predict` script:

```
python predict.py --image_path=photos/truc.jpg --model_path=models/bestmodel-09-0.97.hdf5 --mapper_path=models/mapper.json
```

where:
- `photo/truc.jpg` is the file_path of the photo to check
- `models/bestmodel-09-0.97.hdf5` is the file_path of the model to use

## API
An api allows to get a prediction from a given file.
First, launch the api

```
python api.py --model_path=path_to_a_model
```

You can then request the api from : http://localhost:9090/api-birdhouse-1

### Endpoints
```
GET /
GET /heartbeat
GET /supervision
POST /bird --header 'Content-Type: application/x-www-form-urlencoded' --form 'file=@local_path_to_file'
```

### Prediction
The main endpoint allows to predict the class of a given bird picture.

Request
```
curl --location --request POST 'localhost:9090/api-birdhouse-1/bird' \
--header 'Content-Type: application/x-www-form-urlencoded' \
--form 'file=@/home/john/mybird.jpg'
```

Response
```json
{
    "label": "accenteur mouchet",
    "proba": 97.74004817008972
}
```
### OpenApi documentation
The `/doc` route leads to the OpenApi documentation

## Run the notebook
All the machine learning stuff has been developed step by step using a jupyter notebook.
Feel free to run it to have a more summarized view of the deep learning process.
Run jupyter from the project root dir and use your browser to open the file `/notebook/WTB.ipynb`
```
python jupyter notebook
```
