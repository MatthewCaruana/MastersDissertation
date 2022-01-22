# Entity Detection

This is for the process of identifying whether or not a token within a sentence/question is an entity

## Mohammed's Implementation

To train the model:

```
python EntityDetection.py --entity_detection_mode LSTM --fix_embed
or
python EntityDetection.py --entity_detection_mode LSTM --fix_embed --no_cuda
```

To test the model:

```
python top_retrieval.py --trained_model [path/to/trained_model.pt] --entity_detection_mode LSTM
```

### For CRF:

enter crf folder and run bash auto_run.sh

## My Implementation

### Training 

To train model:

### Predicition



