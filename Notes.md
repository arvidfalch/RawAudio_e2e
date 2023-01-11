# Way forward
## Prelims
Train CRNN and LSTM with both Raw Audio input and log mel spec input at ML Nodes
* Code logmel option into Frontend
* Note computational time and consider this for future process

## Evaluations
* Add PSDS_eval script and function for test set and predictions to pd dataframe
* Add F1 and error score metrics
* Create DESED evaluation set (1000 audio clips)
* Run evaluations on prelim models

## Phase 1
### Hyper parameter search
* Activation function for frontend (now ReLU, CAFx uses Softplus)
* Size of frontend
* Understand Conv_1D_local layer functions
* Timesteps and framesize

* Decide if class 11 ("None") is legal and decide how to negotiate this


Compare results with "A BENCHMARK OF STATE-OF-THE-ART SOUND EVENT DETECTION SYSTEMS
EVALUATED ON SYNTHETIC SOUNDSCAPES
Francesca Ronchini1, Romain Serizel"
* Check results on how many non-target events are detected (as target events) when using a synth_ntg eval set (eval set with only non-target events)
* Use these results to decide on Class 11 problem

## Phase 2
Train and evaluate CRNN and LSTM on Raw Audio Input and Log Mel Spectrograms

## Analysis



