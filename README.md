# screamStream

The repository on our work "The Scream Stream: Multimodal Affect Analysis of Horror Game Spaces" presented the **Dungeons, Neurons, and Dialogues: Social Interaction Dynamics in Contextual Games (DnD: SIDC)** Workshop.  

**Contributors:** Emmanouil Xylakis, Antonios Liapis, Georgios N Yannakakis 

Description: Virtual environments allow us to study the impact of space on the emotional patterns of a user as they navigate through it. Similarly, digital games are capable of eliciting intense emotional responses to their players; moreso when the game is explicitly designed to do this, as in 
the Horror game genre. A growing body of literature has already explored the relationship between varying virtual space contexts and user emotion manifestation in horror games, often relying on physiological data or self-reports. In this paper, instead, we study emotion manifestation in this 
game genre from a third-person perspective. Specifically, we analyse facial expressions, voice signals, and verbal narration of YouTube streamers while playing the Horror game Outlast. We document the collection of the Outlast Asylum Affect corpus from in-the-wild videos, and its analysis into three different affect streams based on the streamer's speech and face camera data. These affect streams are juxtaposed with manually labelled gameplay and spatial transitions during the streamer's exploration of the virtual space of the Asylum map of the Outlast game. Results in terms of linear and non-linear relationships between captured emotions and the labelled features demonstrate the importance of a gameplay context when matching affect to level design parameters. This study is the first to leverage state-of-the-art pre-trained models to derive affect from streamers' facial expressions, voice levels, and utterances and opens up exciting avenues for future applications that treat streamers' affect manifestations as in-the-wild affect corpora.

**Published:** in Proceedings of the International Conference on Affective Computing and Intelligent Interaction Workshops and Demos, 2024. 

The repository contains the following:

### Scripts
The `Python_scripts` folder contains the main Python scripts for running the Random Forest classifier (RF) built for the study. It includes two files:
- `LeaveOneOut_RandomForest.py` contains the Random Forest architecture for Hyperparameter tuning and Train-test on the processed data files.
- `RF_defs.py` contains the necessary functions for tuning the RF classifier.

### Data

#### Random Forest Data Files
The `Data/Random_Forest` folder contains all the necessary data to run the `LeaveOneOut_RandomForest.py` script. It includes 10 data files for RF 
train-test on 5 affect dimensions:
- Arousal_VOICE
- Fear_FACE
- Surprise_FACE
- Fear_UTER
- Surprise_UTER

Each affect dimension uses 2 affect signal measures (Mean & Amplitude). Each data file includes:
- Target affect labels (single column with binary values)
- Input features (14 columns for each studied game or level input)
- Streamer ID column

#### Raw Data
The `Data/Raw` folder contains:
1. All the corresponding YouTube video URLs that were used as part of the present dataset.
2. A detailed data file containing the pre-processed affect and feature values with 1Hz resolution (the set sampling resolution used throughout the study).
3. The coding for each feature category on the 41 selected rooms of the Asylum level of Outlast.

### Outlast Maps
The `Outlast_maps` folder includes the 3 floor levels that are part of the Asylum map:
- `Assylum_1F.jpg` and `Assylum_2F.jpg` acquired from the Steam community website: [Steam Community](https://steamcommunity.com/sharedfiles/filedetails/?id=1543156495).
