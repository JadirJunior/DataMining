# DataMining Project

# Data Base
A Project using a Mushrooms DataSet available in: https://archive.ics.uci.edu/dataset/73/mushroom

https://www.kaggle.com/code/sahistapatel96/mushroom-classification

<div style="display: inline-block">
  <iframe src="https://www.kaggle.com/embed/sahistapatel96/mushroom-classification?cellIds=4&kernelSessionId=51326635" height="300" style="margin: 0 auto; width: 100%; max-width: 950px;" frameborder="0" scrolling="auto" title="Mushroom Classification"></iframe>
</div> <br />
# Objective
- Find out which mushrooms are poisonous and which are edible

# Columns
- Poisonous -> Dependent Variable
- Cap_Shape 
- Cap_Surface
- Cap_Color
- Bruises
- Odor
- Gill_Attachment
- Gill_Spacing
- Gill_Size
- Gill_Color
- Stalk_Shape
- Stalk_Root
- Stalk_Surface_Above_Ring
- Stalk_Surface_Bellow_Ring
- Stalk_Color_Above_Ring
- Stalk_Color_Bellow_Ring
- Veil_Type
- Veil_Color
- Ring_Number
- Ring_Type
- Spore_Print_Color
- Population
- Habitat

Each variable describes a particular attribute of a Mushroom, describing your format, type, color and other things.
You can see the meaning of each column bellow:

- Poisonous: 'Poisonous', 'Edible'
- Cap_Shape: 'bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'
- Cap_Surface: 'fibrous', 'grooves', 'scaly', 'smooth'
- Cap_Color: 'brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'
- Gill_Attachment: 'attached', 'descending', 'free', 'notched'
- Gill_Spacing: 'close', 'crowded', 'distant'
- Gill_Size: 'broad', 'narrow'
- Gill_Color: 'brown', 'buff', 'gray', 'green', 'pink', 'purple', 'orange', 'chocolate', 'black', 'red', 'white', 'yellow'
- Stalk_Shape: 'enlarging', 'tapering'
- Stalk_Surface_Above_Ring: 'fibrous', 'scaly', 'silky', 'smooth'
- Stalk_Surface_Bellow_Ring: 'fibrous', 'scaly', 'silky', 'smooth'
- Stalk_Color_Above_Ring: 'brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'
- Stalk_Color_Bellow_Ring: 'brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'
- Veil_Color: 'brown', 'orange', 'white', 'yellow'
- Ring_Number: 'None', 'One', 'Two'
- Ring_Type: 'cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'
- Bruises: 'True', 'False'
- Odor: 'almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'
- Spore_Print_Color: 'black', 'brown', 'buff', 'chocoloate', 'green', 'Orange', 'Purple', 'White', 'Yellow'
- Population: 'abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'
- Habitat: 'grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'
- Veil_Type: 'partial', 'universal'

# Descriptive Analysis
An Analysis to describe the most influent characteristics of a Poisonous or Edible Mushrooms.

### Odor
- Pungent, Foul, Creosote, Fishy, Spicy and Musty are 100% Poisonous;
- Almond, Anise are 100% Edible;
- None: 96.59% Edible/3.401% Poisonous

### Stalk_Color_Above_Ring
- Gray, Red and Orange are 100% Edible;
- Cinnamon, Yellow, and Buff are 100% Poisonous;
- Brown: 96,42% Poisonous/3.571% Edible;
- White: 38.351% Poisonous/61.64% Edible;

### Veil_Color
- Brown and Orange are 100% Edible;
- Yellow are 100% Poisonous;
- White: 49.318% Poisonous/50.681% Edible

### Ring_Type
- Large and none are 100% Poisonous;
- Flaring is 100% Edible;
- Pendant: 79.43% Edible/20.564% Poisonous;
- Evanescent: 63.688% Poisonous/36.31% Edible;

### Population
- Numerous and Abundant are 100% Edible;
- Scattered: 70.51% Edible/29.487% Poisonous;
- Several: 70.49% Poisonous/29.504% Edible;
- Solitary: 62.149% Edible/37.85% Poisonous;
- Clustered: 84.70% Edible/15.29% Poisonous