import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# Examine the columns, descriptions of the player. Then look at how the type of pitch was recorded, whether ball (B), strike(S), or other (X)
#print(aaron_judge.columns)
#print(aaron_judge.description.unique())
#print(aaron_judge['type'])

# Since we only care about the Balls and Strikes, going to relable these as Ball = 0 and Strike = 1 using a data map
aaron_judge['type'] = aaron_judge['type'].map({'B':0,'S':1})
#print(aaron_judge['type']) # -- Verified

## We want to predict whether the pitch is a ball or strike based on the ball's location on the plate. The X-axis and Z-axis are the two axes of interest, as Y would be the depth into the plate. 
#print(aaron_judge['plate_x']) # 0 denotes the center of the plate, negative is distance to the left and positive to the right -- Verified
#print(aaron_judge['plate_z']) # These should ideally all be positive unless the ball hits the ground before it gets to the plate -- Verified

## Remove all NaN using .dropna().  This can be done using just the subset of the data I'm interested in, particularly the 'type', 'plate_x', and 'plate_z'
aaron_judge = aaron_judge.dropna(subset = ['type', 'plate_x', 'plate_z'])
#print(aaron_judge['type']) # Verified

## Plot the data using a scatter plot
plt.scatter(aaron_judge['plate_x'], aaron_judge['plate_z'], c = aaron_judge['type'], cmap = plt.cm.coolwarm, alpha = 0.25)  # The cmap sets the color map such that the balls are blue and the strikes are red

# We want to create a decision boundary to determine the real strike zone for this player. Set up training and validation sets. 

training_set, validation_set = train_test_split(aaron_judge, train_size = 0.8, test_size = 0.2, random_state = 1)

# Set up the SVC process - since the x and z are both input parameters, this usese the two_column = df[['A', 'B']] setup for the x parameter, and the df['type'] for y input parameter
classifier = SVC(kernel = 'rbf') # Yielded a score of 82% accurate
classifier = SVC(kernel = 'rbf', C = 100, gamma = 100) # Very overfit, but with accuracy of 76% - new accuracy dropped

classifieds = []


## Optimize C and gamma
for n in range(1,10):
  for m in range(1,10):
    classifier = SVC(kernel = 'rbf', C = n, gamma = m)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    classifieds.append([100*classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']), n, m])

#classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

# To visualize the boundary, call the draw_boundary function, which is not part of the scikitlearn package
draw_boundary(ax, classifier)

plt.show()

#print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
print(max(classifieds)) ## this found a max at C = 7, gamma = 1, with the max at 83.1%







####################
# Making a function to perform the data cleaning and processing

def strike_zone(pitcher, color):
  # First - remap the strings to binary numbers
  pitcher['type'] = pitcher['type'].map({'B':0,'S':1})
  # Next - Drop NA values to only view balls and strikes
  pitcher = pitcher.dropna(subset = ['type', 'plate_x', 'plate_z'])
  # Set up scatter plot data
  plt.scatter(pitcher['plate_x'], pitcher['plate_z'], c = pitcher['type'], cmap = color, alpha = 0.15)
  # Establish Training and Validation Sets
  training_set, validation_set = train_test_split(pitcher, train_size = 0.8, test_size = 0.2, random_state = 1)
  # Establish Classifier
  classifier = SVC(kernel = 'rbf', C = 1, gamma = 7)
  # Run the Model Fit
  classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
  # Draw the Boundaries
  draw_boundary(ax, classifier)  # TO be commented out if want to see a cleaner plot overlay with the different people. Also cannot function with more than 2 parameters. 
  
#####################
# Plotting the Data
#######################
fig, ax = plt.subplots()

color1 = plt.cm.coolwarm
color2 = plt.cm.seismic
color3 = plt.cm.PRGn
color4 = plt.cm.PiYG

strike_zone(aaron_judge, color1)  
strike_zone(jose_altuve, color2)
strike_zone(david_ortiz, color3)

ax.set_ylim(-2,6)
ax.set_xlim(-3,3)
plt.show()