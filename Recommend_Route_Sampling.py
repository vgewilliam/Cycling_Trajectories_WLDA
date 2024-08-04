import numpy as np

Track_Topic_Distribution = np.loadtxt('data/tmp/model_theta.dat').reshape(-1, 3)

#Green: [0.00,0.00,1.00]
#Leisure: [0.60, 0.20, 0.20]

User_Preference_Vector = np.array([0.60, 0.20, 0.20])

# Calculating the Euclidean distance
distances = np.linalg.norm(Track_Topic_Distribution - User_Preference_Vector, axis=1)

# Searching the minimal distance
min_distance = np.min(distances)
min_distance_indices = np.where(distances == min_distance)[0]

if len(min_distance_indices) == 1:
    closest_index = min_distance_indices[0]
else:
    B_max_index = np.argmax(User_Preference_Vector)
    for index in min_distance_indices:
        if np.argmax(Track_Topic_Distribution[index]) == B_max_index:
            closest_index = index
            break 
print("The recommended cycling routes is number:", closest_index+1)
print("The track-topic distribution is:", Track_Topic_Distribution[closest_index])