import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import MDS

def plot_pcoa(distance_matrix):
  
    mds = MDS(n_components=2, dissimilarity='precomputed',random_state=51)
    pcoa_coords = mds.fit_transform(distance_matrix)

   
    plt.figure(figsize=(8, 6))
    colors = ['purple', 'teal', 'darkblue', 'orange', 'red'] 
    topics5 = ['Topic 1', 'Topic 2', 'Topic 3','Topic 4','Topic 5']
    topics4 = ['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4']
    topics3 = ['Topic 1', 'Topic 2', 'Topic 3']
    scatter_points = []
    for i in range(pcoa_coords.shape[0]):
        plt.scatter(pcoa_coords[i, 0], pcoa_coords[i, 1], color=colors[i],s=300,label=topics3[i])
        scatter_points.append( plt.scatter(pcoa_coords[i, 0], pcoa_coords[i, 1], color=colors[i], s=80))
   
    for i in range(pcoa_coords.shape[0]):
        for j in range(i+1, pcoa_coords.shape[0]):
            x = [pcoa_coords[i, 0], pcoa_coords[j, 0]]
            y = [pcoa_coords[i, 1], pcoa_coords[j, 1]]
            plt.plot(x, y, color='gray', linestyle='dotted',label="distance",linewidth=1.5)
            plt.text(np.mean(x), np.mean(y), f'{distance_matrix[i, j]:.2f}', ha='right', va='top', fontsize=10)

    plt.xlabel('PCoA 1')
    plt.ylabel('PCoA 2')
    plt.title('Topic distribution')
    plt.grid(ls='--', linewidth=0.25)
    plt.legend(scatter_points, topics3 ,prop = {'size':10}) 

    plt.show()

# Distance Matrix Examples

distanceMatrix3 = np.array([[0, 0.33843359, 0.37169369],
                            [0.33843359, 0, 0.32729066],
                            [0.37169369, 0.32729066, 0]])

plot_pcoa(distanceMatrix3)