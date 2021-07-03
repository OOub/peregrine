from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
from IPython.display import HTML
from scipy.special import logsumexp
    
# normalise data by dividing by min: gives ratio of increase compared to smallest value
def normalize_data(data):
    return data / np.min(data)

# read parameters of the model at every EM iteration
def read_parameters(save_dir):

    # check that the parameters.npy file exists (created by the C++ gmm library)
    parameter_path = save_dir/'parameters.npy'
    if parameter_path.exists():
        # read the parameters.npy
        tmp = np.load(parameter_path).T
        
        # create a rec array
        return np.core.records.fromarrays([tmp[0].astype(int), tmp[1], tmp[2]], names='iteration,free_energy,sigma')
    
# plots an animation of the centers at every iteration
def plot_center_evolution(save_dir, ground_truth='', save_figure=True):
    
    # checking the the centers folder exists (created by the C++ gmm library)
    center_path = save_dir/'centers'
    if center_path.exists():
        
        # reading centers
        centers = [np.load(filename) for filename in center_path.iterdir()]
        
        # plotting ground truth
        fig, ax = plt.subplots()
        if len(ground_truth) != 0:
            gt_center = np.load(ground_truth)
            ax.plot(gt_center[:,0],gt_center[:,1], marker='x', ls='', markersize=10)
            legend_name = ['ground truth', 'prediction']
        else:
            legend_name = ['prediction']

        # plotting final centers as an initialisation
        pred, = ax.plot(centers[-1][:,0], centers[-1][:,1], marker='.', ls='', markersize=10)
        plt.legend(legend_name,loc='upper right', fontsize=12)
        plt.tick_params(labelsize=12)
        plt.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y',fontsize=12)
        plt.tight_layout()
        
        # save final centers vs ground truth (if present)
        if save_figure:
            fig.savefig(save_dir/'center_evolution.pdf')
          
        # function that updates each frame of the animation with corresponding iteration centers
        def update(frame):
            pred.set_data(frame[:,0], frame[:,1])
            return pred,

        # create the animation
        anim = animation.FuncAnimation(fig, update, frames=centers, blit=True, repeat=True)
        rc('animation', html='html5')

        # save animation
        if save_figure:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            anim.save(save_dir/'center_evolution.mp4', writer=writer)
        
        return anim
    
    else:
        print('no centers folder in the save directory')
    
# create histograms used for classification from hard clusters
# requires tr_count.npy or te_count.npy (number of features in each data point)
# requires tr_labels.npy or te_labels.npy (assigned labels)
# if using spatial features requires tr_cells.npy or te_cells.npy (histogram index for each datapoint)
def create_histograms(save_dir, training_path, n_centers, train, normalise=False):
    
    # selecting files for training or test set depending on train argument
    dataset_type = 'tr_' if train else 'te_'
    
    # check that the required files exist    
    count_path = save_dir/str(dataset_type+'count.npy')
    cells_path = save_dir/str(dataset_type+'cells.npy')
    if count_path.exists() and cells_path.exists():
        X = []
        Y = []
        
        # find how many cells there are
        n_cells = int(np.loadtxt(training_path+'/header.txt')[-1])
        
        # reading npy files (created by the C++ gmm library)
        lengths = np.load(count_path)
        cells = np.load(save_dir/str(dataset_type+'cells.npy'))
        labels = np.load(save_dir/str(dataset_type+'labels.npy'))
        
        # splitting labels according to the number of features of every data point 
        running_index = 0
        for n in lengths:
            assert (labels[running_index:running_index + n, 0]==labels[running_index, 0]).all()
            tmp_label = labels[running_index:running_index + n, 1:]
            tmp_cell = cells[running_index:running_index + n]
            
            hists = []
            labs = []
            for i in np.arange(n_cells):
                selection = (tmp_cell == i)

                if len(tmp_label[selection]) > 0:
                    if normalise:
                        hists.append(np.histogram(tmp_label[selection], bins=np.arange(0, n_centers+1))[0] / (tmp_label[selection].shape[0]*tmp_label[selection].shape[1]))
                    else:
                        hists.append(np.histogram(tmp_label[selection], bins=np.arange(0, n_centers+1))[0])
                else:
                    hists.append(np.zeros(n_centers))
            X.append(np.concatenate(hists))
            Y.append(labels[running_index, 0])
            running_index += n
    
        return np.array(X), np.array(Y)
    
    elif count_path.exists():
        
        # reading npy files (created by the C++ gmm library)
        lengths = np.load(count_path)
        labels = np.load(save_dir/str(dataset_type+'labels.npy'))
        
        # splitting labels according to the number of features of every data point 
        running_index = 0
        parsed_labels = []
        cluster_assignments = []
        for n in lengths:
            parsed_labels.append(labels[running_index, 0])
            assert (labels[running_index:running_index + n, 0]==labels[running_index, 0]).all()
            cluster_assignments.append(labels[running_index:running_index + n, 1:])
            running_index += n

        # when not using a dataset (single file npy data) you'll need to provide your own labels as this will return an empty list  
        if normalise:    
            return np.array([np.histogram(x, bins=np.arange(0, n_centers+1))[0]/((x.shape[0]*x.shape[1])) for x in cluster_assignments]), np.array(parsed_labels)
        else:
            return np.array([np.histogram(x, bins=np.arange(0, n_centers+1))[0] for x in cluster_assignments]), np.array(parsed_labels)
    else:
        raise Exception('Required files are not available')

# create soft features for classification from soft clusters
# requires tr_count.npy or te_count.npy (number of features in each data point)
# requires tr_labels.npy or te_labels.npy (assigned labels)
# if using spatial features requires tr_cells.npy or te_cells.npy (histogram index for each datapoint)
def create_soft_features(save_dir, training_path, n_centers, train):     
    # selecting files for training or test set depending on train argument
    dataset_type = 'tr_' if train else 'te_'
    X = []
    Y = []
    # check that the required files exist    
    count_path = save_dir/str(dataset_type+'count.npy')
    cells_path = save_dir/str(dataset_type+'cells.npy')
    if count_path.exists() and cells_path.exists():
        
        # find how many cells there are
        n_cells = int(np.loadtxt(training_path+'/header.txt')[-1])
        
        # reading npy files (created by the C++ gmm library)
        lengths = np.load(count_path)
        cells = np.load(save_dir/str(dataset_type+'cells.npy'))
        tmp_labels = np.load(save_dir/str(dataset_type+'labels_soft.npy'))
        labels = tmp_labels[:,0]
        features = tmp_labels[:,1:]
        
        # splitting labels according to the number of features of every data point 
        running_index = 0
        for n in lengths:
            assert (labels[running_index:running_index + n]==labels[running_index]).all()
            
            tmp_label = labels[running_index:running_index + n]
            tmp_features = features[running_index:running_index + n,:]
            tmp_cell = cells[running_index:running_index + n]
            
            hists = []
            labs = []
            for i in np.arange(n_cells):
                selection = (tmp_cell == i)
                
                if len(tmp_label[selection]) > 0:
                    these_features = tmp_features[selection]
                    hist = these_features.sum(0)
                else:
                    hist = np.zeros((n_centers))
                hists.append(hist)
            
            X.append(np.concatenate(hists))
            Y.append(labels[running_index])
            running_index += n
        
        return np.array(X), np.array(Y)
            
    elif count_path.exists():
        # reading npy files (created by the C++ gmm library)
        lengths = np.load(count_path)
        tmp_labels = np.load(save_dir/str(dataset_type+'labels_soft.npy'))
        labels = tmp_labels[:,0]
        features = tmp_labels[:,1:]
        
        # splitting labels according to the number of features of every data point 
        running_index = 0
        for n in lengths:
            assert (labels[running_index:running_index + n]==labels[running_index]).all()
            
            tmp_label = labels[running_index:running_index + n]
            tmp_features = features[running_index:running_index + n,:]
            tmp_features = np.exp(tmp_features)
            tmp_features /= tmp_features.sum(1)[:,None]
            
            hists = []
            labs = []
            X.append(tmp_features.sum(0))
            Y.append(labels[running_index])
            running_index += n  
        return np.array(X), np.array(Y)
    
    else:
        raise Exception('Required files are not available')

# goes through all the ground truth centers
# calculates the distance with all the algorithm centers
# estimates the error with the nearest center
# adds it to a cumulative error
# removes the center from the list of centers
# and iterate this with all the rest of algorithm centers
def perm_diff(gt,alg, verbose=False):
    assert gt.shape[0]==alg.shape[0]
    nc = gt.shape[0]
    l = list(range(nc))
    all_err=0
    for i in range(nc):
        al = np.array(l)
        this_c = gt[i][np.newaxis,:]
        
        err = np.sqrt(np.sum((this_c - alg[al])**2,axis=1))
        idx = np.argmin(err)
        l.remove(al[idx])
        if verbose:
            print(this_c)
            print(err[idx])
        if np.isnan(err[idx]):
            err[idx]=0
        all_err+=err[idx]
    return all_err
