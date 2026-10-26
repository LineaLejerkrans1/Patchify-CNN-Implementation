import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch 
import copy
import time



# Flip algorithm
aa = np.int32(np.arange(32)).reshape((32, 1))
bb = np.int32(np.arange(31, -1, -1)).reshape((32, 1))
vv = np.tile(32 * aa, (1, 32))
ind_flip = vv.reshape((32*32, 1)) + np.tile(bb, (32, 1))
inds_flip = np.vstack((ind_flip, 1024 + ind_flip))
inds_flip = np.vstack((inds_flip, 2048 + ind_flip)).flatten()


"""
Shuffles the data and splits off a portion to be used as a validation set
"""
def create_validation_split(X, Y, y, val_size=1000):
    n = X.shape[1]
    indices = np.random.permutation(n)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    return X[:, train_idx], Y[:, train_idx], y[train_idx], X[:, val_idx], Y[:, val_idx], y[val_idx]


"""
Load CIFAR-10 binary from a file
"""
def LoadBatch(filename): 
    # Load a batch of training data
    cifar_dir = '/Users/linea/Kth/År4/Deep Learning/Assignment1/Datasets/cifar-10-batches-py/'
    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    # Extract data
    X = dict[b'data']
    y = np.array(dict[b'labels'])

    # Normalize and transpose
    X = X.astype(np.float64) / 255.0
    X = X.transpose()

    # One-hot encoding
    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n))
    
    for i in range(n):
        Y[y[i], i] = 1
    
    return X, Y, y


"""
Transforms hard one-hot labels into smoothed labels.
"""
def applyLabelSmoothing(Y, epsilon=0.1, K=10):
    # y_smooth = (1-eps) if i==y else eps/(K-1)
    Y_smoothed = np.where(Y == 1, 1 - epsilon, epsilon / (K - 1))
    return Y_smoothed


"""
Initializes the network weights (W) and biases (b), drawn from a Guassian distribution
"""
def InitialParams(f, nf, nh):
    K = 10

    rng = np.random.default_rng(42)
    L = 3

    # Calculate the new input dimension for W
    n_p = (32 // f)**2
    d_conv_out = n_p * nf
    
    # Create dict
    init_net = {}
    init_net['W'] = [None] * L
    init_net['b'] = [None] * L

    # First layer: Convolutional Filters
    init_net['W'][0] = rng.standard_normal(size=(f*f*3, nf)) * np.sqrt(2 / (f*f*3))
    init_net['b'][0] = np.zeros((nf, 1))

    # Second layer: fully connected 1
    init_net['W'][1] = rng.standard_normal(size=(nh, d_conv_out)) * np.sqrt(2 / d_conv_out)
    init_net['b'][1] = np.zeros((nh, 1))

    # Third layer: fully connected 2
    init_net['W'][2] = rng.standard_normal(size=(K, nh)) * np.sqrt(2 / nh)
    init_net['b'][2] = np.zeros((K, 1))

    return init_net


"""
Preforms forward pass
Using a first layer of convolution
"""
def ApplyNetwork(mX, net):
    F_flat = net['W'][0]
    W1, W2 = net['W'][1], net['W'][2]
    b1, b2 = net['b'][1], net['b'][2]

    n_p = mX.shape[0] # nr of 
    nf = F_flat.shape[1]
    n = mX.shape[2] # nr of images

    # Convolutional step, 
    V = np.einsum('ijn, jl -> iln', mX, F_flat, optimize=True) 
    V += net['b'][0].reshape(1, -1, 1) # add bias term 

    # Reshape and apply ReLU 
    conv_flat = np.fmax(V.reshape((n_p * nf, n), order='C'), 0)

    # Compute first score
    s1 = W1 @ conv_flat + b1 
    h1 = np.maximum(0, s1) # ReLu

    # Compute secound score
    s2 = W2 @ h1 + b2

    # Stability
    s = s2 - np.max(s2, axis=0, keepdims=True)

    exp_s = np.exp(s)
    P = exp_s / np.sum(exp_s, axis=0)

    return P, {'V': V, 'conv_flat': conv_flat, 'h1': h1}


"""
Compute Cross-entropy loss
"""
def ComputeLoss(P, y):
    # Get the right probobilities
    n = P.shape[1]
    p = P[y, np.arange(n)]

    eps = 1e-15

    # Avoid log(0)
    p = np.maximum(p, eps)

    loss = -np.log(p)
    mean_loss = np.mean(loss)

    return mean_loss


"""
Compares the class with the highest probability against the actual label
"""
def ComputeAccuracy(P, y):
    preds = np.argmax(P, axis=0)
    acc = np.mean(preds == y)

    # Ckeck for correct guesses
    correct = []
    wrong = []

    for i in range(len(y)):
        p_true = P[y[i], i]
        
        if preds[i] == y[i]:
            correct.append(p_true)
        else:
            wrong.append(p_true)

    return acc, correct, wrong


"""
Compute cost = loss + regularization
"""
def ComputeCost(P, y, network, lam):
    # Compute the mean cross-entropy loss 
    l_cross = ComputeLoss(P, y)
    
    # Add L2 regularization
    W_filter, W1, W2 = network['W'][0], network['W'][1], network['W'][2]
    sum_W_sq = (np.sum(W_filter**2) + np.sum(W1**2) + np.sum(W2**2))
    
    cost = l_cross + lam * sum_W_sq

    return cost


"""
Preform backwardpass, computes the gradients
"""
def BackwardPass(mX, Y, P, fp_data, network, lam):
    # Matrix operations for efficiency
    n = Y.shape[1]
    W_filter, W1, W2 = network['W'][0], network['W'][1], network['W'][2]
    
    h1 = fp_data['h1']
    conv_flat = fp_data['conv_flat']
 
    # How many rightly classified
    G = P - Y

    # Compute gradients 
    # Output layer 
    grad_W2 = (1/n) * G @ h1.T + 2 * lam * W2
    grad_b2 = (1/n) * G @ np.ones((n,1))

    # Backprop to next layer
    G = W2.T @ G
    G = G * (h1 > 0)

    # Layer 1
    grad_W1 = (1/n) * G @ conv_flat.T + 2 * lam * W1
    grad_b1 = (1/n) * G @ np.ones((n,1))

    # Backprop to conv layer
    G = W1.T @ G
    G = G * (conv_flat > 0)

    # Reshape 
    n_p = fp_data['V'].shape[0]
    n_f = fp_data['V'].shape[1]
    GG = G.reshape((n_p, n_f, n), order='C')

    # Multiply mX transposed by G_v to get grad_F
    mXt = np.transpose(mX, (1, 0, 2))
    grad_F = np.einsum('jin, iln -> jl', mXt, GG, optimize=True) / n + 2 * lam * W_filter

    grad_b_conv = np.sum(GG, axis=(0, 2)).reshape(-1, 1) / n


    return {
        'W': [grad_F, grad_W1, grad_W2],
        'b': [grad_b_conv, grad_b1, grad_b2]
    }


"""
Algorightm to vary the learning rate cyclically between a minimum value and a maximum value 
"""
def basicCyclicEta(t, eta_min, eta_max, ns):
    cycle = np.floor(1 + t / (2 * ns))
    x = np.abs(t / ns - 2 * cycle + 1)
    eta = eta_min + (eta_max - eta_min) * max(0, (1 - x))
    return eta


"""
Traing loop, updates W and b with computed gradients
"""
def MiniBatchGD(mX_train, Y_train, y_train, mX_val, Y_val, y_val, GDparams, net, lam):
    n = mX_train.shape[2]
    history = {'train_cost': [], 'val_cost': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 't': []}
    
    t_global = 0
    current_ns = GDparams['ns']
    n_cycles = GDparams['n_cycles']

    # Pre-shuffle indices 
    indices = np.random.permutation(n)
    idx_ptr = 0

    # Run trough cycles
    for cycle in range(n_cycles):
        steps_in_cycle = 2 * current_ns

        for t_local in range(steps_in_cycle):
            # Handling batches
            if idx_ptr + GDparams['n_batch'] > n:
                indices = np.random.permutation(n)
                idx_ptr = 0
            
            # Create mini-batches, randomly shuffled
            batch_indices = indices[idx_ptr:idx_ptr + GDparams['n_batch']]
            Xb = mX_train[:, :, batch_indices]
            Yb = Y_train[:, batch_indices]
            idx_ptr += GDparams['n_batch']

            # Forward and backward pass
            P, fp_data = ApplyNetwork(Xb, net)
            grads = BackwardPass(Xb, Yb, P, fp_data, net, lam)

            # Cyclical learning rate
            eta = basicCyclicEta(t_local, GDparams['eta_min'], GDparams['eta_max'], GDparams['ns'])
            
            # Update weights
            for i in range(len(net['W'])):
                net['W'][i] -= eta * grads['W'][i]
                net['b'][i] -= eta * grads['b'][i]

            # Log progress every few steps 
            if t_global % (GDparams['ns'] // 2) == 0:
                # Use a chunk of 5000 images 
                chunk_size = 5000 
                mX_train_chunk = mX_train[:, :, :chunk_size]
                y_train_chunk = y_train[:chunk_size]

                P_t, _ = ApplyNetwork(mX_train_chunk, net)
                P_v, _ = ApplyNetwork(mX_val, net)
                
                # Cost (Loss + Regularization)
                history['train_cost'].append(ComputeCost(P_t, y_train_chunk, net, lam))
                history['val_cost'].append(ComputeCost(P_v, y_val, net, lam))
                
                # Loss
                history['train_loss'].append(ComputeLoss(P_t, y_train_chunk))
                history['val_loss'].append(ComputeLoss(P_v, y_val))
                
                # Accuracy
                acc_t, _, _ = ComputeAccuracy(P_t, y_train_chunk)
                acc_v, _, _ = ComputeAccuracy(P_v, y_val)
                history['train_acc'].append(acc_t)
                history['val_acc'].append(acc_v)
                
                history['t'].append(t_global)
                print(f"Cycle {cycle+1}, Step {t_global}: Val Acc = {acc_v:.4f}")

            t_global += 1

        # Update ns for the next cycle if the flag is set
        if GDparams.get('increasing_ns', False):
            current_ns *= 2
            
    return net, history


"""
Preform random search for lambda in provided intervall
"""
def RandomSearch(trainX, trainY, trainy, valX, valY, valy, GDparams, l_min, l_max):
    best_results = []
    num_tries = 10

    print(f"\nRunning Random Search ({num_tries} trials)")
    for i in range(num_tries):
        # Log-scale random lambda 
        l = l_min + (l_max - l_min) * np.random.rand()
        lam = 10**l
        
        # Train net
        net = InitialParams() 
        trained_net, history = MiniBatchGD(trainX, trainY, trainy, valX, valY, valy, GDparams, net, lam)
        
        # Log accuracy
        max_acc = max(history['val_acc'])
        best_results.append((lam, max_acc))
        print(f"Trial {i+1}: Lambda: {lam:.6f}, Best Val Acc: {max_acc:.4f}")

    # Sort and print top performers
    best_results.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 3 Results")
    for r in best_results[:3]:
        print(f"Lambda: {r[0]:.6f} | Accuracy: {r[1]:.4f}")



"""
Slow implementation of convolution.
X_ims: 4D array, Fs: 4D array of filters
"""
def SlowConvolution(X_ims, Fs):
    # Extract dimensions
    f, _, _, nf = Fs.shape # f - filter size, nf - nr of filters
    _, _, _, n = X_ims.shape # nr of images
    
    # nr of patches on a row/col
    n_p_dim = 32 // f
    # allocate space for output matrix
    conv_output = np.zeros((n_p_dim, n_p_dim, nf, n))

    # Loop over images
    for i in range(n): 
        # Loop over filters
        for k in range(nf): 
            # to visit each non-overlapping sub-patch (strid = f)
            for r_idx, r in enumerate(range(0, 32, f)):
                for c_idx, c in enumerate(range(0, 32, f)):
                    # Extract sub-patch
                    patch = X_ims[r:r+f, c:c+f, :, i]
           
                    # Compute dot product between patch pixel and weight in filter
                    conv_output[r_idx, c_idx, k, i] = np.sum(np.multiply(patch, Fs[:, :, :, k]))
                    
    return conv_output


"""
Constructs MX matrix ones at the start of training
X_ims: 4D array, f: filter size
"""
def Get_MX(X_ims, f):
    n_p_dim = 32 // f
    n_p = n_p_dim**2
    d_patch = f * f * 3
    n = X_ims.shape[3]
    
    # Initialize MX as (nr of patches, flatten patch size, nr of images)
    mX = np.zeros((n_p, d_patch, n))
    
    # Loop though images 
    for i in range(n):
        l = 0 # the l-th sub-region index
        # to visit each non-overlapping sub-patch (strid = f)
        for r in range(0, 32, f):
            for c in range(0, 32, f):
                # Extract patch 
                patch = X_ims[r:r+f, c:c+f, :, i]

                # flatten and fill matrix row 
                mX[l, :, i] = patch.reshape((1, d_patch), order='C')
                l += 1
    return mX


"""
Reshape matrix
"""
def prepare_MX(X, f):
    n = X.shape[1]
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    return Get_MX(X_ims, f)


"""
Test the accuracy of the computation of the convolution by two diffrent approches.
One slow with nested for-loops and one faster with matrix multiplication
"""
def testConvolutionComputations():
    # Test computation of convolution
    # Load the debug data
    debug_file = '/Users/linea/Kth/År4/Deep Learning/Assignmnet3/debug_info.npz'
    load_data = np.load(debug_file)
    X = load_data['X']
    Fs = load_data['Fs']
    n = X.shape[1]
    f = Fs.shape[0]
    nf = Fs.shape[3]

    # Reshape X into 4D images
    X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))

    # Call the slow coputation to get output matrix
    slow_output = SlowConvolution(X_ims, Fs)

    # Compare with results from loaded file 
    expected_output = load_data['conv_outputs']
    relative_error = np.abs(slow_output - expected_output) / np.maximum(1e-15, np.abs(slow_output) + np.abs(expected_output))
    print(f"Max relative error: {np.max(relative_error)}")

    # Create MX matrix
    MX_test = Get_MX(X_ims, f)

    # Flatten filtters
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')

    # Run fast version with einsum
    fast_output = np.einsum('ijn, jl ->iln', MX_test, Fs_flat, optimize=True)

    # Reshape slowoutput to fit the fast 
    slow_reshaped = slow_output.reshape(-1, Fs.shape[3], n)

    # Compare result of the fast approch with the slow
    error = np.abs(fast_output - slow_reshaped).max()
    print(f"Fast vs Slow difference: {error}")


"""
Check that the forward pass is working by comparing it to the debug file
"""
def checkForwardPass():
    load_data = np.load('/Users/linea/Kth/År4/Deep Learning/Assignmnet3/debug_info.npz')
    
    net = {'W': [None]*3, 'b': [None]*3}
    
    # Flatten filters
    Fs = load_data['Fs']
    net['W'][0] = Fs.reshape(-1, Fs.shape[3], order='C') 
    net['b'][0] = np.zeros((Fs.shape[3], 1))
    
    net['W'][1] = load_data['W1']
    net['b'][1] = load_data['b1']
    net['W'][2] = load_data['W2']
    net['b'][2] = load_data['b2']

    # Pre-process MX from debug file
    X_debug = load_data['X']
    n = X_debug.shape[1]
    f = Fs.shape[0]
    X_ims = np.transpose(X_debug.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    MX = Get_MX(X_ims, f)

    # Test
    P, intermediate = ApplyNetwork(MX, net)

    # Compare
    err_P = np.abs(P - load_data['P']).max()
    err_conv = np.abs(intermediate['conv_flat'] - load_data['conv_flat']).max()
    
    print(f"P Error: {err_P}")
    print(f"Conv_flat Error: {err_conv}")


"""
Check that the backward pass with gradients is working by comparing it to the debug file
"""
def checkBackwardPass():
    load_data = np.load('/Users/linea/Kth/År4/Deep Learning/Assignmnet3/debug_info.npz')
    
    net = {'W': [None]*3, 'b': [None]*3}
    
    # Flatten filters
    Fs = load_data['Fs']
    net['W'][0] = Fs.reshape(-1, Fs.shape[3], order='C') 
    net['b'][0] = np.zeros((Fs.shape[3], 1))
    
    net['W'][1] = load_data['W1']
    net['b'][1] = load_data['b1']
    net['W'][2] = load_data['W2']
    net['b'][2] = load_data['b2']

    # Pre-process MX from debug X
    X_debug = load_data['X']
    Y_debug = load_data['Y']

    n = X_debug.shape[1]
    f = Fs.shape[0]
    X_ims = np.transpose(X_debug.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
    MX = Get_MX(X_ims, f)

    # Test
    P, intermediate = ApplyNetwork(MX, net)

    grads = BackwardPass(MX, Y_debug, P, intermediate, net, lam=0)

    # Compare your grad_F with load_data['grad_Fs_flat']
    ga = grads['W'][0]
    gn = load_data['grad_Fs_flat']
    err_F = (np.abs(ga - gn) / np.maximum(1e-15, np.abs(ga) + np.abs(gn))).max()
    print(f"Filter Gradient Error: {err_F}")

    # Compare W1 and W2 gradients
    ga = grads['W'][1]
    gn = load_data['grad_W1']
    err_W1 = (np.abs(ga - gn) / np.maximum(1e-15, np.abs(ga) + np.abs(gn))).max()
    print(f"W1 Gradient Error: {err_W1}")

    ga = grads['W'][2]
    gn = load_data['grad_W2']
    err_W2 = (np.abs(ga - gn) / np.maximum(1e-15, np.abs(ga) + np.abs(gn))).max()
    print(f"W2 Gradient Error: {err_W2}")


"""
Function to plot the bar chart
"""
def PlotBarChart():
    # Example data
    models = ['f=2, nf=3', 'f=4, nf=10', 'f=8, nf=40', 'f=16, nf=160']
    accuracies = [51.01, 56.86, 58.67, 56.75]
    runtimes = [37.49, 31.29, 24.98, 38.59] 

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Chart 1: Performance
    ax1.bar(models, accuracies, color='skyblue')
    ax1.set_title('Final Test Performance (%)')
    ax1.set_ylabel('Accuracy')

    # Chart 2: Run Time
    ax2.bar(models, runtimes, color='salmon')
    ax2.set_title('Training Run Time (s)')
    ax2.set_ylabel('Seconds')

    plt.tight_layout()
    plt.show()


"""
Plots cost and loss over time
"""
def PlotMetrics(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Plot
    ax[0].plot(history['t'], history['train_loss'], label='Train Loss')
    ax[0].plot(history['t'], history['val_loss'], label='Val Loss')
    ax[0].set_title('Cost')
    ax[0].legend()
    
    # Accuracy Plot
    ax[1].plot(history['t'], history['val_acc'], label='Val Acc')
    ax[1].plot(history['t'], history['train_acc'], label='Training Acc')
    ax[1].set_title('Validation Accuracy')
    ax[1].legend()
    
    plt.show()

    # Calculate the gap at the final update step
    avg_train = np.mean(history['train_loss'][-4:])
    avg_val = np.mean(history['val_loss'][-4:])
    print(f"Average Generalization gap: {np.abs(avg_val - avg_train):.4f}")


"""
Plots cost, loss and accuracy over time
"""
def PlotCurves(history):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    steps = history['t']

    # Cost plot 
    ax[0].plot(steps, history['train_cost'], 'g-', label='training', linewidth=1)
    ax[0].plot(steps, history['val_cost'], 'r-', label='validation', linewidth=1)
    ax[0].set_xlabel('update step')
    ax[0].set_ylabel('cost')
    ax[0].set_title('Cost plot')
    ax[0].legend()

    # Loss plot
    ax[1].plot(steps, history['train_loss'], 'g-', label='training', linewidth=1)
    ax[1].plot(steps, history['val_loss'], 'r-', label='validation', linewidth=1)
    ax[1].set_xlabel('update step')
    ax[1].set_ylabel('loss')
    ax[1].set_title('Loss plot')
    ax[1].legend()

    # Accuracy plot
    ax[2].plot(steps, history['train_acc'], 'g-', label='training', linewidth=1)
    ax[2].plot(steps, history['val_acc'], 'r-', label='validation', linewidth=1)
    ax[2].set_xlabel('update step')
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('Accuracy plot')
    ax[2].legend()

    plt.tight_layout()
    plt.show()






def main():
    # Load data
    # Load all 5 batches
    X_list, Y_list, y_list = [], [], []

    for i in range(1, 6):
        X, Y, y = LoadBatch(f'data_batch_{i}')
        X_list.append(X)
        Y_list.append(Y)
        y_list.append(y)

    trainX = np.concatenate(X_list, axis=1)
    trainY = np.concatenate(Y_list, axis=1)
    trainy = np.concatenate(y_list) 

    # Create validation and training split and load test data
    trainX, trainY, trainy, valX, valY, valy = create_validation_split(trainX, trainY, trainy, val_size=1000)
    testX, testY, testy = LoadBatch('test_batch') 

    d = trainX.shape[0]

    # Normlize data
    mean_X = np.mean(trainX, axis=1).reshape(d, 1)
    std_X = np.std(trainX, axis=1).reshape(d, 1)
    std_X[std_X == 0] = 1

    trainX = (trainX - mean_X) / std_X
    valX = (valX - mean_X) / std_X
    testX = (testX - mean_X) / std_X

    # Label smoothing
    trainY_smoothed = applyLabelSmoothing(trainY)

    # Specify parameters
    f = 4 # filter size
    nf = 40 # nr of filters
    nh = 300 # nr of hidden nodes
    lam = 0.0025
    GDparams = {
        'n_batch': 100,
        'eta_min': 1e-5,
        'eta_max': 1e-1,
        'ns': 800, 
        'n_cycles': 4,
        'increasing_ns': False # change to use basic or increasing stepsize
    }

    # Compute convolution at the start before training
    MX_train = prepare_MX(trainX, f)
    MX_val = prepare_MX(valX, f)
    MX_test = prepare_MX(testX, f)

    # Initilaze network
    net = InitialParams(f, nf, nh)

    # Start the clock
    start_time = time.time()

    # Run the training
    net_trained, history = MiniBatchGD(MX_train, trainY_smoothed, trainy, MX_val, valY, valy, GDparams, net, lam)
    #net_trained, history = MiniBatchGD(MX_train, trainY, trainy, MX_val, valY, valy, GDparams, net, lam)

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training took {training_duration:.2f} seconds")

    # Final Evaluation
    P_test, _ = ApplyNetwork(MX_test, net_trained)
    acc, _, _ = ComputeAccuracy(P_test, testy)
    print("\nTest Accuracy:", acc)
    PlotMetrics(history)
    PlotBarChart()









    
if __name__ == "__main__":
    #testConvolutionComputations()
    checkForwardPass()
    checkBackwardPass()
    main()