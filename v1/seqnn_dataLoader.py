"""
SEQNN Data Loader (PyTorch Compatible)
======================================
Data loading utilities for the SEQNN model.

Supports:
- SAT-6 dataset
- So2Sat LCZ42 dataset  
- Overhead-MNIST dataset
- CIFAR-10 dataset

Original implementation from: https://github.com/zhu-xlab/SEQNN
"""

import numpy as np
import scipy.io
import os
import pickle
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
# DATA LOADER FUNCTIONS
# =============================================================================

def get_sat_data(path):
    """Load and preprocess SAT-6 dataset."""
    data = scipy.io.loadmat(path)
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    annotations = data['annotations']

    def reshape(train_x, train_y):
        out_x = np.zeros((train_x.shape[3], train_x.shape[0], train_x.shape[1], train_x.shape[2]))
        out_y = np.zeros((train_y.shape[1], train_y.shape[0]))
        for i in range(train_x.shape[3]):
            out_x[i,:,:,:] = train_x[:,:,:,i]
        for i in range(train_y.shape[1]):
            out_y[i,:] = train_y[:,i]
        return out_x, out_y
    
    def relabel(train_y, annotations):
        labels = {}
        for annotation in annotations:
            labels[annotation[0][0]] = annotation[1][0]
        output = []
        for i in range(train_y.shape[0]):
            temp = ''.join([str(int(x)) for x in train_y[i]])
            output.append(labels[temp])
        return np.array(output)
        
    def samples(train_x, train_y, n, labels):
        out_x, out_y = [], []
        for label in labels:
            temp_x, temp_y = [], []
            for i in range(train_y.shape[0]):
                if label == train_y[i]:
                    temp_x.append(train_x[i])
                    temp_y.append(label)
            temp_x, temp_y = shuffle(np.array(temp_x), np.array(temp_y), random_state=33)
            out_x = out_x + list(temp_x[:n])
            out_y = out_y + list(temp_y[:n])
        return shuffle(np.array(out_x), np.array(out_y), random_state=33)

    def normalize(img):
        img = img.astype('float64')
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    def iqr(image):
        for i in range(image.shape[2]):
            boundry1, boundry2 = np.percentile(image[:,:,i], [2, 98])
            image[:,:,i] = np.clip(image[:,:,i], boundry1, boundry2)
        return image
    
    def data_process(imgs, labels):
        labels = np.array([str(label).strip() for label in labels])
        processed_img = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = normalize(img)
            img = iqr(img)
            # Pad to 32x32 if needed (SAT-6 is 28x28)
            if img.shape[0] < 32:
                img = np.stack([np.pad(img[:, :, c], [(2, 2), (2, 2)], mode='constant') for c in range(4)], axis=2)
            processed_img.append(img)
        return np.array(processed_img), labels    
    
    train_x, train_y = reshape(train_x, train_y)
    test_x, test_y = reshape(test_x, test_y)
    train_y = relabel(train_y, annotations)
    test_y = relabel(test_y, annotations)
    labels = np.unique(train_y, return_counts=False)
    train_x, train_y = samples(train_x, train_y, 900, labels)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=1200, random_state=33)
    test_x, test_y = samples(test_x, test_y, 200, labels)
    
    train_x, train_y = data_process(train_x, train_y)
    valid_x, valid_y = data_process(valid_x, valid_y)
    test_x, test_y = data_process(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_lcz_data(path):
    """Load and preprocess So2Sat LCZ42 dataset."""
    rawdata = scipy.io.loadmat(path)
    data = rawdata['setting0']
    train_x = data['train_x'][0][0]
    train_y = data['train_y'][0][0][0]
    test_x = data['test_x'][0][0]
    test_y = data['test_y'][0][0][0]
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=2000, random_state=42)

    def iqr(image):
        for i in range(image.shape[2]):
            boundry1, boundry2 = np.percentile(image[:,:,i], [2, 98])
            image[:,:,i] = np.clip(image[:,:,i], boundry1, boundry2)
        return image

    def normalize(img):
        img = img.astype('float64')
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    def data_process(imgs, labels):
        labels = np.array([str(label).strip() for label in labels])
        processed_img = []
        for i in range(imgs.shape[0]):
            img, _ = imgs[i]
            img = iqr(img)
            img = normalize(img)
            processed_img.append(img)
        return np.array(processed_img), labels
    
    train_x, train_y = data_process(train_x, train_y)
    valid_x, valid_y = data_process(valid_x, valid_y)
    test_x, test_y = data_process(test_x, test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_overhead_data(path):
    """Load and preprocess Overhead-MNIST dataset."""
    def normalize(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    
    def iqr(image):
        boundry1, boundry2 = np.percentile(image, [2, 98])
        image = np.clip(image, boundry1, boundry2)
        return image
    
    def load_images(folder, label):
        images = []
        if not os.path.exists(folder):
            return [], []
        for filename in os.listdir(folder):
            img = mpimg.imread(os.path.join(folder, filename))
            img = normalize(img)
            img = iqr(img)
            if img.shape[0] < 32:
                img = np.pad(img, [(2, 2), (2, 2)], mode='constant')
            img = img.reshape(32, 32, 1)
            images.append(img)
        return images, [label] * len(images)
    
    train_x, train_y, test_x, test_y = [], [], [], []
    training_path = os.path.join(path, 'training')
    test_path = os.path.join(path, 'testing')
    labels = ['car', 'ship', 'plane', 'harbor', 'parking_lot']
    
    for label in labels:
        temp_training_path = os.path.join(training_path, label, '')
        temp_test_path = os.path.join(test_path, label, '')
        
        temp_x, temp_y = load_images(temp_training_path, label)
        train_x = train_x + temp_x
        train_y = train_y + temp_y
        
        temp_x, temp_y = load_images(temp_test_path, label)
        test_x = test_x + temp_x
        test_y = test_y + temp_y
        
    train_x, train_y = shuffle(np.array(train_x), np.array(train_y), random_state=33)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.15, random_state=33)
    test_x, test_y = shuffle(np.array(test_x), np.array(test_y), random_state=33)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def get_cifar10_data(root_path):
    """Load CIFAR-10 dataset from pickle files."""
    def load_batch(f_path):
        with open(f_path, 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3) for NHWC
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            Y = np.array(Y)
            return X, Y

    # Load Training Batches (1-5)
    x_train_list = []
    y_train_list = []
    for i in range(1, 6):
        f = os.path.join(root_path, 'data_batch_%d' % i)
        if os.path.exists(f):
            X, Y = load_batch(f)
            x_train_list.append(X)
            y_train_list.append(Y)
    
    if not x_train_list:
        raise FileNotFoundError(f"No CIFAR-10 data batches found in {root_path}")

    train_x = np.concatenate(x_train_list)
    train_y = np.concatenate(y_train_list)
    
    # Load Test Batch
    test_path = os.path.join(root_path, 'test_batch')
    if os.path.exists(test_path):
        test_x, test_y = load_batch(test_path)
    else:
        test_x, test_y = np.array([]), np.array([])
    
    # Normalize
    train_x = train_x.astype('float32') / 255.0
    test_x = test_x.astype('float32') / 255.0
    
    # Validation split
    valid_x = train_x[-5000:]
    valid_y = train_y[-5000:]
    train_x = train_x[:-5000]
    train_y = train_y[:-5000]

    return train_x, train_y, valid_x, valid_y, test_x, test_y


# =============================================================================
# SYNTHETIC DATA GENERATOR (for testing without real datasets)
# =============================================================================

def get_synthetic_data(n_classes=6, n_channels=4, n_train=1000, n_valid=200, n_test=200):
    """
    Generate synthetic data for testing the model without real datasets.
    
    Args:
        n_classes: Number of classes
        n_channels: Number of image channels
        n_train: Number of training samples
        n_valid: Number of validation samples
        n_test: Number of test samples
        
    Returns:
        Tuple of (train_x, train_y, valid_x, valid_y, test_x, test_y)
    """
    np.random.seed(42)
    
    def generate_samples(n_samples):
        # Generate random images (32x32 with n_channels)
        x = np.random.rand(n_samples, 32, 32, n_channels).astype(np.float32)
        # Generate random labels
        y = np.random.randint(0, n_classes, n_samples)
        y_str = np.array([str(label) for label in y])
        return x, y_str
    
    train_x, train_y = generate_samples(n_train)
    valid_x, valid_y = generate_samples(n_valid)
    test_x, test_y = generate_samples(n_test)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y


# =============================================================================
# MAIN CLASS
# =============================================================================

class DataLoader:
    """
    DataLoader class matching original implementation.
    
    Supports multiple Earth Observation datasets:
    - 'sat': SAT-6 dataset (28x28 -> 32x32, 4 channels, 6 classes)
    - 'lcz': So2Sat LCZ42 dataset (32x32, 4 channels, 5 classes)
    - 'overhead': Overhead-MNIST dataset (28x28 -> 32x32, 1 channel, 5 classes)
    - 'cifar10': CIFAR-10 dataset (32x32, 3 channels, 10 classes)
    - 'synthetic': Synthetic data for testing
    """
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        
        # Initialize variables
        train_x, train_y = None, None
        valid_x, valid_y = None, None
        test_x, test_y = None, None

        if dataset == 'sat':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_sat_data('Data/SAT-6/sat-6-full.mat')
        elif dataset == 'lcz':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_lcz_data('Data/LCZ/data_5fold_5classes.mat')
        elif dataset == 'overhead':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_overhead_data('Data/overhead')
        elif dataset == 'cifar10':
            path = 'Data/cifar-10-batches-py'
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_cifar10_data(path)
        elif dataset == 'synthetic':
            train_x, train_y, valid_x, valid_y, test_x, test_y = get_synthetic_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset}. "
                           f"Supported: 'sat', 'lcz', 'overhead', 'cifar10', 'synthetic'")

        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.test_y = test_y   
        
    def get_categories(self):
        """Get unique class names."""
        class_name = [str(x).strip() for x in np.unique(self.train_y)]
        return class_name
    
    def get_data(self): 
        """Get data with one-hot encoded labels."""
        # One-hot encode labels
        lb = LabelBinarizer()
        lb.fit(np.concatenate([self.train_y, self.valid_y, self.test_y]))
        
        train_y = lb.transform(self.train_y)
        valid_y = lb.transform(self.valid_y)
        test_y = lb.transform(self.test_y)
        
        # Handle binary classification case
        if train_y.shape[1] == 1:
            train_y = np.hstack([1 - train_y, train_y])
            valid_y = np.hstack([1 - valid_y, valid_y])
            test_y = np.hstack([1 - test_y, test_y])
        
        return self.train_x, train_y, self.valid_x, valid_y, self.test_x, test_y
    
    def get_info(self):
        """Get dataset information."""
        return {
            'dataset': self.dataset,
            'n_train': len(self.train_x),
            'n_valid': len(self.valid_x),
            'n_test': len(self.test_x),
            'input_shape': self.train_x.shape[1:],
            'n_classes': len(self.get_categories()),
            'classes': self.get_categories()
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    # Test with synthetic data
    print("Testing DataLoader with synthetic data...")
    loader = DataLoader('synthetic')
    train_x, train_y, valid_x, valid_y, test_x, test_y = loader.get_data()
    
    print(f"\nDataset info:")
    for key, value in loader.get_info().items():
        print(f"  {key}: {value}")
    
    print(f"\nData shapes:")
    print(f"  train_x: {train_x.shape}, train_y: {train_y.shape}")
    print(f"  valid_x: {valid_x.shape}, valid_y: {valid_y.shape}")
    print(f"  test_x: {test_x.shape}, test_y: {test_y.shape}")
    
    print("\nDataLoader test passed!")
