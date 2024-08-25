import sys
import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder, StandardScaler

columns_to_one_hot_encode = [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 16, 17, 18, 19, 20, 21, 22, 24]
column_names_for_one_hot_encode = [
    "Hospital Service Area",
    "Hospital County",
    "Operating Certificate Number",
    "Permanent Facility Id",
    "Facility Name",
    "Age Group",
    "Zip Code - 3 digits",
    "Race",
    "Ethnicity",
    "Type of Admission",
    "Patient Disposition",
    "APR MDC Code",
    "APR Severity of Illness Description",
    "APR Risk of Mortality",
    "APR Medical Surgical Description",
    "Payment Typology 1",
    "Payment Typology 2",
    "Payment Typology 3",
    "Emergency Department Indicator"
]

columns_to_target_encode = [13, 14, 15]
column_names_for_target_encode = [
    "CCSR Diagnosis Code",
    "CCSR Procedure Code",
    "APR DRG Code",
]

numerical_columns = [7, 10, 23]
numerical_column_names = [
    "Total Costs",
    "Length of Stay",
    "Birth Weight",
]

def target_encode(train, Y, columns):
    target_encoded_columns = []
    target_encoded_column_names = []
    target_encoders = {}
    for col in columns:
        unique_values = np.unique(train[:, col])
        mean_encoded = {val: np.mean(Y[train[:, col] == val]) for val in unique_values}
        target_encoders[col] = mean_encoded
        target_encoded_columns.append(np.vectorize(mean_encoded.get)(train[:, col]))
        target_encoded_column_names.append([f"{column_names_for_target_encode[columns.index(col)]}_{val}" for val in unique_values])
    return np.column_stack(target_encoded_columns), target_encoded_column_names, target_encoders

def encode(data, Y, encoder=None, target_encoder=None):
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_columns = encoder.fit_transform(data[:, columns_to_one_hot_encode])
    else:
        encoded_columns = encoder.transform(data[:, columns_to_one_hot_encode])
        # Handle unknown categories in test set
        for i, categories in enumerate(encoder.categories_):
            col = data[:, columns_to_one_hot_encode[i]]
            unknown_mask = ~np.isin(col, categories)
            if np.any(unknown_mask):
                encoded_columns[unknown_mask, sum(len(c) for c in encoder.categories_[:i]):sum(len(c) for c in encoder.categories_[:i+1])] = 0
    
    non_encoded_columns = data[:, numerical_columns]
    
    if target_encoder is None:
        target_encoded_columns, target_encoded_column_names, target_encoders = target_encode(data, Y, columns_to_target_encode)
    else:
        target_encoded_columns = []
        target_encoded_column_names = target_encoder['column_names']
        target_encoders = target_encoder['encoders']
        for col in columns_to_target_encode:
            col_data = data[:, col]
            mean_encoded = target_encoders[col]
            encoded_col = np.vectorize(mean_encoded.get)(col_data)
            unknown_mask = ~np.isin(col_data, list(mean_encoded.keys()))
            encoded_col[unknown_mask] = 0
            target_encoded_columns.append(encoded_col)
        target_encoded_columns = np.column_stack(target_encoded_columns)
    
    encoded = np.hstack((encoded_columns, target_encoded_columns, non_encoded_columns))
    
    # Generate column names for the encoded columns
    encoded_column_names = []
    for i, categories in enumerate(encoder.categories_):
        col_name = column_names_for_one_hot_encode[i]
        encoded_column_names.extend([f"{col_name}_{category}" for category in categories])
    
    encoded_column_names.extend([name for sublist in target_encoded_column_names for name in sublist])
    encoded_column_names.extend(numerical_column_names)
    
    return encoded, encoded_column_names, encoder, {'column_names': target_encoded_column_names, 'encoders': target_encoders}

def generate_data(train, test):
    X_train = train[:, :-1]
    Y_train = train[:, -1]
    Y_train = (Y_train + 1) / 2
    Y_train = Y_train.astype(np.int64)

    # Replace zero values in the Birth Weight column with the mean of the remaining entries
    birth_weight_col = X_train[:, 23]
    non_zero_birth_weights = birth_weight_col[birth_weight_col != 0]
    mean_birth_weight = np.mean(non_zero_birth_weights)
    birth_weight_col[birth_weight_col == 0] = mean_birth_weight
    X_train[:, 23] = birth_weight_col
    
    X_train, encoded_column_names, encoder, target_encoder = encode(X_train, Y_train)
    X_train = X_train.astype(np.float64)
    
    birth_weight_col = test[:, 23]
    birth_weight_col[birth_weight_col == 0] = mean_birth_weight
    test[:, 23] = birth_weight_col
    unique, counts = np.unique(Y_train, return_counts=True)
    W = np.zeros((X_train.shape[1]+1, len(unique)), dtype=np.float64)

    X_test, _, _, _ = encode(test, Y_train, encoder, target_encoder)
    X_test = X_test.astype(np.float64)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    one = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((one, X_train))
    
    one = np.ones((X_test.shape[0], 1))
    X_test = np.hstack((one, X_test))

    return X_train, Y_train, W, X_test, counts

def g(W,x,j):
    z = np.dot(x,W)
    sm= softmax(z)
    return sm[j]

def loss(X,Y,W,counts):
    n = X.shape[0]
    Z = X @ W
    softmax_probs = softmax(Z, axis=1)
    indices = (np.arange(n), Y)
    correct_class_probs = softmax_probs[indices]
    scaled_probs = np.log(correct_class_probs) / counts[Y]
    loss_value = -np.mean(scaled_probs) / 2
    return loss_value

def compute_gradient(X, Y, W, counts):
    n, m = X.shape
    k = W.shape[1]
    z = X @ W  
    softmax_probs = softmax(z, axis=1) 
    indices = (np.arange(n), Y)
    Y_one_hot = np.zeros((n, k))
    Y_one_hot[indices] = 1
    grad_W = X.T @ ((softmax_probs - Y_one_hot) / counts[Y][:, np.newaxis]) / (2 * n) 
    return grad_W

def compute_n(X, Y, W, gradient, n0, counts):
    nl = 0.0
    nh = n0
    prev_loss= loss(X, Y, W, counts)
    while prev_loss > loss(X, Y, W - nh*gradient, counts):
        nh *= 2
        prev_loss= loss(X, Y, W - nh*gradient, counts)
    if nh>n0:
        nl= nh/2
    else:
        while loss(X, Y, W, counts) < loss(X, Y, W - nh*gradient, counts):
            nh /= 2
        nh *= 2
    for _ in range(5):
        n1 = (2*nl + nh)/3
        n2 = (nl + 2*nh)/3
        if loss(X, Y, W - n1*gradient, counts) > loss(X, Y, W - n2*gradient, counts):
            nl = n1
        else:
            nh = n2
    return (nl+nh)/2

def gradient_descent(X, Y, W, counts, n0, epochs, batch_size):
    n = X.shape[0]
    for _ in range(epochs):
        batch_num = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            Y_batch = Y[start:end]
            batch_loss = loss(X_batch, Y_batch, W, counts)
            print(f"Epoch{_+1}, Batch{1+int(start/batch_size)}, Loss{batch_loss}")

            gradient = compute_gradient(X_batch, Y_batch, W, counts)
            learning_rate= compute_n(X_batch, Y_batch, W, gradient, n0, counts)
            W -= learning_rate * gradient
            batch_num += 1
    return W

def write(array,file):
    np.savetxt(file, array, delimiter='\n')

def read_csv(file):
    array = np.loadtxt(file, skiprows=1, delimiter=',')
    return array

def read(file):
    array = np.loadtxt(file)
    return array

if (len(sys.argv) != 4):
    print("Usage: python3 logistic_competitive.py <train_file> <test_file> <output_file>")
    exit()

train= read_csv(sys.argv[1])
test = read_csv(sys.argv[2])
X, Y, W, X_test, counts = generate_data(train, test)
W = gradient_descent(X,Y,W,counts,1e4,25,87595)
Z = X_test @ W
output_model_pred = softmax(Z, axis=1)
predicted_class = np.argmax(output_model_pred, axis=1)
predicted_class[predicted_class == 0] = -1
write(predicted_class,sys.argv[3])