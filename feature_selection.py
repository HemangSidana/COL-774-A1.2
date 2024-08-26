import sys
import numpy as np

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

def create_features(train):
    data = train[:,:-1]
    created_features = []   # list of names of created features (as strings)
    for col in columns_to_one_hot_encode:
        unique_values = np.unique(data[:, col])
        for val in unique_values:
            created_features.append(f"One_Hot_{column_names_for_one_hot_encode[columns_to_one_hot_encode.index(col)]}_{val}")
    for col in column_names_for_target_encode:
        created_features.append(f"Target_Encoded_{col}")
    for col in numerical_column_names:
        created_features.append(f"Numerical_{col}")
    return created_features


def write(array,file):
    np.savetxt(file, array, fmt='%s', delimiter='\n')

def read_csv(file):
    array = np.loadtxt(file, skiprows=1, delimiter=',')
    return array

def read(file):
    array = np.loadtxt(file)
    return array

if (len(sys.argv) != 4):
    print("Usage: python3 feature_selection.py <train_csv> <created_txt> <selected_txt>")
    exit(1)

train = read_csv(sys.argv[1])
created = create_features(train)
selected = [1]*len(created)
write(created, sys.argv[2])
write(selected, sys.argv[3])