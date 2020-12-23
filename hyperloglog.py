import pandas as pd
import numpy as np


def string_to_binary_hash(string, p=17, prime_value=(2**64)-1):
    """
    First Hash function that converts strings (User ID's) into binary values of 64 binary values

    :param string: Hexadecimal string that will be converted into a binary value. We assume that this string has been
    randomly generated
    :param p: Closest prime value to 16 (will be used to generate the coefficients)
    :param prime_value: Prime value closest to 2**64 (64 because of the length of our final binary value)
    :return: Binary value. It is important, and has been checked, that the probability of obtaining a 0 or 1 is
    uniformly distributed over the whole hashed value
    """
    letter_list = np.array([int(letter, 16) for letter in string])
    coefficients = np.array([p**i for i in range(len(letter_list))])
    hash_value = np.sum(letter_list*coefficients)%prime_value
    return format(hash_value, 'b').zfill(64)


def decimal_to_binary(integer, decimals=None):
    """
    Function that converts an integer into a binary. This function will be used to compute the groups used to classify
    the different binary values

    :param integer: Integer
    :param decimals: Length of final binary value
    :return: Binary value
    """
    binary = ''
    i = integer
    while i >= 1:
        binary = str(int(i%2)) + binary
        i = i/2
    if decimals:
        binary = '0'*(decimals-len(binary)) + binary
    return binary if binary else '0'


def string_to_binary_df(df, column='Binary'):
    """
    Function to convert dataframe strings into binary using the first hash function

    :param df: Dataframe in which strings (User_IDs) are contained
    :param column: Column in which strings are stored
    :return: Same df replacing strings with binary values. This will be the dataframe over which the hyperloglog will
    be computed
    """

    df[column] = df[column].apply(lambda x: string_to_binary_hash(x))

    return df


def split_binary(list_, m=64, max_zeros_per_bucket={}):
    """
    This function will take a df with binary values (binary values are the hashed user ids)
    and splits them into a set of buckets (m). For each of these buckets the function will also compute
    the maximum number of zero's at the end of each binary number. The function will then return a
    list which will then be used to estimate the distinct count value

    :param list_: List with a column of binary values
    :param m: Number of buckets (log2(m) will be the length of the binary to classify binary values into each group
    accordingly)
    :param max_zeros_per_bucket: Dictionary with m groups and their max number of zero's at the end of the binary.
    :return: Update the max_zeros_per_bucket dictionary with new max values
    """
    len_buckets = int(np.log2(m))

    for user_id in list_:
        group = user_id[:len_buckets]
        num_final_zeros = len(user_id) - len(user_id.rstrip('0')) + 1
        max_zeros_per_bucket[group] = max(max_zeros_per_bucket[group], num_final_zeros)

    return max_zeros_per_bucket


def get_bucket_groups(m=64):
    """
    Function that will give us the initial binary values to bucket our hashed binary values
    """
    buckets = {}
    len_binary = int(np.log2(m))
    for i in range(m):
        bucket_binary = decimal_to_binary(i, decimals=len_binary)
        buckets[bucket_binary] = 0
    return buckets


def compute_alpha(m):
    """
    Function to compute the alpha value used correct a systematic multiplicative bias
    Instead of computing the integral we have taken the constant values from wikipedia
    """

    if m == 16:
        alpha = 0.673
    elif m == 32:
        alpha = 0.697
    elif m == 64:
        alpha = 0.709
    else:
        alpha = 0.7213 / (1 + (1.079 / m))

    return alpha


def hyperloglog_estimate(max_zeros):
    """
    Returns the estimated hyperloglog estimate and its estimated error

    :param max_zeros: A list with all the max values over which the harmonic mean will be applied
    """

    max_zeros = np.array(max_zeros)
    m = len(max_zeros)

    Z = float(2) ** (-max_zeros)
    Z = 1 / (np.sum(Z))

    estimate = compute_alpha(m) * m ** 2 * Z

    error = 1.04 / np.sqrt(m)

    return estimate, error


def hyperloglog(path_df='data/binary.txt', num_substreams=4096, chunksize=1000000):
    """
    This function takes as input a txt file and estimates the length of unique values using the hyperloglog using a given
    number of sub-streams (number of sub-streams over which the harmonic mean will be computed).

    It is also necesary to provide the used hashing table (maps hexadecimal values into binary).

    Assumptions:
    - The unique values are in hexadecimal form
    - Unique values are randomly and uniformly distributed

    :param path_df: Path with the txt file that includes all the user id values
    :param num_substreams: Number of buckets used. This will allow us to make the estimate more or less accurate
    (increasing this value would also increase the memory required)
    :param chunksize: Chunksize over which analysis is performed
    :return: Cardinality and Error estimate
    """

    max_zeros_per_bucket = get_bucket_groups(num_substreams)

    for users_df_binary in pd.read_csv(path_df, sep=" ", header=0, chunksize=chunksize, index_col=0):

        max_zeros_per_bucket = split_binary(users_df_binary['Binary'].tolist(), m=num_substreams,
                                            max_zeros_per_bucket=max_zeros_per_bucket)

    return hyperloglog_estimate(list(max_zeros_per_bucket.values()))
