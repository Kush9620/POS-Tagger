import sys
import codecs
import os
import random
import math
import helper  # Custom module used for HMM tagging after clustering

# Reads the input file and returns all tokens (words) as a flat list
def get_tokens(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as file_pointer:
        file_contents = file_pointer.readlines()

    tokens = []
    for line in file_contents:
        line = line.strip().split(' ')
        tokens.extend([word for word in line if word != ""])  # Filter out empty strings
    return tokens

# Extracts all unique words from the input file
def get_unique_words(file_path):
    with codecs.open(file_path, 'r', encoding='utf-8') as file_pointer:
        file_contents = file_pointer.readlines()

    word_types = []
    for line in file_contents:
        line = line.strip().split(' ')
        for word in line:
            if word != "" and word not in word_types:
                word_types.append(word)
    return word_types

# Identifies the top-N most frequent words in the corpus (after skipping most frequent)
def get_frequent_words(file_path, n):
    chop_off_distance = 25  # Skip top 25 common words (like stopwords)
    with codecs.open(file_path, 'r', encoding='utf-8') as file_pointer:
        file_contents = file_pointer.readlines()

    word_list = {}

    for line in file_contents:
        line = line.strip().split(' ')
        for word in line:
            if word != "":
                word_list[word] = word_list.get(word, 0) + 1

    sorted_words = sorted(word_list.items(), key=lambda item: item[1], reverse=True)
    top_n_words = [sorted_words[i + chop_off_distance][0] for i in range(n)]
    return top_n_words

# Generates feature vectors for each word using its surrounding context
def get_feature_vectors(tokens, unique_words, feature_words):
    feature_vectors = [[0 for _ in range(2 * len(feature_words))] for _ in range(len(unique_words))]

    for i in range(len(tokens)):
        if i < len(tokens) - 4:
            for j in range(1, 5):
                if tokens[i + j] in feature_words:
                    feature_vectors[unique_words.index(tokens[i])][feature_words.index(tokens[i + j])] += 1

        if i > 3:
            for j in range(1, 5):
                if tokens[i - j] in feature_words:
                    feature_vectors[unique_words.index(tokens[i])][len(feature_words) + feature_words.index(tokens[i - j])] += 1

    return feature_vectors

# Assigns each word to the closest cluster (based on centroid distance)
def map_clusters_with_data(centroids, feature_vectors):
    cluster_data_map = [[] for _ in centroids]
    error = 0

    for i, data_point in enumerate(feature_vectors):
        closest_cluster = min(range(len(centroids)), key=lambda j: dist(data_point, centroids[j]))
        cluster_data_map[closest_cluster].append(i)
        error += dist(data_point, centroids[closest_cluster])

    return cluster_data_map, error

# Calculates Euclidean distance between a data point and centroid
def dist(data_point, centroid):
    return math.sqrt(sum((data_point[x] - centroid[x]) ** 2 for x in range(len(data_point))))

# Recalculates centroids as the mean of all points assigned to each cluster
def recompute_centroids(feature_vectors, cluster_data_map):
    return [mean_of_data_points(feature_vectors, cluster) for cluster in cluster_data_map]

# Computes the mean vector for a cluster
def mean_of_data_points(feature_vectors, list_of_points):
    if not list_of_points:
        return [0] * len(feature_vectors[0])

    mean = [0] * len(feature_vectors[0])
    for index in list_of_points:
        for i, val in enumerate(feature_vectors[index]):
            mean[i] += val

    return [val / len(list_of_points) for val in mean]

# Main execution flow
def main():
    print("Processing... It may take a few second(s)\n")

    textfiles = ["./data/hindi.txt"]
    languages = ["hindi"]
    file_path = textfiles[int(sys.argv[1])]

    number_of_top_words = 100
    tokens = get_tokens(file_path)
    word_types = get_unique_words(file_path)
    top_n_words = get_frequent_words(file_path, number_of_top_words)
    feature_vectors = get_feature_vectors(tokens, word_types, top_n_words)

    # Define POS tag categories
    clusters = ['NN', 'NST', 'NNP', 'PRP', 'DEM', 'VM', 'VAUX', 'JJ', 'RB', 'PSP', 'RP', 'CC', 'WQ', 'QF', 'QC', 'QO',
                'CL', 'INTF', 'INJ', 'NEG', 'UT', 'SYM', 'COMP', 'RDP', 'ECH', 'UNK']
    number_of_clusters = len(clusters)

    # Select initial centroids by index (word samples chosen manually)
    selected_word_index = [1, 94, 109, 44, 77, 131, 156, 406, 244, 444, 14, 29, 295, 806, 157, 362, 855, 781, 100, 494,
                           2017, 222, 199, 40, 400, 689]
    centroids = [feature_vectors[i] for i in selected_word_index]

    # Run K-means clustering for 10 iterations
    for loop in range(10):
        cluster_data_map, total_error = map_clusters_with_data(centroids, feature_vectors)
        print("Iteration", loop + 1, "Error:", total_error)
        centroids = recompute_centroids(feature_vectors, cluster_data_map)

    # Write cluster assignments to output file
    output_path = f"./output/{languages[int(sys.argv[1])]}_clusters.txt"
    with codecs.open(output_path, 'w', 'utf-8') as file_output:
        for cluster in cluster_data_map:
            file_output.write("BEGIN CLUSTER\n")
            for index in cluster:
                file_output.write(word_types[index] + "\n")
            file_output.write("END CLUSTER\n\n")
    print(f"Kindly check {output_path} for clusters and words in that cluster\n")

    # Generate pseudo-tagged corpus for HMM training
    training_file_path = f"./data/{languages[int(sys.argv[1])]}_training_unsupervised.txt"
    with codecs.open(training_file_path, 'w', 'utf-8') as file_output, \
         codecs.open(file_path, 'r', encoding='utf-8') as file_pointer:

        for line in file_pointer:
            words = line.strip().split(' ')
            file_output.write("<s> START\n")
            for word in words:
                for i, cluster in enumerate(cluster_data_map):
                    if any(word_types[j] == word for j in cluster):
                        file_output.write(f"{word} C{i}\n")
                        break
            file_output.write("</s> END\n")

    # HMM training and POS tagging using helper.py
    helper.main(int(sys.argv[1]), sys.argv[2])

# CLI Entry point
if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            main()
        else:
            print("Usage: python unsupervised.py <language> <test_file_path>")
            print("Example: python unsupervised.py 0 ./data/hindi_testing.txt")
            print("More Info: Check ./Readme - Unsupervised.txt for detailed information")

    except ImportError as error:
        print(f"Couldn't find the module - {error.name}, kindly install before proceeding.")
