import codecs
import os
import sys

# Define tag labels for unsupervised clustering (26 cluster-based pseudo-tags)
tags = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 
        'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 
        'C20', 'C21', 'C22', 'C23', 'C24', 'C25']

# Utility function used in Viterbi decoding to compute max probability path
def max_connect(x, y, viterbi_matrix, emission, transmission_matrix):
    max_val = -99999
    path = -1
    for k in range(len(tags)):
        val = viterbi_matrix[k][x - 1] * transmission_matrix[k][y]
        if val * emission > max_val:
            max_val = val * emission
            path = k
    return max_val, path

def main(language, test_file_path):
    exclude = ["<s>", "</s>", "START", "END"]
    filepath = ["./data/hindi_training_unsupervised.txt"]
    languages = ["hindi"]

    # Load the pseudo-tagged training data
    with codecs.open(filepath[language], 'r', encoding='utf-8') as f:
        file_contents = f.readlines()

    wordtypes = []  # Unique words from the training data
    tagscount = [0] * len(tags)  # Count of each tag

    # Count tags and collect word vocabulary
    for line in file_contents:
        line = line.strip().split(' ')
        for i, word in enumerate(line):
            if i == 0 and word not in exclude and word not in wordtypes:
                wordtypes.append(word)
            elif word in tags and word not in exclude:
                tagscount[tags.index(word)] += 1

    # Initialize HMM matrices
    emission_matrix = [[0 for _ in range(len(wordtypes))] for _ in range(len(tags))]
    transmission_matrix = [[0 for _ in range(len(tags))] for _ in range(len(tags))]

    # Re-read to fill emission and transition matrices
    with codecs.open(filepath[language], 'r', encoding='utf-8') as f:
        file_contents = f.readlines()

    row_id = -1
    for line in file_contents:
        line = line.strip().split(' ')
        if line[0] not in exclude and len(line) >= 2:
            col_id = wordtypes.index(line[0])
            prev_row_id = row_id
            row_id = tags.index(line[1])
            emission_matrix[row_id][col_id] += 1
            if prev_row_id != -1:
                transmission_matrix[prev_row_id][row_id] += 1
        else:
            row_id = -1

    # Normalize emission matrix
    for x in range(len(tags)):
        for y in range(len(wordtypes)):
            if tagscount[x] != 0:
                emission_matrix[x][y] /= tagscount[x]

    # Normalize transmission matrix
    for x in range(len(tags)):
        for y in range(len(tags)):
            if tagscount[x] != 0:
                transmission_matrix[x][y] /= tagscount[x]

    # Load test file to be tagged
    with codecs.open(test_file_path, 'r', encoding='utf-8') as file_test:
        test_input = file_test.readlines()

    # Output file to store the POS tagged result
    output_path = f"./output/{languages[language]}_tags_unsupervised.txt"
    with codecs.open(output_path, 'w', 'utf-8') as f:
        pass  # Clear previous output

    # Process each test sentence
    for line in test_input:
        test_words = []
        pos_tags = []

        line = line.strip().split(' ')
        for word in line:
            test_words.append(word)
            pos_tags.append(-1)

        # Initialize Viterbi matrices
        viterbi_matrix = [[0 for _ in range(len(test_words))] for _ in range(len(tags))]
        viterbi_path = [[0 for _ in range(len(test_words))] for _ in range(len(tags))]

        # Fill Viterbi matrix
        for x in range(len(test_words)):
            for y in range(len(tags)):
                if test_words[x] in wordtypes:
                    word_index = wordtypes.index(test_words[x])
                    emission = emission_matrix[y][word_index]
                else:
                    emission = 0.001  # Smoothing for unknown words

                if x > 0:
                    max_val, viterbi_path[y][x] = max_connect(x, y, viterbi_matrix, emission, transmission_matrix)
                else:
                    max_val = 1  # Initial probability
                viterbi_matrix[y][x] = emission * max_val

        # Backtrack to find optimal tag sequence
        maxval = -999999
        maxs = -1
        for x in range(len(tags)):
            if viterbi_matrix[x][len(test_words) - 1] > maxval:
                maxval = viterbi_matrix[x][len(test_words) - 1]
                maxs = x

        for x in range(len(test_words) - 1, -1, -1):
            pos_tags[x] = maxs
            maxs = viterbi_path[maxs][x]

        # Write tagged output to file
        with codecs.open(output_path, 'a', 'utf-8') as file_output:
            for i, x in enumerate(pos_tags):
                file_output.write(f"{test_words[i]}_{tags[x]} ")
            file_output.write(" ._.\n")

    print(f"Kindly check {output_path} file for POS tags.")

# Entry point: expects language index and test file path as arguments
if __name__ == "__main__":
    try:
        if len(sys.argv) == 3:
            lang = int(sys.argv[1])
            test_file = sys.argv[2]
            main(lang, test_file)
        else:
            print("Usage: python unsupervised.py <language_id> <test_file_path>")
            print("Example: python unsupervised.py 0 ./data/hindi_testing_unsupervised.txt")
    except ImportError as error:
        print(f"Missing module: {str(error)}")
