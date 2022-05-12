import os
from sys import argv
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc, delta
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier


training_directory = 'project_audio/train'
test_directory = 'project_audio/test'

words_by_id = {1: 'cars', 2: 'cool', 3: 'dive', 4: 'drive', 5: 'eat', 6: 'for', 7: 'icecream', 8: 'mars', 9: 'snacks',
         10: 'snakes', 11: 'to', 12: 'travel', 13: 'treasure'}

word_id = {'cars': 1, 'cool': 2, 'dive': 3, 'drive': 4, 'eat': 5, 'for': 6, 'icecream': 7, 'mars': 8, 'snacks': 9,
         'snakes': 10, 'to': 11, 'travel': 12, 'treasure': 13}

age_by_id = {'2': 'toddler 1', '3': 'kindergarten', '4': 'adult', '6': 'toddler 2'}  # wrt to datafile names

mfccs = {}  # dictionary of mfccs for all training files, formatted { [word]: [[mfcc1], [mfcc2], ...], [word]: etc }


def extract_mfcc(audio_file, wrd, test=True):
    """Extracts mel-frequency coefficients, deltas, and double-deltas from each audio file sample, building an
    np array of mfccs/sample and encapsulating that in a list to
        a. return if the function is being run in test mode (default), or
        b. append to a global dictionary list of mfccs by word if the mode is training.

   Parameters:
       audio_file: string name of the file to be dimensionalized
       wrd: string of word being said in the audio_file
       test: boolean indicating mode of function call (test or train)
    Return value: if run in test mode, return the full_cepstrum data for the audio file"""
    samp_rate, data = read(audio_file)
    # extract 13 cepstral coefficients from audio data at default segmentation params,
    # with first coefficient replaced by a measure of that frame's energy
    # nfft set per warning relative to my max file-size
    mfcc_array = mfcc(data, samp_rate, nfft=2048, appendEnergy=True)
    d_mfcc_array = delta(mfcc_array, 2)  # differential coefficients for the mfcc
    dd_mfcc_array = delta(d_mfcc_array, 2)  # acceleration coefficients for the mfcc
    full_cepstrum = np.hstack((mfcc_array, d_mfcc_array, dd_mfcc_array))  # dimensionalized data for the file combined
    if not test:
        if word not in mfccs:
            mfccs[wrd] = [full_cepstrum]
        else:
            mfccs[wrd].append(full_cepstrum)
    else:
        return full_cepstrum


def build_prediction_model():
    """Compare every mfcc-word-representation-matrix in the mfcc dictionary to every other entry and itself, computing
    the distance between the matrix pairs as a single value via a distance-time-warped shortest path helper function.
    For each word-to-[every other word and itself] comparison make an int entry in an id list indicating which word is
    being compared to all other words and a matching-indexed entry in a list of lists of all of those comparison values.
    Feed those two lists to a 3-nearest neighbors classifier builder.

    Return value: the nearest neighbors classifier."""
    global mfccs
    dist_ids = []  # flat list of known word_ids to assoc. with file-to-file distance calculations in model training
    dist = []  # corresponding list-of-lists of distance calculations to be used in model training. Each entry is a list
    # of distances from [the corresponding dist_id word/file reference] to every file (itself included) in the training
    # folder
    print('calculating the distance between all training word-representations as inputs for k-neighbors classifier\n'
          'this could take a while (a few minutes)')
    for word_1 in mfccs:
        for cepstrum_matrix_1 in mfccs[word_1]:
            dist_comp = []
            for word_2 in mfccs:
                for cepstrum_matrix_2 in mfccs[word_2]:
                    # uncomment if you want to track the progress of the DTW initial model calculations
                    # print('comparing {} and {}'.format(word_1, word_2))
                    dist_comp.extend([dtw(cepstrum_matrix_1, cepstrum_matrix_2)])
            dist.append(dist_comp)
            dist_ids.append(word_id[word_1])
    print('initializing the classifier model')
    classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    print('fitting the classifier model')
    classifier.fit(dist, dist_ids)
    return classifier


def dtw(c1, c2):
    """Helper function implementation of dynamic-time-warping algorithm. Calculates the euclidian distance between
    cepstral matrices c1 and c2, storing them in a cost_matrix of size c1 frame length X c2 frame length, then in place
    traverses that matrix to determine the shortest cost path of matching-up (aka time-warping) the two audio-files.

   Parameters:
       c1: cepstral matrix for one word
       c2: cepstral matrix a comparison word
   Return value: the floating-point cost of the shortest-path method of matching the two audio file matrices"""
    cost_matrix = cdist(c1, c2, metric='euclidean')
    m, n = np.shape(cost_matrix)
    # initialize the shortest distance path costs for the first row and first column of the cost_matrix
    for i in range(m):
        cost_matrix[i, 0] = cost_matrix[i, 0] + cost_matrix[i - 1, 0]
    for j in range(n):
        cost_matrix[0, j] = cost_matrix[0, j] + cost_matrix[0, j - 1]
    for i in range(1, m):
        for j in range(1, n):
            cost_matrix[i, j] = cost_matrix[i, j] + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1],
                                                        cost_matrix[i - 1, j - 1])
    return cost_matrix[m - 1, n - 1]


def predict_word_distances(unk_mfcc, mode):
    """Takes in mfcc-array of an unknown (test) word and returns either a distance list of the dtw values calculated
    between it and all of the words in the training set to then be fed to the KNN classifier or a single int
    representing the word ID of the word whose DTW result was lowest relative to that of the unknown word.

   Parameters:
       unk_mfcc: cepstral matrix for word to be predicted
       mode: string value of the prediction mode the program is being run in (KNN or DTW)
   Return value: either an array of float distance values (KNN mode) or a single integer ID value (DTW mode)"""
    distances = [] if mode == 'knn' else {}
    for word in mfccs:
        if mode == 'dtw':
            distances[word] = []
        for cepstrum in mfccs[word]:
            shortest_distance = dtw(unk_mfcc, cepstrum)
            if mode == 'knn':
                distances.append(shortest_distance)
            else:
                distances[word].append(shortest_distance)
                distances[word].sort()
    if mode == 'knn':
        return distances
    else:
        minIndex = 1
        for k in range(2, len(distances) + 1):
            if distances[words_by_id[k]] < distances[words_by_id[minIndex]]:
                minIndex = k
        return minIndex


# to run: python [this filename] [prediction method]
# prediction method must be 'knn' or 'dtw'
if __name__ == '__main__':
    prediction_method = argv[1]
    if prediction_method != 'knn' and prediction_method != 'dtw':
        print('Please include valid prediction method argument (knn or dtw) in your start command')
        exit()
    # document success rate of the predictor
    results = {'toddler 1': [0, 0], 'toddler 2': [0, 0], 'kindergarten': [0, 0], 'adult': [0, 0]}
    for filename in os.listdir(training_directory):  # read in files from training directory
        if filename.endswith('.wav'):
            fname = filename.split('_')
            word = fname[0]
            extract_mfcc(training_directory + '/' + filename, word, False)
    if prediction_method == 'knn':
        classifier_model = build_prediction_model()
    for filename in os.listdir(test_directory):
        if filename.endswith('.wav'):
            fname = filename.split('_')
            word = fname[0]
            age = age_by_id[fname[1]]
            train_mfcc = extract_mfcc(test_directory + '/' + filename, word)
            prediction = predict_word_distances(train_mfcc, prediction_method)
            if prediction_method == 'knn':
                prediction = classifier_model.predict([prediction])[0]
            results[age][1] += 1
            if word == words_by_id[prediction]:
                results[age][0] += 1
            print('{} -- expected word: {}, predicted word: {}'.format(age, word, words_by_id[prediction]))
    print('toddler 1 percentage correct predicted: {}/{} = {}'.format(results['toddler 1'][0], results['toddler 1'][1],
                                                                      100 * results['toddler 1'][0] /
                                                                      results['toddler 1'][1]))
    print('toddler 2 percentage correct predicted: {}/{} = {}'.format(results['toddler 2'][0], results['toddler 2'][1],
                                                                      100 * results['toddler 2'][0] /
                                                                      results['toddler 2'][1]))
    print('kindergarten percentage correct predicted: {}/{} = {}'.format(results['kindergarten'][0],
                                                                         results['kindergarten'][1],
                                                                         100 * results['kindergarten'][0] /
                                                                         results['kindergarten'][1]))
    print('adult percentage correct predicted: {}/{} = {}'.format(results['adult'][0], results['adult'][1],
                                                                  100 * results['adult'][0] / results['adult'][1]))
