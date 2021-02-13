'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import numpy as np
import matplotlib
import matplotlib.cm as cmap
import time
import os.path
import os
import scipy 
from struct import unpack
from brian2 import *
# from brian2tools import * 


from functions.data import get_labeled_data

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------     

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in range(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]

def get_new_assignments(result_monitor, input_numbers):
    print(result_monitor.shape)
    assignments = np.ones(n_e) * -1 # initialize them as not assigned
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in range(10):
        num_inputs = len(np.where(input_nums == j)[0])
        if num_inputs > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_inputs
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j 
    return assignments

MNIST_data_path = './mnist/'
data_path = './activity/'
training_ending = '10000'
testing_ending = '10000'
start_time_training = 0
end_time_training = int(training_ending)
start_time_testing = 0
end_time_testing = int(testing_ending)

n_e = 400
n_input = 784
ending = ''

print('load MNIST')
training = get_labeled_data(MNIST_data_path + 'training')
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)

print('load results')
training_result_monitor = np.load(data_path + 'resultPopVecs' + training_ending + ending + '.npy')
training_input_numbers = np.load(data_path + 'inputNumbers' + training_ending + '.npy')
testing_result_monitor = np.load(data_path + 'resultPopVecs' + testing_ending + '.npy')
testing_input_numbers = np.load(data_path + 'inputNumbers' + testing_ending + '.npy')
print(training_result_monitor.shape)

print('get assignments')
test_results = np.zeros((10, end_time_testing-start_time_testing))
test_results_max = np.zeros((10, end_time_testing-start_time_testing))
test_results_top = np.zeros((10, end_time_testing-start_time_testing))
test_results_fixed = np.zeros((10, end_time_testing-start_time_testing))
assignments = get_new_assignments(training_result_monitor[start_time_training:end_time_training], 
                                  training_input_numbers[start_time_training:end_time_training])
print(assignments)
counter = 0 
num_tests = end_time_testing // 10000
sum_accurracy = [0] * num_tests
while (counter < num_tests):
    end_time = min(end_time_testing, 10000*(counter+1))
    start_time = 10000*counter
    test_results = np.zeros((10, end_time-start_time))
    print('calculate accuracy for sum')
    for i in range(end_time - start_time):
        test_results[:,i] = get_recognized_number_ranking(assignments, 
                                                          testing_result_monitor[i+start_time,:])
    difference = test_results[0,:] - testing_input_numbers[start_time:end_time]
    correct = len(np.where(difference == 0)[0])
    incorrect = np.where(difference != 0)[0]
    sum_accurracy[counter] = correct/float(end_time-start_time) * 100
    print('Sum response - accuracy: ', sum_accurracy[counter], ' num incorrect: ', len(incorrect))
    counter += 1
print('Sum response - accuracy --> mean: ', np.mean(sum_accurracy),  '--> standard deviation: ', np.std(sum_accurracy))


show()
