Experiment environment:
    Python 3.6.1
    NumPy 1.13.1 
    matplotlib 2.0.2
    scikit-learn 0.18.1

Instructions:
    I used the 1000 data for this assignment.
    By the command below, all steps can be done.
        $ python perceptron.py 1000.dat

    First, five figures for five different learning rates will be shown one by one. By closing the graph, it will try the next learning rate.
    It may take a while because this step calculates 1000 epochs for each learning rate.

    Then, it will start another calculation with a termination condition. If validation error hikes 5% from the previous epoch, it will stop the training.

    Finally, it will train the perceptron again with the validation set and the training set as one training set.
    With the optimal learning rate and termination condition, it will calculate and output the generalization error on the console.
    
