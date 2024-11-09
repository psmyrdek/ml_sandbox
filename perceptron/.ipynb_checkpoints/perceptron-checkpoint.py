# The perceptron learning algorithm is an iterative process that adjusts the weights and threshold of the perceptron based on how close it’s getting to the training data.

# Here’s a high-level overview of the perceptron learning algorithm:

# 1. Initialize the weights and threshold with random values.
# 2. For each input-output pair in the training data:
# 3. Compute the perceptron’s output using the current weights and threshold.
# 4. Update the weights and threshold based on the difference between the desired output and the perceptron’s output – the error.

# Repeat steps 2 and 3 until the perceptron classifies all input-output pairs correctly, or a specified number of iterations have been completed.
# The update rule for the weights and threshold is simple:

# If the perceptron’s output is correct, do not change the weights or threshold.
# If the perceptron’s output is too low, increase the weights and decrease the threshold.
# If the perceptron’s output is too high, decrease the weights and increase the threshold.

# 0.12601332724913494
# 0.9819518594076044
# 0.8020277001351964
# 1.6579662322936657


from data import and_data
import random

learning_rate = 0.5
weights = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
threshold = [random.uniform(-0.5, 0.5)]


def calc_dot_product(input_vector, weights, threshold):
    return (input_vector[0] * weights[0] + input_vector[1] * weights[1]) + threshold[0]


def activate(dot_product, threshold):
    return 1 if dot_product >= threshold[0] else 0


def train_perceptron(training_data, weights, threshold, learning_rate):
    cur_epoch = 0
    epoch_limit = 100
    trained = False

    while cur_epoch < epoch_limit and not trained:
        has_error = False
        for input_vector, target in training_data:
            # Calculate dot product and error
            dot_product = calc_dot_product(input_vector, weights, threshold)
            error = target - activate(dot_product, threshold)

            # Check for errors and adjust weights and threshold if needed
            if error != 0:
                has_error = True
                weights[0] += input_vector[0] * learning_rate * error
                weights[1] += input_vector[1] * learning_rate * error
                threshold[0] += learning_rate * error  # Adjust threshold as bias term

        # If no errors during this epoch, mark as trained
        if not has_error:
            trained = True
        cur_epoch += 1
        print(f"Epoch {cur_epoch}: Weights = {weights}, Threshold = {threshold[0]}")

    print("Training completed in epochs:", cur_epoch)


# Train the perceptron with the AND data
train_perceptron(and_data, weights, threshold, learning_rate)
