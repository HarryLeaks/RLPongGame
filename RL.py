import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2 #read in pixel data
import pong #our class
import numpy as np #math
import random #random 
from collections import deque #queue data structure. fast appends. and pops. replay memory
from tensorflow.compat.v1.summary import FileWriter


#hyper params
ACTIONS = 3 #up,down, stay
#define our learning rate
GAMMA = 0.99
#for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
#how many frames to anneal epsilon
EXPLORE = 500000 
OBSERVE = 50000
#store our experiences, the size of it
REPLAY_MEMORY = 500000
#batch size to train on
BATCH = 100

def createGraph():
    # First convolutional layer. Bias vector.
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    b_conv1 = tf.Variable(tf.zeros([32]))

    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    b_conv2 = tf.Variable(tf.zeros([64]))

    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    b_conv3 = tf.Variable(tf.zeros([64]))

    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    b_fc4 = tf.Variable(tf.zeros([784]))

    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    b_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # Input for pixel data
    s = tf.keras.Input(shape=(84, 84, 4), dtype=tf.float32)

    # Computes rectified linear unit activation function on a 2-D convolution given 4-D input and filter tensors.
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + b_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

#deep q network. feed in pixel data to graph session 
def trainGraph(inp, out, sess):
# To calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS]) 
    gt = tf.placeholder("float", [None])  # Ground truth

    # Action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices=1)
    # Cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    # Optimization function to minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # Initialize our game
    game = pong.PongGame()
    
    # Create a queue for experience replay to store policies
    D = deque()

    # Initial frame
    frame = game.getPresentFrame()
    # Convert RGB to grayscale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # Binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # Stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # Saver
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    t = 0
    epsilon = INITIAL_EPSILON
    
    # Training time
    while True:
        # Output tensor
        out_t = out.eval(feed_dict={inp: [inp_t]}, session=sess)[0]
        # Argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1
        
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # Reward tensor if score is positive
        reward_t, frame = game.getNextFrame(argmax_t)
        # Get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # New input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)
        
        # Add our input tensor, argmax tensor, reward, and updated input tensor to the stack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # If we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        
        # Training iteration
        if t > OBSERVE:
            # Get values from our replay memory
            minibatch = random.sample(D, BATCH)
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]
        
            gt_batch = []
            out_batch = out.eval(feed_dict={inp: inp_t1_batch}, session=sess)
            
            # Add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # Train on that batch
            train_step.run(feed_dict={
                gt: gt_batch,
                argmax: argmax_batch,
                inp: inp_batch
            }, session=sess)
        
        # Update our input tensor to the next frame
        inp_t = inp_t1
        t += 1

        # Print our progress and save checkpoints
        if t % 10000 == 0:
            saver.save(sess, './' + 'pong' + '-dqn', global_step=t)

        print("TIMESTEP", t, "/ EPSILON", epsilon, "/ ACTION", maxIndex, "/ REWARD", reward_t, "/ Q_MAX %e" % np.max(out_t))


def main():
    #create session
    sess = tf.compat.v1.Session()
    #input layer and output layer by creating graph
    inp, out = createGraph()
    #train our graph on input and output with session variables
    trainGraph(inp, out, sess)

if __name__ == "__main__":
    main()