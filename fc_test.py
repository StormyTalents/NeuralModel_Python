import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors

###############################################################################
###############################################################################
###############################################################################
from model import *

# Model
model = FeedForwardModel(2, 4, [FullyConnectedComponent(8),
                                ActivationComponent(tf.nn.sigmoid),
                                #DropoutComponent(0.99),
                                FullyConnectedComponent()])
x, out = model.build()

print
print model # Displays configured model components
print

learning_rate = 0.1

### Other variables used in training/evaluation
y = tf.placeholder(tf.int64, [None])
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(out, y, 1), tf.float32))


###############################################################################
###############################################################################
###############################################################################
# Training procedure/display setup below (not important for this demo)

###############################################################################
### Settings

# Training settings
# Note: Training terminates when the sustained loss is below loss_threshold, or when training has reached max_epochs
max_epochs = 100000
loss_threshold = 1e-12
decay_rate = 0.30 # Exponential decay used to calculate sustained loss
use_GPU = False # Use CUDA acceleration

# Display settings
show_progress = True
display_step = 500
delay = 0.001
interpolation = None # None to use default (eg. "nearest", "bilinear")
resolution = 10
margin = 0.5
boundary_blur_size = 0.5
    
###############################################################################
### Training data

point_label_map = {
    (1, 9): 0,
    (3, 1): 0,
    (3, 2): 0,
    (3, 3): 0,
    (3, 4): 0,
    (3, 5): 0,
    (4, 5): 0,
    (5, 5): 0,
    (6, 5): 0,
    (7, 5): 0,
    
    (1, 1): 1,
    (1, 2): 1,
    (1, 3): 1,
    (1, 4): 1,
    (1, 5): 1,
    (1, 6): 1,
    (1, 7): 1,
    (2, 7): 1,
    (3, 7): 1,
    (4, 7): 1,
    (5, 7): 1,
    (6, 7): 1,
    (7, 3): 1,
    (7, 7): 1,
        
    (7, 1): 2,
}

#point_label_map = {
#    (3, 3): 0,
#    (2, 4): 0,
#    (2, 5): 0,
#    (2, 6): 0,
#    (3, 7): 0,
#    (4, 8): 0,
#    (5, 8): 0,
#    (6, 8): 0,
#    (7, 7): 0,
#    (8, 6): 0,
#    (8, 5): 0,
#    (8, 4): 0,
#    (7, 3): 0,
#    (6, 2): 0,
#    (5, 2): 0,
#    (4, 2): 0,
#    #(5, -1): 0,
#    #(5, 11): 0,
#    #(-1, 5): 0,
#    #(11, 5): 0,
#    
#    (1, 3): 1,
#    (1, 4): 1,
#    (1, 5): 1,
#    (1, 6): 1,
#    (1, 7): 1,
#    (1, 8): 1,
#    (1, 9): 1,
#    (2, 9): 1,
#    (3, 9): 1,
#    (4, 9): 1,
#    (5, 9): 1,
#    (6, 9): 1,
#    (7, 9): 1,
#    (8, 9): 1,
#    (9, 9): 1,
#    (9, 8): 1,
#    (9, 7): 1,
#    (9, 6): 1,
#    (9, 5): 1,
#    (9, 4): 1,
#    (9, 3): 1,
#    (9, 2): 1,
#    (9, 1): 1,
#    (8, 1): 1,
#    (7, 1): 1,
#    (6, 1): 1,
#    (5, 1): 1,
#    (4, 1): 1,
#    (3, 1): 1,
#    (2, 1): 1,
#    (1, 1): 1,
#    (1, 2): 1,
#    (2, 2): 1,
#    (8, 2): 1,
#    (2, 8): 1,
#    (8, 8): 1,
#    
#    (-1, -1): 2,
#    (-1, 0): 2,
#    (-1, 1): 2,
#    (-1, 2): 2,
#    (-1, 3): 2,
#    (-1, 4): 2,
#    (-1, 5): 2,
#    (-1, 6): 2,
#    (-1, 7): 2,
#    (-1, 8): 2,
#    (-1, 9): 2,
#    (-1, 10): 2,
#    (-1, 11): 2,
#    (0, 11): 2,
#    (1, 11): 2,
#    (2, 11): 2,
#    (3, 11): 2,
#    (4, 11): 2,
#    (5, 11): 2,
#    (6, 11): 2,
#    (7, 11): 2,
#    (8, 11): 2,
#    (9, 11): 2,
#    (10, 11): 2,
#    (11, 11): 2,
#    (11, 10): 2,
#    (11, 9): 2,
#    (11, 8): 2,
#    (11, 7): 2,
#    (11, 6): 2,
#    (11, 5): 2,
#    (11, 4): 2,
#    (11, 3): 2,
#    (11, 2): 2,
#    (11, 1): 2,
#    (11, 0): 2,
#    (11, -1): 2,
#    (10, -1): 2,
#    (9, -1): 2,
#    (8, -1): 2,
#    (7, -1): 2,
#    (6, -1): 2,
#    (5, -1): 2,
#    (4, -1): 2,
#    (3, -1): 2,
#    (2, -1): 2,
#    (1, -1): 2,
#    (0, -1): 2,
#    
#    (5, 4): 3,
#    (4, 5): 3,
#    (5, 5): 3,
#    (6, 5): 3,
#    (5, 6): 3,
#}


###############################################################################
### Display setup

x_values = [point[0] for point in point_label_map]
y_values = [point[1] for point in point_label_map]
x_min = min(x_values)
x_max = max(x_values)
y_min = min(y_values)
y_max = max(y_values)
x_range = x_max - x_min
y_range = y_max - y_min
x_left = x_min - int(x_range * margin)
x_right = x_max + int(x_range * margin)
y_bottom = y_min - int(y_range * margin)
y_top = y_max + int(y_range * margin)

def transform_x(x):
    return (x - x_left) * resolution
def transform_y(y):
    return (y - y_bottom) * resolution
def untransform_x(x):
    return float(x) / resolution + x_left
def untransform_y(y):
    return float(y) / resolution + y_bottom

x_limits = [transform_x(x_left), transform_x(x_right)]
y_limits = [transform_y(y_bottom), transform_y(y_top)]
hm_width = int(x_right - x_left + 1) * resolution
hm_height = int(y_top - y_bottom + 1) * resolution

fig = plt.figure()
test_points = [[untransform_x(i), untransform_y(j)] for j in range(hm_height) for i in range(hm_width)]

map_colors = ['#990000', '#004C99', '#009900', '#990099']
marker_colors = list(map_colors)
cmap = colors.ListedColormap(map_colors)
bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
norm = colors.BoundaryNorm(bounds, cmap.N)
ticks = [0, 1, 2, 3]

heatmap = np.zeros((hm_height, hm_width))
def formatter_x(x, p):
    return "{}".format(int(x / resolution + x_left)) if x / resolution % 1 == 0 else ""
def formatter_y(y, p):
    return "{}".format(int(y / resolution + y_bottom)) if y / resolution % 1 == 0 else ""
def format_display():
    fig.canvas.set_window_title('Epoch {}'.format(epoch))
    fig.clear()
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    axes = plt.gca()
    axes.get_xaxis().set_major_formatter(ticker.FuncFormatter(formatter_x))
    axes.get_yaxis().set_major_formatter(ticker.FuncFormatter(formatter_y))
    
def display(session, loss_values, points, labels, out_val, done):
    loss_values = loss_values[-2000:]
    # Format display
    format_display()
    
    # Draw heatmap
    global heatmap
    test_result = session.run([out], feed_dict={x: test_points})
    #test_result[0] /= boundary_blur_size * (np.max(test_result[0]) - np.min(test_result[0]))
    
    #plt.subplot(1, 2, 1)
    for i in range(hm_height):
        for j in range(hm_width):
            n = i * hm_width + j
            heatmap[i][j] = np.argmax(test_result[0][n])
    hm = plt.imshow(heatmap, interpolation=interpolation, origin='lower', cmap=cmap, norm=norm)
    plt.colorbar(hm, ticks=ticks)

    # Draw points
    for i in range(len(points)):
        correct = np.argmax(out_val[i]) == labels[i]
        color = marker_colors[labels[i]]
        plt.scatter(transform_x(points[i][0]), transform_y(points[i][1]), color=color, s=60, edgecolors=('black' if correct else 'white'), linewidth=(1 if correct else 2))
        
    # Update text
    plt.title("Loss: {0:.2E}".format(loss_values[-1]))
    
#    plt.subplot(1, 2, 2)
#    plt.plot(list(range(len(loss_values))), loss_values)

    # Delay
    plt.pause(delay)
    
###############################################################################
### Training

# Initialize environment
initialize = tf.global_variables_initializer()

# Session config
config = tf.ConfigProto(device_count = {'GPU': 1 if use_GPU == True else 0})

# Run model
with tf.Session(config=config) as session:
    session.run(initialize)
    
    # Get training data
    points = point_label_map.keys()
    labels = point_label_map.values()
        
    done = False
    epoch = 0
    sustained_loss = 0.0
    loss_values = []
    while not done:

        # Trains on the data
        _, loss_val, accuracy_val, out_val = session.run([train_op, loss, accuracy, out], feed_dict={x: points, y: labels})
        sustained_loss = decay_rate * sustained_loss + (1.0 - decay_rate) * loss_val
        loss_values.append(loss_val)
            
        epoch += 1
        #print "Epoch {}".format(epoch)
        #print "  Loss: {}".format(loss_val)
        #print "  Accuracy: {}".format(accuracy_val)
                
        # Termination condition
        if epoch >= max_epochs or sustained_loss < loss_threshold:
            done = True
        
        # Show/update display
        if epoch % display_step == 0 and show_progress or done:
            display(session, loss_values, points, labels, out_val, done)

# Display results
print("Epoch count: {}".format(epoch))    
plt.show()

#plt.figure('Loss')
#plt.plot(loss_values)
#plt.show()
