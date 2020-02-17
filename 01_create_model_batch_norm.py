############################
# Code written by:  Richard Downey
# Last updated:  01/03/2018
# Code purpose:  Create a hybrid NN with Tensorflow
###########################

# Import the required libraries
import tensorflow as tf
import pandas as pd
import shutil
import numpy as np
from functools import partial

data_path = "C:/Users/downey richard/PycharmProjects/Recommender/Output Data/"

# Read in the training and validation sets
order_products_train = pd.read_pickle(data_path + "order_products_train.pkl")
order_products_valid = pd.read_pickle(data_path + "order_products_valid.pkl")

# Get list of dow
unique_order_dow = list(order_products_train['order_dow'].drop_duplicates())

# Determine CSV and label columns
non_factor_columns = 'aisle,department,product_id,add_to_cart_order,reordered,order_number,order_dow,' \
                     'order_hour_of_day,days_since_prior_order'.split(',')
embedd_columns = ["item_embedd_{}".format(i) for i in range(1, 21)] + ["user_embedd_{}".format(i) for i in range(1, 21)]
csv_columns = non_factor_columns + embedd_columns
label_column = 'reordered'

# Set default values for each CSV column
non_embedd_defaults = [['Unknown'], ['Unknown'], [0], [0], [0], [0], [0], [0.0], [0.0]]
embedd_defaults = [[0.0] for i in range(1, 21)] + [[0.0] for i in range(1, 21)]
defaults = non_embedd_defaults + embedd_defaults


# Create input function for train and eval
def read_dataset(filename, mode, batch_size=512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(records=value_column, record_defaults=defaults)
            features = dict(zip(csv_columns, columns))
            label = features.pop(label_column)
            return features, label

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename=filename)

        # Create dataset from file list
        dataset = tf.data.TextLineDataset(filenames=file_list).map(map_func=decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1

        dataset = dataset.repeat(count=num_epochs).batch(batch_size=batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn


# Create feature columns to be used in model
def create_feature_columns(args):

    # Create the department column
    categorical_department_column = tf.feature_column.categorical_column_with_vocabulary_file(
        key="department",
        vocabulary_file=data_path + "/" + "departments.csv",
        vocabulary_size=21,
        num_oov_buckets=1)

    # Convert categorical department column into indicator column
    indicator_department_column = tf.feature_column.indicator_column(categorical_column=categorical_department_column)

    # Create the aisle column
    aisle_column = tf.feature_column.categorical_column_with_hash_bucket(
        key="aisle",
        hash_bucket_size=len(order_products_train['aisle'].drop_duplicates()) + 1)

    # Embed into a lower dimensional representation
    embedded_aisle_column = tf.feature_column.embedding_column(
        categorical_column=aisle_column,
        dimension=args['aisle_embedding_dimensions'])

    # Create add_to_cart_order boundaries list for our binning
    add_to_cart_order_boundaries = list(range(1, 80, 10))

    # Create add_to_cart_order feature column using raw data
    add_to_cart_order_column = tf.feature_column.numeric_column(
        key="add_to_cart_order")

    # Create bucketized add_to_cart_order feature column using our boundaries
    add_to_cart_order_bucketized = tf.feature_column.bucketized_column(
        source_column=add_to_cart_order_column,
        boundaries=add_to_cart_order_boundaries)

    # Create the order_number column
    order_number_column = tf.feature_column.numeric_column(
        key="order_number")

    # Create the dow column as categorical column
    categorical_dow_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key="order_dow",
        vocabulary_list=unique_order_dow,
        num_oov_buckets=1)

    # Convert categorical dow column into indicator column
    indicator_dow_column = tf.feature_column.indicator_column(categorical_column=categorical_dow_column)

    # Create boundaries for order_hour_of_day
    order_hour_of_day_boundaries = [0, 7, 12, 17]

    # Create order_hour_of_day feature column using raw data
    order_hour_of_day_column = tf.feature_column.numeric_column(
        key="order_hour_of_day")

    # Create bucketized order_hour_of_day feature column using our boundaries
    order_hour_of_day_bucketized = tf.feature_column.bucketized_column(
        source_column=order_hour_of_day_column,
        boundaries=order_hour_of_day_boundaries)

    # Create boundaries for days_since_prior_order
    days_since_prior_order_boundaries = list(range(0, 30, 5))

    # Create days_since_prior_order feature column using raw data
    days_since_prior_order_column = tf.feature_column.numeric_column(
        key="days_since_prior_order")

    # Create bucketized days_since_prior_order feature column using our boundaries
    days_since_prior_order_bucketized = tf.feature_column.bucketized_column(
        source_column=days_since_prior_order_column,
        boundaries=days_since_prior_order_boundaries)

    # Create user and item embedding feature columns from the trained WALS model
    user_embedd = [tf.feature_column.numeric_column(key="user_embedd_" + str(i)) for i in range(1, 21)]
    item_embedd = [tf.feature_column.numeric_column(key="item_embedd_" + str(i)) for i in range(1, 21)]

    feature_columns = [indicator_department_column,
                       embedded_aisle_column,
                       add_to_cart_order_bucketized,
                       order_number_column,
                       indicator_dow_column,
                       order_hour_of_day_bucketized,
                       days_since_prior_order_bucketized] + user_embedd + item_embedd

    return feature_columns


# Create the model function
def model_fn(features, labels, mode, params):

    # Set the batch norm params

    batch_norm_momentum = 0.99

    # Create neural network input layer using our feature columns defined above
    net = tf.feature_column.input_layer(features=features,
                                        feature_columns=params['feature_columns'])

    he_init = tf.variance_scaling_initializer()

    if mode == tf.estimator.ModeKeys.TRAIN:
        batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=True,
            momentum=batch_norm_momentum)

    else:
        batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=False,
            momentum=batch_norm_momentum)

    dense_layer = partial(
        tf.layers.dense,
        kernel_initializer=he_init)

    # Create hidden and batch norm layers
    hidden1 = dense_layer(inputs=net, units=params['hidden_units'][0])
    bn1 = tf.nn.relu(batch_norm_layer(hidden1))
    hidden2 = dense_layer(inputs=bn1, units=params['hidden_units'][1])
    bn2 = tf.nn.relu(batch_norm_layer(hidden2))
    hidden3=dense_layer(inputs=bn2, units=params['hidden_units'][2])
    bn3 = tf.nn.relu(batch_norm_layer(hidden3))

    # Compute logits (1 per class) using the output of our last hidden layer
    logits = tf.layers.dense(inputs=bn3, units=1, activation=None)

    # If the mode is prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Create predictions dict
        predictions_dict = {
            'probabilities': tf.nn.sigmoid(logits),
            'logits': logits
        }

        # Create export outputs
        export_outputs = {"predict_export_outputs": tf.estimator.export.PredictOutput(outputs=predictions_dict)}

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=None,
            train_op=None,
            eval_metric_ops=None,
            export_outputs=export_outputs)

    # Continue on with training and evaluation modes

    # Compute loss using sigmoid cross entropy since this is classification and our labels
    # and probabilities are mutually exclusive

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.one_hot(labels, depth=1),
                                           logits=logits)

    predicted_classes = tf.cast(tf.argmax(logits, 1), tf.float32)

    # Compute evaluation metrics of total accuracy and auc
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    auc = tf.metrics.auc(labels=labels,
                         predictions=predicted_classes,
                         name='auc')

    # Put eval metrics into a dictionary
    eval_metrics = {'accuracy': accuracy,
                    'auc': auc}

    # Create scalar summaries to see in TensorBoard
    tf.summary.scalar(name='accuracy', tensor=accuracy[1])
    tf.summary.scalar(name='auc', tensor=auc[1])

    logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                               "accuracy": accuracy[1],
                                               "auc": auc[1]}, every_n_iter=100)

    # If the mode is evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=None,
            loss=loss,
            train_op=None,
            eval_metric_ops=eval_metrics,
            export_outputs=None)

    # Continue on with training mode

    # If the mode is training

    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create a custom optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

    # Create train op

    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=None,
        loss=loss,
        training_hooks=[logging_hook],
        train_op=train_op,
        eval_metric_ops=None,
        export_outputs=None)


# Create serving input function
def serving_input_fn():
    feature_placeholders = {
        colname: tf.placeholder(dtype=tf.float32, shape=[None])
        for colname in embedd_columns
    }

    feature_placeholders['department'] = tf.placeholder(dtype=tf.string, shape=[None])
    feature_placeholders['aisle'] = tf.placeholder(dtype=tf.string, shape=[None])
    feature_placeholders['add_to_cart_order'] = tf.placeholder(dtype=tf.float32, shape=[None])
    feature_placeholders['order_number'] = tf.placeholder(dtype=tf.float32, shape=[None])
    feature_placeholders['order_dow'] = tf.placeholder(dtype=tf.int32, shape=[None])
    feature_placeholders['order_hour_of_day'] = tf.placeholder(dtype=tf.string, shape=[None])
    feature_placeholders['days_since_prior_order'] = tf.placeholder(dtype=tf.string, shape=[None])

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# Create train and evaluate loop to combine all of the pieces together.
tf.logging.set_verbosity(tf.logging.INFO)


def train_and_evaluate(args):
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args['output_dir'],
        params={
            'feature_columns': create_feature_columns(args),
            'hidden_units': args['hidden_units'],
            'learning_rate': args['learning_rate']
        })

    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(filename=args['train_data_paths'],
                              mode=tf.estimator.ModeKeys.TRAIN,
                              batch_size=args['batch_size']),
        max_steps=args['train_steps'])

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(filename=args['eval_data_paths'],
                              mode=tf.estimator.ModeKeys.EVAL,
                              batch_size=args['batch_size']),
        steps=None,
        start_delay_secs=args['start_delay_secs'],
        throttle_secs=args['throttle_secs'],
        exporters=exporter)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# Function for getting predictions from trained model
def get_preds(args):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=args['output_dir'],
                                       params={
                                           'feature_columns': create_feature_columns(args),
                                           'hidden_units': args['hidden_units'],
                                           'learning_rate': args['learning_rate']
                                       })

    pred = list(estimator.predict(input_fn=read_dataset(filename=args['pred_data_paths'],
                                                        mode=tf.estimator.ModeKeys.PREDICT)))

    probabilities = pd.DataFrame([np.asscalar(d["probabilities"]) for d in pred])
    probabilities.columns = ['probabilities']

    logits = pd.DataFrame([np.asscalar(d["logits"]) for d in pred])
    logits.columns = ['logits']

    output = probabilities.merge(logits, left_index=True, right_index=True)

    return output


# Call train and evaluate loop
outdir = 'hybrid_recommendation_trained'
shutil.rmtree(outdir, ignore_errors=True)  # start fresh each time

arguments = {
        'train_data_paths': data_path + "order_products_train.csv",
        'eval_data_paths': data_path + "order_products_valid.csv",
        'pred_data_paths': data_path + "order_products_valid.csv",
        'output_dir': outdir,
        'batch_size': 128,
        'learning_rate': 0.1,
        'hidden_units': [256, 128, 64],
        'aisle_embedding_dimensions': 4,
        'train_steps': 10000,
        'start_delay_secs': 30,
        'throttle_secs': 30
    }

train_and_evaluate(arguments)

# Get predictions
predictions = get_preds(arguments)