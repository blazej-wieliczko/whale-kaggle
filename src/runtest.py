from nolearn import dbn
from nolearn.model import AveragingEstimator
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.linear_model
import sklearn.ensemble
from ripdata import load_data, runpca, load_x, calculateAuc

train_input_path = "../data/full-data/train/fbank/"
test_input_path = "../data/full-data/test/fbank/"
labels_path = '../data/full-data/train.csv'

X, y, filenames = load_data(train_input_path, labels_path)
print 'with pca'
x_kaggle =  load_x(test_input_path)
print 'loaded kaggle evaluation data'
print 'whitened kaggle evaluation data'

X, pca_transformer = runpca(X)
x_kaggle = pca_transformer.transform(x_kaggle)


x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=47)

#x_train, y_train = X,y



def evaluate_validation(classifier, epoch):
    if not epoch % 50:
        print 'Epoch:', epoch, '.Validation:', calculateAuc(classifier, x_test, y_test)
    if not epoch % 50:
        print 'Epoch:', epoch, '. Train:', calculateAuc(classifier, x_train, y_train)
        print classifier.layer_sizes

classifiers = [
    dbn.DBN([-1, 500, 500, -1], momentum=0.9, learn_rates=0.3, learn_rate_minimums=0.01,learn_rate_decays=0.996,
            epochs=400,minibatch_size=100, fine_tune_callback=evaluate_validation, epochs_pretrain=[100, 60, 60, 60],
            dropouts=[0.1, 0.5, 0.5, 0],momentum_pretrain=0.9, learn_rates_pretrain=[0.001, 0.01, 0.01, 0.01],
            verbose=1),
     GradientBoostingClassifier(),




]
#
# classifiers = [sklearn.linear_model.LogisticRegression()] # 
#
# classifiers = [
#    sklearn.ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1),
# ]

print "training set shape:", x_train.shape

for i, classifier in enumerate(classifiers):

    print '-------------------- trying iteration ', i, '---------------------'
    try:
        print classifier
        classifier.fit(x_train, y_train)
        predictions = classifier.predict_proba(x_kaggle)[:, 1]
        output = '\n'.join([str(p) for p in predictions])

        n_submission = str(i + 6000) + 'submission-best-rf.csv'
        f = open('/home/blazej/projects/whale-kaggle/submission' + n_submission + '.csv', 'w')
        f.write(output)
        f.close()
        print '.Validation:', calculateAuc(classifier, x_test, y_test)
    except BaseException as e:
        print 'Had error', e





