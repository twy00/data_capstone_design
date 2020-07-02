from tensorflow.python.util import deprecation
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import random
import sys
import os
import utils
import logging
import random
from itertools import combinations
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import PIL
from tensorflow.contrib.tensorboard.plugins import projector
import cv2
logging.getLogger('tensorflow').disabled = True
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import time
import csv
import shutil
import queue
from PIL import Image
import PIL.Image as pilimg
loc = {"13":"Atelectasis", "7":"Cardiomegaly", "11":"Consolidation", "10":"Edema", "15":"Pleural Effusion"}
label = {"Atelectasis":0, "Cardiomegaly":1, "Consolidation":2, "Edema":3, "Pleural Effusion":4}

class0, class1, class2, class3, class4, class5 = 5400, 5400, 2700, 9500, 13500, 3000
testset = 400
test_size = 1000
minimum = 2700

data_num =  {0:class0+testset, 1:class1+testset, 2:class2+testset, 3:class3+testset, 4:class4+testset, 5:class5+testset}
data_count = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
classes = [7, 10, 11, 13, 15]
path = "/data/"

class TrainOps(object):

    def __init__(self, model, mode = '', gan_type = '', generator='',loss_function = '', dataset = '', num_labels = 10, image_size_width = 28,image_size_height = 28,  image_depth = 1, epoch = 50,  num_major_set = 2, num_major_data = 6000, num_minor_data = 150,  
                num_gan_data = 150,log_dir='./logs/logs', model_save_path='.model/', train_rate = 1, train_feature_generator_iters = 150001,batch_size=80,
                pretrained_feature_extractor='feature_extractor', pretrained_feature_generator='feature_generator', pretrain_epochs=50,  redefine=False,
                make_tsne=0, save_accuracy=1, save_file_name='',classifier= '', image_size=500, extractor_learning_rate = 1e-4, generator_learning_rate=1e-4, discriminator_learning_rate=1e-4, 
                classifier_learning_rate=1e-4, import_weight="", note='', noise_dim = 1000, noise_type = "",
                d_iter=5, g_iter=5, cri=5, cfi=5):

        self.model = model
        self.batch_size = batch_size
        self.epoches = epoch
        self.mode = mode
        # Number of iterations for Step 0, 1, 2.
        self.pretrain_epochs = pretrain_epochs
        self.train_feature_generator_iters = train_feature_generator_iters
        self.import_weight = import_weight

        if not os.path.isdir("extractor"):
            os.mkdir("extractor/")
        if not os.path.isdir("generator"):
            os.mkdir("generator/")
        if not os.path.isdir("classifier"):
            os.mkdir("classifier/")
            
        # Dataset directory
        self.dataset = dataset
        self.num_labels = num_labels
        self.gan_type = gan_type
        self.loss_function = loss_function
        self.classifier = classifier
        self.generator = generator
        self.image_size = image_size
        if not self.import_weight == "":
            self.curtime = int(self.import_weight[:import_weight.find("_")])
        else:
            self.curtime = int(time.time())
        print("time: {}".format(int(self.curtime)))
        self.model_save_path = model_save_path  + self.dataset + "_" + self.classifier + "_" +  self.gan_type + '_' + self.generator + '_' + self.loss_function  

        self.extractor_learning_rate = extractor_learning_rate
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.classifier_learning_rate = classifier_learning_rate
        
        with open("log.txt", 'a') as writefile:
            writefile.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(str(int(self.curtime)),dataset, classifier, gan_type, generator, loss_function, extractor_learning_rate, generator_learning_rate, discriminator_learning_rate, classifier_learning_rate, note))
            writefile.write("\n")
        
        self.pretrained_feature_extractor = self.gan_type + '_' + self.loss_function  +  '_' + self.classifier + '_' +  pretrained_feature_extractor + '_' + str(self.pretrain_epochs)
        self.pretrained_feature_extractor = os.path.join(self.model_save_path, self.pretrained_feature_extractor)
        # print("pretrained_feature_extractor")
        # print(self.pretrained_feature_extractor)

        self.pretrained_feature_generator = self.gan_type +'_'+ self.loss_function + '_' + self.classifier + '_' +  pretrained_feature_generator + '_' + str(self.train_feature_generator_iters)

        self.training_image_save_path = os.path.join(self.model_save_path, './training_feature')
        self.training_image_save_path = os.path.join(self.training_image_save_path, self.pretrained_feature_generator)

        self.real_image_save_path = os.path.join(self.model_save_path, './real_feature')
        self.real_image_save_path = os.path.join(self.real_image_save_path, self.pretrained_feature_generator)

        self.gen_image_save_path = os.path.join(self.model_save_path, 'gen_feature_'+str(self.curtime))

        self.pretrained_feature_generator = os.path.join(self.model_save_path, self.pretrained_feature_generator)
        # print("pretrained_feature_generator")
        # print(self.pretrained_feature_generator)
        
        self.log_dir = log_dir + "_" + self.gan_type + '_' + self.loss_function  +  '_' + self.classifier + '_' +  pretrained_feature_extractor +  '_' + str(self.pretrain_epochs) + "/"
        # print("logdir")
        # print(self.log_dir)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True

        self.image_size_width = image_size_width
        self.image_size_height = image_size_height
        self.image_depth = image_depth

        self.make_tsne = make_tsne
        self.save_accuracy = save_accuracy
        self.save_file_name = save_file_name + "_"

        self.diter, self.noise_dim =d_iter, noise_dim
        self.giter=g_iter
        self.cri=cri
        self.cfi=cfi

        self.noise_type=noise_type

    # CheXpert Dataset 로딩하기
    def load_image(self, size=250, mode='train_feature_extractor'):
        print("loading data...")
        start = time.time()
        x_train, y_train = [],[]
        ccount = 0
        with open("data.csv", "r") as datacsv:
            with open("/data/CheXpert-v1.0-small/train.csv") as imagecsv:
                datareader = list(csv.reader(datacsv))
                imagereader = list(csv.reader(imagecsv))
                count = 0
                for row in datareader:
                    data = int(row[0])
                    label = int(row[1])
                    if data_count[label] < data_num[label]:
                        im = pilimg.open(path + imagereader[data][0])
                        im_resize = utils.make_square(im)
                        cv2_image = np.array(im_resize)[:,:,:3]
                        shrink = cv2.resize(cv2_image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                        pix = np.array(shrink)
                        x_train.append(pix)
                        y_train.append(label)
                        data_count[label] += 1
                        ccount += 1
                        print("{}, data: {}".format(data_count, ccount), end="\r")

                        if data_count == data_num:
                            break

        for i in range(5):
            train_shuffle = list(zip(x_train,y_train))
            random.shuffle(train_shuffle)
            x_train,y_train = zip(*train_shuffle)
        
        print(data_count)

        xtrain, ytrain, xtest, ytest = [], [], [], []
        datalist = [[],[],[],[],[],[]]
        for u in range(len(y_train)):
            temp = []
            temp.append(x_train[u])
            temp.append(y_train[u])
            datalist[y_train[u]].append(temp)

        for t in range(minimum):
            pair = datalist[0].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[0].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])
        
        for t in range(minimum):
            pair = datalist[1].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[1].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])

        for t in range(minimum):
            pair = datalist[2].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[2].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])

        for t in range(minimum):
            pair = datalist[3].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[3].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])

        for t in range(minimum):
            pair = datalist[4].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[4].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])

        for t in range(minimum):
            pair = datalist[5].pop()
            xtrain.append(pair[0])
            ytrain.append(pair[1])
        for t in range(testset):
            pair = datalist[5].pop()
            xtest.append(pair[0])
            ytest.append(pair[1])      
        
        train_shuffle = list(zip(xtrain,ytrain))
        test_shuffle = list(zip(xtest,ytest))

        random.shuffle(train_shuffle)
        random.shuffle(test_shuffle)

        xtrain,ytrain = zip(*train_shuffle)
        xtest,ytest = zip(*test_shuffle)

        a,b,c,d = np.asarray(xtrain), np.asarray(ytrain), np.asarray(xtest), np.asarray(ytest)
        print("x_train shape: {}".format(a.shape))
        print("x_test shape: {}".format(c.shape))

        # a = a/255
        # c = c/255
        
        print("data loaded")
        

        return a,b,c,d

    # Test 데이터로 모델 평가하기
    def evaluate(self, model, x_data, y_data):
        num_examples = len(x_data)
        total_accuracy = 0
        total_loss = 0
        evaluate_keep_prob = 0.5
        evaluate_batch_prob = True
        sess = tf.get_default_session()

        for start, end in zip(range(0, len(x_data), self.batch_size), range(self.batch_size, len(x_data), self.batch_size)):
            batch_x, batch_y = x_data[start:end], y_data[start:end]
            if self.mode == "train_feature_generator":
                accuracy, loss = sess.run((model.accuracy, model.c_loss_real), feed_dict={model.images: batch_x, model.labels: batch_y, model.keep_prob:evaluate_keep_prob,model.batch_prob:evaluate_batch_prob})
            if self.mode == "train_feature_extractor":
                accuracy, loss = sess.run((model.accuracy, model.loss), feed_dict={model.images: batch_x, model.labels: batch_y, model.keep_prob:evaluate_keep_prob,model.batch_prob:evaluate_batch_prob})
            total_accuracy += accuracy * self.batch_size
            total_loss += loss * self.batch_size

        return total_accuracy / num_examples, total_loss / num_examples

    # Step 1: Feature Extractor 학습하기
    def train_feature_extractor(self):
        
        #copy
        dst = "extractor_backup/" + str(self.curtime) + ".py"
        src = "gan.py"

        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copyfile(src, dst)

        #훈련중인 모델 불러오기
        if not self.import_weight == "":
            extractor_variables = {}
            with tf.Session(config=self.config) as sess:
                sess.run(tf.global_variables_initializer())
                print ('Loading pretrained feature extractor...')
                saver = tf.train.import_meta_graph(str("extractor/" + self.import_weight + ".meta"))
                saver.restore(sess,str("extractor/" + self.import_weight))
                graph = tf.trainable_variables()
                for g in graph:
                    extractor_variables[g.name] = g.eval()
            tf.reset_default_graph()
        
        best_acc, best_loss, cepoch = 0, 10, 0
        train_keep_prob = 0.5
        train_batch_prob = True

        current_op = ''
        if self.mode == 'all':
            current_op = 'train_feature_extractor'

        self.x_train, self.y_train, self.x_test, self.y_test = self.load_image(size=self.image_size)

        print("size of train data: " + str(len(self.x_train)) + ", " + str(len(self.y_train)))
        print("size of test data : " + str(len(self.x_test)) + ", " + str(len(self.y_test)))
        # build a graph
        model = self.model
        model.set_image_dimension(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3])
        self.sess = model.build_model(current_op = current_op)
        
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            if not self.import_weight == "":
                extractor_filter = [g for g in tf.trainable_variables() if "feature_extractor" in g.name]
                for e in extractor_filter:
                    sess.run(tf.assign(e, extractor_variables[e.name]))
                cur_epoch = int(self.import_weight[self.import_weight.find("_") + 1:])
                print("Training extractor from Epoch {}".format(cur_epoch+1))
            else:
                cur_epoch = 0


            t = 0
            print("start training...batch size: {}".format(self.batch_size))
            for i in range(cur_epoch, self.pretrain_epochs):
            
                print ('Epoch',str(i+1))
                loss = 0
                accuracy = 0

                if i < 100:
                    self.extractor_learning_rate = self.extractor_learning_rate
                elif i ==100:
                    self.extractor_learning_rate = self.extractor_learning_rate / 5
                elif i == 200:
                    self.extractor_learning_rate = self.extractor_learning_rate / 5

                # self.extractor_learning_rate = self.extractor_learning_rate * 0.9
                    
                for start, end in zip(range(0, len(self.x_train), self.batch_size), range(self.batch_size, len(self.x_train), self.batch_size)):
                    
                    if start % self.batch_size == 0:
                        print(start, end="\r")
                    t+=1
                    feed_dict = {model.images: self.x_train[start:end], model.labels: self.y_train[start:end], model.extractor_learning_rate:self.extractor_learning_rate, model.keep_prob:train_keep_prob, model.batch_prob:train_batch_prob}  
                    
                    sess.run(model.train_op, feed_dict) 
            

                rand_idxs = np.random.permutation(self.x_train.shape[0])[:1000]
                train_acc, train_loss = self.evaluate(model, self.x_train[rand_idxs], self.y_train[rand_idxs])
                print ('Step: [%d/%d] loss: [%.4f] accuracy: [%.4f]'%(i+1, self.pretrain_epochs, train_loss, train_acc))

                validation_accuracy, _ = self.evaluate(model, self.x_test, self.y_test)
                print("EPOCH [%d] Validation Accuracy = [%.3f]"%(i+1, validation_accuracy))
            

                if (i+1) % 5 == 0:
                    self.extractor_learning_rate = self.extractor_learning_rate * 0.95
                if best_acc < validation_accuracy:
                    best_acc = validation_accuracy
                    
                    if self.save_accuracy == 1:
                        
                        filename = "model/accuracy_extractor" + str(self.pretrain_epochs) + "_" + str(int(self.curtime))  + "_accuracy_extractor.txt"
                        if not os.path.isfile(filename):
                            save_result = open(filename, "w")
                        else:
                            save_result = open(filename, "a") 
                        
                        save_result.write("Epoch "+ str(i+1) + ": " + str(best_acc) + "\n")
                        save_result.close()
                    
                        if (i+1) % 2 == 0 and (i+1) < 20:
                            print ('Saving')
                            saver.save(sess, "extractor/" + str(int(self.curtime))+ "_" +  str(i+1)) 
                        
                        elif (i+1) >= 20 and best_acc > 0.57:
                            print ('Saving')
                            saver.save(sess, "extractor/" + str(int(self.curtime))+ "_" +  str(i+1)) 
                            
                if train_loss < best_loss:
                    best_loss = train_loss


    # Step 2: Feature Generator 학습하기
    def train_feature_generator(self):
        train_keep_prob = 0.5
        train_batch_prob = True
        extractor_variables = {}
        generator_variables = {}
        best_acc = 0.59
        closs_queue = queue.Queue(5)
        
        #copy
        dst = "generator_backup/" + str(self.curtime) + "_gan.py"
        src = "gan.py"
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copyfile(src, dst)

        dst = "generator_backup/" + str(self.curtime) + "_trainOps.py"
        src = "trainOps.py"
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copyfile(src, dst)

        #  Step 2 중간에 저장한 Generator weight값 불러오기
        if not self.import_weight == "":
            print("loading pretrained feature generator")
            generator_weight = []
            with tf.Session(config=self.config) as sess:
                tf.global_variables_initializer().run()
                saver = tf.train.import_meta_graph("generator/" + self.import_weight + ".meta")
                saver.restore(sess, "generator/" + self.import_weight)
                graph = tf.trainable_variables()

                for g in graph:
                    generator_variables[g.name] = g.eval()   
                print("Generator Loaded")
            tf.reset_default_graph()

        # Step 1 Extractor weight 가져오고 feature 뽑기
        with tf.Session(config=self.config) as sess:

            tf.global_variables_initializer().run()
            print ('Loading pretrained feature extractor.')
            saver = tf.train.import_meta_graph("extractor/[STEP1EXTRACTOR_WEIGHT]" + ".meta") # Step 1 학습 후 직접 weight 값 불러오기
            saver.restore(sess, "extractor/[STEP1EXTRACTOR_WEIGHT]")

            graph = tf.trainable_variables()
            
            for g in graph:
                extractor_variables[g.name] = g.eval()

            print ('Extractor Loaded')
        tf.reset_default_graph()

        current_op = ''
        if self.mode == 'all':
            current_op = 'train_feature_generator'
        
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_image(size=self.image_size)
        self.y_train = utils.one_hot(self.y_train, self.num_labels)
        self.y_test = utils.one_hot(self.y_test, self.num_labels)
        
        # Extractor Weight 적용하기
        model = self.model
        model.set_image_dimension(self.x_train.shape[2], self.x_train.shape[1], self.x_train.shape[3])
        self.sess = model.build_model(current_op = "train_feature_extractor")
        
        with tf.Session(config=self.config) as sess:
            sess.run(tf.global_variables_initializer())
            extractor_filter = [g for g in tf.trainable_variables() if "feature_extractor" in g.name]
            for e in extractor_filter:
                sess.run(tf.assign(e, extractor_variables[e.name]))
                del extractor_variables[e.name]
        

            # extractor 뒷 부분을 classifier로 변환

            real_features = []
            real_features_test = []
            print("Extracting features...")
            for start, end in zip(range(0, len(self.x_train)+self.batch_size, self.batch_size), range(self.batch_size, len(self.x_train)+self.batch_size, self.batch_size)):
                    print("{} to {}".format(start, end), end="\r")
                    feed_dict = {model.images: self.x_train[start:end], model.labels: self.y_train[start:end], model.extractor_learning_rate:self.extractor_learning_rate, model.keep_prob:1.0, model.batch_prob:True}  
                    batch_feat = sess.run(model.logits, feed_dict) 
                    for feat in batch_feat:
                        real_features.append(feat)

            for start, end in zip(range(0, len(self.x_test)+1, self.batch_size), range(self.batch_size, len(self.x_test)+1, self.batch_size)):
                    print("{} to {}".format(start, end), end="\r")
                    feed_dict = {model.images: self.x_test[start:end], model.labels: self.y_test[start:end], model.extractor_learning_rate:self.extractor_learning_rate, model.keep_prob:1.0, model.batch_prob:True}  
                    batch_feat = sess.run(model.logits, feed_dict) 
                    for feat in batch_feat:
                        real_features_test.append(feat)
            
            self.x_train = np.asarray(real_features)
            self.x_test = np.asarray(real_features_test)
            print("Extracted features shape: {}".format(self.x_train.shape))
        
        tf.reset_default_graph()

        # build a graph
        model = self.model
        model.set_image_dimension(self.x_train.shape[2], self.x_train.shape[1], self.x_train.shape[3])
        self.sess = model.build_model(current_op = current_op)
        #noise_dim = 100

        classifier_variables = [g for g in tf.trainable_variables() if "feature_classifier" in g.name]
        
        # Step 2 Training
        with tf.Session(config=self.config) as sess:
            
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            print("extractor to classifier")
            for v in range(len(list(extractor_variables.keys()))):
                print("{} to {}".format(classifier_variables[v].name, list(extractor_variables.keys())[v]))
                try:
                    sess.run(tf.assign(classifier_variables[v], extractor_variables[list(extractor_variables.keys())[v]]))
                    print("assigned")
                except:
                    print("not assigned")
                
            
            
            closs_avg = 1.5
            # Generator, Discriminator, Classifier weight 적용하기
            if not self.import_weight == "":
                print("Loading pretrained generator.")
                gen_weight = [g for g in tf.trainable_variables() if "feature_generator" in g.name or "feature_discriminator" in g.name or "feature_classifier" in g.name]
                for e in gen_weight:
                    sess.run(tf.assign(e, generator_variables[e.name]))
                cur_step = int(self.import_weight[self.import_weight.find("_") + 1:])
                print("Training generator from Step {}".format(cur_step+1))
            else:
                cur_step = 0
            
            #classifier weight
            print("assigned completed")
            test_best = 0
            disc_count, gen_count, cla_count, cla_count2 = 0, 0, 0, 0
            plot_x, acc_plot_y, closs_ploy_y = [], [], []
            print("start training...batch size: {}".format(self.batch_size))
            for step in range(cur_step, self.train_feature_generator_iters):
                starttime = time.time()
                target_real = np.ones((self.batch_size,1), dtype=float)
                target_fake = np.zeros((self.batch_size,1), dtype=float)

                for D_step in range(self.diter):
                    i = disc_count % int(self.x_train.shape[0] / self.batch_size)
                    disc_count += 1
                    print("D step {} disc count: {} to {}".format(step, i*self.batch_size, (i+1)*self.batch_size), end="\r")

                    images_batch = self.x_train[i*self.batch_size:(i+1)*self.batch_size]
                    labels_batch = self.y_train[i*self.batch_size:(i+1)*self.batch_size]
                    noise = utils.sample_Z(self.batch_size, self.noise_dim, self.noise_type)

                    feed_dict = {model.noise: noise, model.real_features: images_batch, model.labels: labels_batch, model.target_real:target_real, model.target_fake:target_fake}
                    
                    real_feature = np.asarray(feed_dict[model.real_features])
                    sess.run(model.d_adam, feed_dict)
                
                for G_step in range(self.giter):
                    g = gen_count % int(self.x_train.shape[0] / self.batch_size)
                    gen_count += 1
                    images_batch = self.x_train[g*self.batch_size:(g+1)*self.batch_size]
                    labels_batch = self.y_train[g*self.batch_size:(g+1)*self.batch_size]
                    noise = utils.sample_Z(self.batch_size, self.noise_dim, self.noise_type)
                    
                    feed_dict = {model.noise: noise, model.real_features: images_batch, model.labels: labels_batch, model.target_real:target_real, model.target_fake:target_fake}
                    g_train_op = sess.run(model.g_adam, feed_dict)
                
                if (step+1) % self.cfi == 0:
                    c = cla_count % int(self.x_train.shape[0] / self.batch_size)
                    cla_count += 1

                    images_batch = self.x_train[c*self.batch_size:(c+1)*self.batch_size]
                    labels_batch = self.y_train[c*self.batch_size:(c+1)*self.batch_size]
                    noise = utils.sample_Z(self.batch_size, self.noise_dim, self.noise_type)

                    feed_dict = {model.noise: noise, model.real_features: images_batch, model.labels: labels_batch, model.target_real:target_real, model.target_fake:target_fake}
                    sess.run(model.c_adam_fake, feed_dict)

                if (step+1) % self.cri == 0:
                    c = cla_count2 % int(self.x_train.shape[0] / self.batch_size)
                    cla_count2 += 1

                    images_batch = self.x_train[c*self.batch_size:(c+1)*self.batch_size]
                    labels_batch = self.y_train[c*self.batch_size:(c+1)*self.batch_size]
                    noise = utils.sample_Z(self.batch_size, self.noise_dim, self.noise_type)

                    feed_dict = {model.noise: noise, model.real_features: images_batch, model.labels: labels_batch, model.target_real:target_real, model.target_fake:target_fake}
                    sess.run(model.c_adam_real, feed_dict)

                    noise = utils.sample_Z(self.batch_size, self.noise_dim, self.noise_type)
                    sess.run(model.c_adam, feed_dict)

                if (step+1) % self.batch_size == 0:
                    
                    rand_idxs = np.random.permutation(self.x_train.shape[0])[:test_size]
                    target_real = np.ones((test_size,1), dtype=int)
                    target_fake = np.zeros((test_size,1), dtype=int)
                    noise = utils.sample_Z(test_size, self.noise_dim, self.noise_type)
                    feed_dict = {model.noise: noise, model.real_features:self.x_train[rand_idxs], model.labels:self.y_train[rand_idxs], model.target_real:target_real, model.target_fake:target_fake}

                    avg_D_fake = sess.run(model.logits_fake, feed_dict)
                    avg_D_real = sess.run(model.logits_real, feed_dict)             

                    dl, gl, cl = sess.run([model.d_loss, model.g_loss, model.c_loss], feed_dict)
                    print ('Step: [%d/%d] d_loss: %.6f g_loss: %.6f c_loss: %.6f avg_d_fake: %.2f avg_d_real: %.2f ' \
                        %(step+1, self.train_feature_generator_iters, dl, gl, cl, avg_D_fake.mean(), avg_D_real.mean()))                    

                    validation_accuracy, _ = self.evaluate(model, self.x_test, self.y_test)

                    if validation_accuracy > test_best:
                        test_best = validation_accuracy

                    print("Step: [%d/%d] Validation Accuracy: [%.3f], best accuracy: [%.3f]  Exec Time: [%.3f]"%(step+1,self.train_feature_generator_iters, validation_accuracy, test_best, time.time()-starttime))
                    
                    plot_x.append(step+1)
                    acc_plot_y.append(validation_accuracy)
                    closs_ploy_y.append(cl)

                    plt.plot(plot_x, acc_plot_y, color='b')
                    plt.plot(plot_x, closs_ploy_y, color='r')
                    plt.savefig("step2/step2_" + str(self.curtime) + ".png")

                    

                    if os.path.isfile("model/generator_" + str(self.curtime) + ".txt"):
                        writeacc = open("model/generator_" + str(self.curtime) + ".txt", "a")
                    else:
                        writeacc = open("model/generator_" + str(self.curtime) + ".txt", "w")
                    writeacc.write("step: {}, c loss: {}, accuracy: {}, best: {}, avg_d_fake: {}, avg_d_real: {} \n".format(step+1, cl, validation_accuracy, test_best, avg_D_fake.mean(), avg_D_real.mean()))

  