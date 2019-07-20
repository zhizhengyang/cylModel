"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

@author: haggarwal
"""
import tensorflow as tf
import numpy as np
from os.path import expanduser
from utils import gaussian2D
import os
import glob
from skimage import io, transform
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)
mask = gaussian2D()#观测矩阵

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn=tf.layers.batch_normalization(x,training=trainning,fused=True,name='BN')
    if not(lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.
    """
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,1,64)
    szW[nLay]=(3,3,64,1)#最后的输出shape是2

    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with tf.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    print('---------',dw.shape)
    return dw





def getLambda():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.05)
    return lam

def getAtb(ori):
    ori = ori.reshape((1,2500))  
    b = np.matmul(ori, mask)
    AT = np.transpose(mask)
    atb = np.matmul(np.add(b,b),AT)
    atb = atb.reshape((1,50,50,1))
    return atb

def minibatches(inputs=None, targets=None, batch_size=1, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1,):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

def newModel(w, h, c):
    """
    This is the main function that creates the model.

    """
    out={}
    x = tf.placeholder(tf.float32, shape=[None, w, h,1], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, w, h,1], name='y_')
    A = tf.convert_to_tensor(mask,dtype=tf.float32)
    ori = tf.reshape(x,[1,2500])  
    b = tf.matmul(ori, A)
    training = True

    AT = tf.transpose(A)
    X0 = tf.matmul(b,AT)
    X0 = tf.reshape(X0,[50,50])
    atb = tf.matmul(tf.add(b,b),AT)
    atb = tf.reshape(atb,[1,50,50,1])
    out['dc0'] = atb
    with tf.name_scope('myModel'):
        with tf.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,5+1):
                j=str(i)
                out['dw'+j]=dw(out['dc'+str(i-1)],training,5)
                #lam1=getLambda()可能会用，先留着
                Zk = out['dw'+j]
                if training:
                    A = tf.reshape(A,[1,250,2500])
                    b = tf.reshape(b,[1,250,1])
                    out['dc'+j]=newDc(A,b,Zk,b)
    return out['dc'+str(5)],x,y_

def newDc(A,b,Zk,noise):
    def fn(tmp):
        A,b,Zk,noise = tmp
        print(A.shape)
        AT = tf.transpose(A)
        m = tf.add(b,noise)
        #print(AT.shape,m.shape)
        atb = tf.matmul(AT,m)
        #print(atb.shape)
        Xn = tf.matmul(AT,A)
        NI = tf.eye(2500)
        Xn0 = tf.add(Xn,NI)
        Xn0 = tf.matrix_inverse(Xn0)#求逆
        atb = tf.reshape(atb,[50,50,1])
        Xn1 = tf.add(atb,Zk)
        Xn1 = tf.reshape(Xn1,[2500,1])
        return tf.matmul(Xn0,Xn1)
    inp = (A,b,Zk,noise)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn2' )
    rec = tf.reshape(rec,[1,50,50,1])
    return rec

def runable(x_train, y_train, optimizer, loss, x, y_, x_val, y_val):
    # 训练和测试数据，可将n_epoch设置更大一些
    n_epoch = 50
    batch_size = 1
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        # training
        train_loss,  n_batch = 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            _, err = sess.run([optimizer, loss], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            n_batch += 1
            print("train loss: %f" % (train_loss / n_batch))
        print('=======',epoch)
        #print("train loss: %f" % (train_loss))
        saver.save(sess, 'save_model/model_name.ckpt')
        print('model saved in file:save_model')
    sess.close()

def read_img(label_path,path,w, h):
    cate = os.listdir(path)
    # print(cate)
    label_dir = 'data/'     #label的地址
    imgs = []
    labels = []
    i = 0
    print('Start read the image ...')
    for im in glob.glob(path + '/*.jpg'):
        # print('Reading The Image: %s' % im)
        img = io.imread(im)
        img = transform.resize(img, (w, h))
        label_img = io.imread(label_path+im.split('/')[-1])
        #img = getAtb(img[:,:,0])
        imgs.append(img[:,:,0].reshape((50,50,1)))
        labels.append(img[:,:,0].reshape((50,50,1)))
        if i % 100 == 0:
            print(i)
        if i > 1000:
            break
        i = i + 1
    print('Finished ...')

    return np.asarray(imgs, np.float), np.asarray(labels, np.float32)

def segmentation(data, label, ratio=0.8):
    num_example = data.shape[0]
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    print('------------',x_train.shape)
    return x_train, y_train, x_val, y_val

def accUNET(conv1, y_):
    loss = tf.reduce_sum(tf.square(conv1 - y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    return loss, optimizer

if __name__ == '__main__':
    imgpath = 'data/'
    w = 50
    h = 50
    c = 1

    ratio = 0.8  # 选取训练集的比例
    data, label = read_img(label_path=imgpath,path=imgpath, w=w, h=h)
    x_train, y_train, x_val, y_val = segmentation(data=data, label=label, ratio=ratio)

    pre,x, y_ = newModel(w=w, h=h, c=c)

    loss, optimizer = accUNET(pre, y_=y_)

    runable(x_train=x_train, y_train=y_train, optimizer=optimizer, loss=loss,
             x=x, y_=y_, x_val=x_val, y_val=y_val)


#newModel()
