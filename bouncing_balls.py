from pylab import *
import pdb
import os
import cv2
from progressbar import ProgressBar
from argparse import ArgumentParser

# Adapted from Ruben Villegas, U Mich

class BouncingBallDataHandler(object):
    def __init__(self, num_balls, seq_length, batch_size, image_size):
        self.SIZE       = 10
        self.T          = seq_length
        self.n          = num_balls
        self.res        = image_size
        self.batch_size = batch_size

    def norm(self, x): return sqrt((x**2).sum())
    def sigmoid(self, x): return 1./(1.+exp(-x))

    def new_speeds(self, m1, m2, v1, v2):
        new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2

    # size of bounding box: self.SIZE X self.SIZE.

    def bounce_n(self, T=128, n=2, r=None, m=None):
        if r==None: r=array([1.2]*n)
        if m==None: m=array([1]*n)
        # r is to be rather small.
        X=zeros((T, n, 2), dtype='float')
        v = randn(n,2)
        v = v / self.norm(v)*.5
        good_config=False
        while not good_config:
            x = 2+rand(n,2)*8
            good_config=True
            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:      good_config=False
                    if x[i][z]+r[i]>self.SIZE:     good_config=False

            # that's the main part.
            for i in range(n):
                for j in range(i):
                    if self.norm(x[i]-x[j])<r[i]+r[j]:
                        good_config=False


        eps = .5
        for t in range(T):
            # for how long do we show small simulation

            for i in range(n):
                X[t,i]=x[i]

            for mu in range(int(1/eps)):

                for i in range(n):
                    x[i]+=eps*v[i]

                for i in range(n):
                    for z in range(2):
                        if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                        if x[i][z]+r[i]>self.SIZE: v[i][z]=-abs(v[i][z]) # want negative


                for i in range(n):
                    for j in range(i):
                        if self.norm(x[i]-x[j])<r[i]+r[j]:
                            # the bouncing off part:
                            w    = x[i]-x[j]
                            w    = w / self.norm(w)

                            v_i  = dot(w.transpose(),v[i])
                            v_j  = dot(w.transpose(),v[j])

                            new_v_i, new_v_j = self.new_speeds(m[i], m[j], v_i, v_j)

                            v[i]+= w*(new_v_i - v_i)
                            v[j]+= w*(new_v_j - v_j)

        return X

    def ar(self, x,y,z):
        return z/2+arange(x,y,z,dtype='float')

    def matricize(self, X,res,r=None):

        T, n= shape(X)[0:2]
        if r==None: r=array([1.2]*n)

        A=zeros((T,res,res), dtype='float')

        [I, J]=meshgrid(self.ar(0,1,1./res)*self.SIZE, self.ar(0,1,1./res)*self.SIZE)

        for t in range(T):
            for i in range(n):
                A[t]+= exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )

            A[t][A[t]>1]=1
        return A

    def bounce_mat(self, res, n=2, T=128, r =None):
        if r==None: r=array([1.2]*n)
        x = self.bounce_n(T,n,r);
        A = self.matricize(x,res,r)
        return A

    def bounce_vec(self, res, n=2, T=128, r =None, m =None):
        if r==None: r=array([1.2]*n)
        x = self.bounce_n(T,n,r,m);
        V = self.matricize(x,res,r)
        return V.reshape(T, res**2)

    def show_single_V(self, V):
        res = int(sqrt(shape(V)[0]))
        show(V.reshape(res, res))

    def show_V(self, V):
        T   = len(V)
        res = int(sqrt(shape(V)[1]))
        for t in range(T):
            print t
            show(V[t].reshape(res, res))

    def unsigmoid(self, x): return log (x) - log (1-x)

    def show_A(self, A):
        T = len(A)
        for t in range(T):
            show(A[t])

    def GetBatch(self):
        seq_batch = np.zeros( ( self.T,
                                self.batch_size,
                                self.res,self.res ) )

        for i in xrange(self.batch_size):
            seq = self.bounce_mat(self.res, self.n, self.T)
            seq_batch[:,i,:] = seq.reshape( seq.shape[0],
                                           seq.shape[1],seq.shape[2] )
        return seq_batch.astype('float32')

def make_dataset(root, mode, num_batches, num_balls, batch_size, image_size, subsample):
    """ Each sequence is a batch """
    handler = BouncingBallDataHandler(num_balls=num_balls, seq_length=batch_size*subsample, batch_size=1, image_size=150)
    data_root = os.path.join(root, mode) + '_nb=' + str(num_balls) + '_bsize=' + str(batch_size) + '_imsize=' + str(image_size) + '_subsamp=' + str(subsample)
    os.mkdir(data_root)
    pbar = ProgressBar()
    for i in pbar(range(num_batches)):
        batch_folder = os.path.join(data_root,'batch'+str(i))
        os.mkdir(batch_folder)
        x = handler.GetBatch()  # this is normalized between 0 and 1
        for j in range(0,batch_size*subsample,subsample):
            # pass
            cv2.imwrite(os.path.join(batch_folder,str(j/subsample)+'.png'),x[j,0,:,:]*255)
    pbar.finish()

if __name__ == "__main__":

    # handler = BouncingBallDataHandler(num_balls=3, seq_length=30, batch_size=1, image_size=150)
    # x = handler.GetBatch()

    parser = ArgumentParser()
    parser.add_argument('-m','--mode', type=str, default='train',
       help='train | test | val')
    parser.add_argument('-b','--batch_size', type=int, default=30,
       help='batch size')
    parser.add_argument('-n','--num_balls', type=int, default=3,
       help='Number of balls')
    parser.add_argument('-s','--subsample', type=int, default=3,
       help='How many frames to subample')
    parser.add_argument('-i','--image_size', type=int, default=150,
       help='Dimension of square image')
    # parser.add_argument('-e','--name', type=str, default='defaultballsname',
    #    help='name of job')
    args = parser.parse_args()

    root = '/om/data/public/mbchang/udcign-data/balls'
    # root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/data/udcign/balls'
    datasets = {'train':9000,'test':1000,'val':1000}

    mode = args.mode #'val'
    num_balls = args.num_balls#6
    batch_size = args.batch_size#30
    image_size = args.image_size#150
    subsample = args.subsample#5
    print mode, num_balls
    print(args)
    # TODO add subsample!
    make_dataset(root, mode, datasets[mode], num_balls, batch_size, image_size, subsample)
