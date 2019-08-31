#coding:utf-8
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

from necython import window_sampling, skip_sampling

class NegativeSampling(object):

    def __init__(self, window_size, batch_size, neg_ratio=5, neg_power=.75, down_sampling=-1):
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.neg_power = neg_power
        self.down_sampling = down_sampling

        self.samples = None

    def init_negative_probs(self, sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        weights = [counter[i] for i in range(len(counter))]
        weights = np.power(weights, self.neg_power)
        self.neg_probs = weights/np.sum(weights)

    def init_samples(self, sequences, shuffle=False):
        self.init_negative_probs(sequences)
        samples = window_sampling(sequences, self.window_size, self.down_sampling, shuffle)
        samples = np.array(samples)
        return samples

    def sample(self, sequences):
        if self.samples is None:
            self.samples = self.init_samples(sequences)
        samples = self.samples
        total_cnt = len(samples)
        pos_cnt = self.batch_size//(self.neg_ratio+1)
        neg_cnt = pos_cnt*self.neg_ratio
        neg_samples = np.random.choice(len(self.neg_probs), size=total_cnt*self.neg_ratio, p=self.neg_probs)

        bar = tqdm(total=total_cnt*(self.neg_ratio+1))
        bar.set_description('  Training')
        
        start_idx = 0
        end_idx = min(start_idx+pos_cnt, total_cnt)
        while start_idx<end_idx:
            pos_cnt_2 = end_idx-start_idx
            neg_cnt_2 = pos_cnt_2*self.neg_ratio

            samp = np.ndarray((pos_cnt_2+neg_cnt_2, 3), dtype=np.int32)
            samp[:pos_cnt_2,:2] = samples[start_idx:end_idx,:]
            samp[:pos_cnt_2,2] = 1
            samp[pos_cnt_2:,0] = np.repeat(samp[:pos_cnt_2,0], self.neg_ratio)
            samp[pos_cnt_2:,1] = neg_samples[start_idx*self.neg_ratio:end_idx*self.neg_ratio]
            samp[pos_cnt_2:,2] = -1
            yield samp[:,0], samp[:,1], samp[:,2]

            bar.update(pos_cnt_2*(self.neg_ratio+1))

            start_idx = end_idx
            end_idx = min(start_idx+pos_cnt, total_cnt)

        bar.close()

class TripletSampling(object):
    def __init__(self, window_size, batch_size, neg_power=.75, down_sampling=-1):
        self.window_size = window_size
        self.batch_size = batch_size
        self.neg_power = neg_power
        self.down_sampling = down_sampling

        self.samples = None

    def init_negative_probs(self, sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        weights = [counter[i] for i in range(len(counter))]
        weights = np.power(weights, self.neg_power)
        self.neg_probs = weights/np.sum(weights)

    def init_samples(self, sequences, shuffle=False):
        self.init_negative_probs(sequences)
        samples = window_sampling(sequences, self.window_size, self.down_sampling, shuffle)
        samples = np.array(samples)
        return samples

    def sample(self, sequences):
        if self.samples is None:
            self.samples = self.init_samples(sequences)
        samples = self.samples
        total = len(samples)
        neg_samples = np.random.choice(len(self.neg_probs), size=total, p=self.neg_probs)

        bar = tqdm(total=total)
        bar.set_description('  Training')
        
        start_idx = 0
        end_idx = min(start_idx+self.batch_size, total)
        while start_idx<end_idx:
            samp = np.ndarray((end_idx-start_idx, 3), dtype=np.int32)
            samp[:,:2] = samples[start_idx:end_idx,:]
            samp[:,2] = neg_samples[start_idx:end_idx]
            yield samp[:,0], samp[:,1], samp[:,2]

            bar.update(end_idx-start_idx)

            start_idx = end_idx
            end_idx = min(start_idx+self.batch_size, total)

        bar.close()

