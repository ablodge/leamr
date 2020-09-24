import math

from scipy.stats import skellam, norm


class Gaussian_Distance_Model:

    def __init__(self):

        self.distance_mean = 0
        self.distance_stdev = 50
        self._distribution_memo = {}

    def update_parameters(self, mean, stdev):
        self._distribution_memo = {}
        self.distance_mean = mean
        self.distance_stdev = stdev
        if self.distance_stdev <= 0:
            self.distance_stdev = 1

    def distance(self, amr, tokens1, tokens2):
        if not tokens1 or not tokens2:
            return self.distance_stdev
        idx1 = None
        idx2 = None
        for i,span in enumerate(amr.spans):
            if tokens1[0] in span:
                idx1 = i
            if tokens2[0] in span:
                idx2 = i
            if idx1 is not None and idx2 is not None:
                break
        return 100*(idx2 - idx1)/len(amr.spans)

    def logp(self, dist):
        dist = round(dist, 4)
        # max_dist = 100
        # min_dist = -100
        # if dist>max_dist: dist = max_dist
        # if dist<min_dist: dist=min_dist
        if dist in self._distribution_memo:
            logp = self._distribution_memo[dist]
        else:
            p = norm.pdf(dist, loc=self.distance_mean, scale=self.distance_stdev)
            if p <= 1e-6:
                p = 1e-6
            logp = math.log(p)
            self._distribution_memo[dist] = logp
        return logp


class Skellam_Distance_Model:

    def __init__(self):

        self.distance_mean = 0
        self.distance_stdev = 10
        self._distribution_memo = {}
        self.mu2 = (self.distance_stdev ** 2 - self.distance_mean) / 2
        self.mu1 = self.distance_mean + self.mu2

    def update_parameters(self, mean, stdev):
        self._distribution_memo = {}
        self.distance_mean = mean
        self.distance_stdev = stdev
        if self.distance_stdev <= 0:
            self.distance_stdev = 1
        self.mu2 = (self.distance_stdev ** 2 - self.distance_mean) / 2
        self.mu1 = self.distance_mean + self.mu2

    def distance(self, amr, tokens1, tokens2):
        if not tokens1 or not tokens2:
            return self.distance_stdev
        idx1 = None
        idx2 = None
        for i,span in enumerate(amr.spans):
            if tokens1[0] in span:
                idx1 = i
            if tokens2[0] in span:
                idx2 = i
            if idx1 is not None and idx2 is not None:
                break
        return idx2 - idx1

    def logp(self, dist):
        dist = int(dist)
        max_dist = 100
        min_dist = -100
        if dist>max_dist: dist = max_dist
        if dist<min_dist: dist=min_dist
        if dist in self._distribution_memo:
            logp = self._distribution_memo[dist]
        else:
            p = skellam.pmf(dist, mu1=self.mu1, mu2=self.mu2)
            if p <= 1e-6:
                p = 1e-6
            logp = math.log(p)
            self._distribution_memo[dist] = logp
        return logp
