# import numpy as np

# # Source: https://github.com/sacmehta/ESPNet/blob/master/train/IOUEval.py

# class IOUEval:
#     def __init__(self, nClasses):
#         self.nClasses = nClasses
#         self.reset()

#     def reset(self):
#         self.overall_acc = 0
#         self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
#         self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
#         self.mIOU = 0
#         self.batchCount = 1

#     def fast_hist(self, a, b):
#         k = (a >= 0) & (a < self.nClasses)
#         return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)

#     def compute_hist(self, predict, gth):
#         hist = self.fast_hist(gth, predict)
#         return hist

#     def addBatch(self, predict, gth):
#         predict = predict.cpu().numpy().flatten()
#         gth = gth.cpu().numpy().flatten()

#         epsilon = 0.00000001
#         hist = self.compute_hist(predict, gth)
#         overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
#         per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
#         per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
#         mIou = np.nanmean(per_class_iu)

#         self.overall_acc +=overall_acc
#         self.per_class_acc += per_class_acc
#         self.per_class_iu += per_class_iu
#         self.mIOU += mIou
#         self.batchCount += 1

#     def getMetric(self):
#         overall_acc = self.overall_acc/self.batchCount
#         per_class_acc = self.per_class_acc / self.batchCount
#         per_class_iu = self.per_class_iu / self.batchCount
#         mIOU = self.mIOU / self.batchCount

#         return overall_acc, per_class_acc, per_class_iu, mIOU


import numpy as np

class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()

    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_dice = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.dice = 0
        self.batchCount = 0

    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses**2).reshape(self.nClasses, self.nClasses)

    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist

    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()

        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

        TP = np.diag(hist)
        FP = hist.sum(axis=0) - TP
        FN = hist.sum(axis=1) - TP
        per_class_dice = 2 * TP / (2 * TP + FP + FN + epsilon)

        self.overall_acc += overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.per_class_dice += per_class_dice
        self.mIOU += np.nanmean(per_class_iu)
        self.dice += np.nanmean(per_class_dice)
        self.batchCount += 1

    def getMetric(self):
        overall_acc = self.overall_acc / self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        per_class_dice = self.per_class_dice / self.batchCount
        mIOU = self.mIOU / self.batchCount
        dice = self.dice / self.batchCount

        return overall_acc, per_class_acc, per_class_iu, mIOU, per_class_dice, dice
