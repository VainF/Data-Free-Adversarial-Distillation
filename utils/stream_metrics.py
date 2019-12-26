import numpy as np
from sklearn.metrics import confusion_matrix

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.ignore_index = ignore_index

    def update(self, label_preds, label_trues):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )
    
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes) & (
            label_true != self.ignore_index)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


class StreamDepthMetrics(_StreamMetrics):
    """This metric is used in depth prediction task.
    **Parameters:**
        - **thresholds** (list of float)
        - **ignore_index** (int, optional): Value to ignore.
    """
    def __init__(self, thresholds, ignore_index=0):
        self.thresholds = thresholds
        self.ignore_index = ignore_index
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        """
        **Type**: numpy.ndarray or torch.Tensor
        **Shape:**
            - **preds**: $(N, H, W)$. 
            - **targets**: $(N, H, W)$. 
        """
        self.preds = np.append(self.preds, preds)
        self.targets = np.append(self.targets, targets)

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="percents within thresholds":
                string += "%s: %f\n"%(k, v)
        
        string+='percents within thresholds:\n'
        for k, v in results['percents within thresholds'].items():
            string += "\tthreshold %f: %f\n"%(k, v)
        return string

    def get_results(self):
        """
        **Returns:**
            - **absolute relative error**
            - **squared relative error**
            - **precents for $r$ within thresholds**: Where $r_i = max(preds_i/targets_i, targets_i/preds_i)$
        """
        masks = self.targets != self.ignore_index
        count = np.sum(masks)
        self.targets = self.targets[masks]
        self.preds = self.preds[masks]

        diff = np.abs(self.targets - self.preds)
        sigma = np.maximum(self.targets / self.preds, self.preds / self.targets)
        rmse = np.sqrt( (diff**2).sum() / count )


        ard = diff / self.targets
        ard = np.sum(ard) / count

        srd = diff * diff / self.targets
        srd = np.sum(srd) / count

        threshold_percents = {}
        for threshold in self.thresholds:
            threshold_percents[threshold] = np.nansum((sigma < threshold)) / count

        return {
            'rmse': rmse,
            'absolute relative': ard,
            'squared relative': srd,
            'percents within thresholds': threshold_percents
        }

    def reset(self):
        self.preds = []
        self.targets = []