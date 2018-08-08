"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import math


def create_lr_schedule(lr_base, decay_rate, decay_epochs, truncated_epoch, start_epoch=0, mode=None):
    return lambda epoch: _lr_schedule(epoch,  _lr_base=lr_base, _decay_rate=decay_rate,
                                      _decay_epochs=decay_epochs, _truncated_epoch=truncated_epoch, start_epoch=start_epoch, _mode=mode)


def _lr_schedule(_current_epoch, _lr_base=0.002, _decay_rate=0.1, _decay_epochs=500,
                 _truncated_epoch=None, start_epoch=100,_mode='constant'):
    if _truncated_epoch is None:
        _truncated_epoch = 2 * _decay_epochs

    if _mode is 'ladder':
        if _current_epoch < _truncated_epoch:
            learning_rate = _lr_base * _decay_rate**math.ceil(_current_epoch/_decay_epochs)
        else:
            learning_rate = _lr_base * _decay_rate**math.ceil(_truncated_epoch/_decay_epochs)
        
    elif _mode is 'exp':  # exponential_decay, exp for shorthand
        if _current_epoch < _truncated_epoch:
            learning_rate = _lr_base * _decay_rate ** (_current_epoch / _decay_epochs)
        else:
            learning_rate = _lr_base * _decay_rate ** (_truncated_epoch / _decay_epochs)
            
    elif _mode is 'constant':
            learning_rate = _lr_base
    elif _mode is "tube_trans":
        if _current_epoch < start_epoch:
            learning_rate = _lr_base
        elif _current_epoch < _truncated_epoch:
            learning_rate = _lr_base * _decay_rate ** ((_current_epoch-start_epoch)/(_truncated_epoch - start_epoch))
        else:
            learning_rate = _lr_base * _decay_rate

    else:
        raise Exception('Please select the defined _mode,i.e.,constant')
    return learning_rate


class create_lr_pieces(object):
    """Create Pieces Function
    anchors:  List, the independent variables ranges without start point
    functions: Str or scalar, the choices is as following:
                    c  --> constant
                    e  --> exponent
                    l  --> linear
    base & rate: Scalar, determine the values list if values are not provided
    values:  Values between each two close anchors

    Tips:
        values is a priority compared with base & rate
    """
    def __init__(self, anchors, functions, base=1.0, rate=1.0, values=None):
        self.num_anchor = len(anchors)
        self.anchors = anchors
        self.funcs = functions
        self.base = base
        self.rate = rate
        if values is None:
            self.values = self.get_values()
        else:
            self.values = values

        self.anchors.insert(0, 0)
        self.values.insert(-1, self.values[-1])
        self.funcs.insert(-1, self.funcs[-1])

    def apply(self, epoch):
        idx = 0
        while idx < self.num_anchor:
            if self.anchors[idx] <= epoch < self.anchors[idx+1]:
                return self.func_out(epoch, idx)
            idx += 1
        else:
            return self.values[-1]

    def func_out(self, epoch, idx):
        tag = self.funcs[idx]
        if tag == "c":
            return self.values[idx]

        vb, vf = self.values[idx], self.values[idx + 1]
        eb, ef = self.anchors[idx], self.anchors[idx + 1]

        if tag == "l":
            return vb + (vf-vb) * (epoch-eb)/(ef-eb)
        if tag == "e":
            return vb * (vf/vb) ** ((epoch-eb)/(ef-eb))
        else:
            print("the function is not in the scope")

    def get_values(self):
        values = [self.base]
        times = 1
        for tag in self.funcs:
            if tag is "c":
                values.append(values[-1])
            else:
                values.append(self.base * self.rate ** times)
                times += 1
        return values


if __name__ == '__main__':
    lr_schedule_c = create_lr_schedule(lr_base=1e-2, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="constant")
    lr_schedule_l = create_lr_schedule(lr_base=2e-2, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="ladder")
    lr_schedule_e = create_lr_schedule(lr_base=2e-2, decay_rate=0.1, decay_epochs=500,
                                       truncated_epoch=2000, mode="exp")


    import matplotlib.pylab as plt
    import numpy as np
    # learning_rate = []
    # for epoch in range(3000):
    #     learning_rate.append([lr_schedule_c(epoch), lr_schedule_l(epoch), lr_schedule_e(epoch)])
    #
    # lr = np.array(learning_rate)
    # print(lr.shape)
    # plt.plot(lr, '-*')
    # plt.legend(['constant', 'ladder', 'exponent decay'])
    # plt.show()

    lr_new = create_lr_pieces([1000, 2000, 3000, 4000, 5000],
                              ["c", "e", "c", "l", "c"], base=0.01, rate=0.1)
    # lr_new.get_values()
    # print(lr_new.func_out(500, 0), lr_new.func_out(1500, 1), lr_new.func_out(2500, 2), lr_new.func_out(3500, 3))
    lr = []
    for epoch in range(6000):
        lr.append(lr_new.apply(epoch))
    lr = np.array(lr)
    print(lr.shape)
    plt.plot(lr, '-*')
    plt.show()




