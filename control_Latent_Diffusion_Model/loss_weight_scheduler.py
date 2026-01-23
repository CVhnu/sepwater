#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

@author: Tu Bui @University of Surrey
"""

class SimpleLossWeightScheduler(object):
    def __init__(self, simple_loss_weight_max=10., wait_steps=50000, ramp=100000) -> None:
        self.simple_loss_weight_max = simple_loss_weight_max
        self.wait_steps = wait_steps
        self.ramp = ramp
    
    def __call__(self, step):
        max_weight = self.simple_loss_weight_max - 1
        w = 1 + min(max_weight, max(0., max_weight*(step - self.wait_steps)/self.ramp))
        return w
