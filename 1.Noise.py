#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:43:36 2022

@author: theowalcot
"""

import torch

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.rand(n_samples, z_dim, device=device)
