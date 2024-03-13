#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random

# 100x100 배열 생성
array = torch.zeros(100, 100)

# 10x10 그리드로 나누기
grid_size = 10
grids = array.unfold(0, grid_size, grid_size).unfold(1, grid_size, grid_size)

# 그리드의 개수 확인
num_grids = grids.shape[0] * grids.shape[1]

# 50%의 그리드를 마스킹
num_grids_to_mask = int(num_grids * 0.5)
grids_to_mask = random.sample(range(num_grids), num_grids_to_mask)

# 마스킹을 적용
for idx, grid_idx in enumerate(grids_to_mask):
    row_idx = grid_idx // grids.shape[1]
    col_idx = grid_idx % grids.shape[1]
    grids[row_idx, col_idx] = torch.ones_like(grids[row_idx, col_idx])

# 결과 확인
print(array)
breakpoint()
abcd = 1
