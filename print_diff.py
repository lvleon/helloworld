# -*- coding: UTF-8 -*-
import pandas as pd

res1 = pd.read_csv('../submission.txt', names=['id'])
res2 = pd.read_csv('../submission_plus.txt', names=['id'])
res1_list = res1.id.tolist()
res2_list = res2.id.tolist()
res3_list = []
for m_id in res2_list:
    if m_id not in res1_list:
        res3_list.append(m_id)
res3 = pd.Series(res3_list)
res3.to_csv('../submission_diff.txt', header=None, index=False)