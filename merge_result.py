# -*- coding: UTF-8 -*-
import pandas as pd

res1 = pd.read_csv('../submission_lgb.txt', names=['id'])
res2 = pd.read_csv('../submission_plus.txt', names=['id'])
res1_list = res1.id.tolist()
res2_list = res2.id.tolist()
res3_set = set()
for m_id in res1_list:
    res3_set.add(m_id)
for m_id in res2_list:
    res3_set.add(m_id)
res3 = pd.Series(list(res3_set))
res3.to_csv('../submission_merge.txt', header=None, index=False)